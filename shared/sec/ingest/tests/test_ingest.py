"""Offline tests using a fake EDGAR client, temporary databases and the real parser."""

from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from shared.sec.form4_parser.discovery import FilingRef
from shared.sec.form4_parser.engines import get_engine
from shared.sec.ingest.storage import COLUMNS, FilingStore
from shared.sec.ingest.updates import quarters, resolve_period, update_filings

FIXTURE = (Path(__file__).resolve().parents[2] / "form4_parser/tests/rnst_2026_form4.xml").read_bytes()


class FakeClient:
    """Serve submissions JSON, a master index and filing bytes without network."""

    def __init__(self, filings, filing_bytes=FIXTURE, error=None, index=b""):
        self.filings = filings            # {cik: [(accession, filing_date) or (accession, filing_date, form), ...]}
        self.filing_bytes = filing_bytes
        self.error = error                # exception raised when a filing is requested
        self.index = index
        self.requests = []

    def get_json(self, url, use_cache=False):
        cik = int(re.search(r"CIK(\d+)", url).group(1))
        if isinstance(self.filings[cik], Exception):
            raise self.filings[cik]
        rows = self.filings[cik]
        return {"filings": {"recent": {
            "form": [(row + ("4",))[2] for row in rows], "accessionNumber": [row[0] for row in rows],
            "filingDate": [row[1] for row in rows], "primaryDocument": ["form4.xml"] * len(rows)}}}

    def get_bytes(self, url, use_cache=True):
        if url.endswith("master.idx"):
            return self.index
        self.requests.append(url)
        if self.error:
            raise self.error
        return self.filing_bytes


JANUARY = {"start": "2025-01-01", "end": "2025-01-31"}
ONE = {9: [("0000000009-25-000001", "2025-01-03"), ("0000000009-24-000001", "2024-12-31")]}
TWO = {9: [("0000000009-25-000001", "2025-01-03"), ("0000000009-25-000002", "2025-01-04")]}


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = FilingStore(Path(self.directory.name) / "test.sqlite")
        self.ref = FilingRef(9, "0000000009-25-000001", "4", "2025-01-03", "", "", "form4.xml")

    def tearDown(self):
        self.store.connection.close()
        self.directory.cleanup()

    def sql(self, query):
        return self.store.connection.execute(query).fetchall()

    def test_periods(self):
        self.assertEqual(resolve_period("day", end="2025-01-03"), ("2025-01-03", "2025-01-03"))
        self.assertEqual(resolve_period("week", end="2025-01-03"), ("2024-12-28", "2025-01-03"))
        self.assertEqual(resolve_period("month", end="2024-03-31"), ("2024-03-01", "2024-03-31"))
        self.assertEqual(resolve_period(start="2003-01-01", end="2025-12-31"), ("2003-01-01", "2025-12-31"))
        for bad in ({}, {"period": "week", "start": "2025-01-01"}, {"period": "quarter"},
                    {"start": "2025-02-01", "end": "2025-01-01"}):
            with self.assertRaises(ValueError):
                resolve_period(**bad)
        self.assertEqual(quarters("2024-12-31", "2025-04-01"), [(2024, 4), (2025, 1), (2025, 2)])

    def test_rows_are_stored_and_read_back(self):
        summary = update_filings(self.store, FakeClient(ONE), ciks=[9], **JANUARY)
        self.assertEqual((summary["discovered"], summary["fetched"], summary["failed"]), (1, 1, 0))
        self.assertEqual(self.sql("SELECT accession, status FROM filings"), [("0000000009-25-000001", "done")])
        row = self.sql('SELECT issuer_ticker, transaction_code, shares, price_per_share, is_director, '
                       '"table", edgar_filing_date, filing_url, officer_title FROM ownership_rows')
        self.assertEqual(row[0][:7], ("RNST", "S", 35000, 43.88, 1, "non_derivative", "2025-01-03"))
        self.assertTrue(row[0][7].endswith("0000000009-25-000001-index.htm"))
        self.assertIsNone(row[0][8])          # blank text is stored as NULL

    def test_second_run_fetches_nothing(self):
        client = FakeClient(ONE)
        update_filings(self.store, client, ciks=[9], **JANUARY)
        requests = len(client.requests)
        summary = update_filings(self.store, client, ciks=[9], **JANUARY)
        self.assertEqual((summary["discovered"], summary["fetched"]), (1, 0))
        self.assertEqual(len(client.requests), requests)
        self.assertEqual(self.sql("SELECT COUNT(*) FROM ownership_rows"), [(1,)])

    def test_failed_filing_is_recorded_and_retried(self):
        summary = update_filings(self.store, FakeClient(ONE, error=OSError("timeout")), ciks=[9], **JANUARY)
        self.assertEqual((summary["fetched"], summary["failed"]), (0, 1))
        self.assertEqual(self.sql("SELECT status, attempts, error FROM filings"), [("failed", 1, "timeout")])   # the real cause is kept
        update_filings(self.store, FakeClient(ONE, error=OSError("timeout")), ciks=[9], **JANUARY)
        self.assertEqual(self.sql("SELECT status, attempts FROM filings"), [("failed", 2)])
        summary = update_filings(self.store, FakeClient(ONE), ciks=[9], **JANUARY)
        self.assertEqual(summary["fetched"], 1)
        self.assertEqual(self.sql("SELECT status, error FROM filings"), [("done", None)])
        self.assertEqual(self.sql("SELECT COUNT(*) FROM ownership_rows"), [(1,)])

    def test_access_rejection_stops_the_run(self):
        client = FakeClient(TWO, error=PermissionError("SEC access rejected (403)"))
        with self.assertRaises(PermissionError):
            update_filings(self.store, client, ciks=[9], **JANUARY)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(self.sql("SELECT COUNT(*) FROM filings"), [(0,)])
        client = FakeClient({1: PermissionError("403"), 2: []})
        with self.assertRaises(PermissionError):
            update_filings(self.store, client, ciks=[1, 2], **JANUARY)

    def test_problems_are_flagged_not_dropped(self):
        empty = b"<ownershipDocument><documentType>4</documentType></ownershipDocument>"
        for payload, status, kind in ((b"not a filing", "error", "unrecognized_format"), (empty, "empty", "no_rows")):
            self.store.connection.execute("DELETE FROM filings")
            self.store.connection.execute("DELETE FROM ownership_rows")
            summary = update_filings(self.store, FakeClient(ONE, filing_bytes=payload), ciks=[9], **JANUARY)
            self.assertEqual((summary["fetched"], summary["flagged"], summary["failed"]), (1, 1, 0))
            self.assertEqual(self.sql("SELECT parse_status, row_kind FROM ownership_rows"), [(status, kind)])
            summary = update_filings(self.store, FakeClient(ONE, filing_bytes=payload), ciks=[9], **JANUARY)
            self.assertEqual((summary["fetched"], summary["flagged"], summary["failed"]), (0, 0, 0))   # not retried
        broken = FakeClient({1: OSError("no such CIK"), 9: ONE[9]})
        summary = update_filings(self.store, broken, ciks=[1, 9], **JANUARY)
        self.assertEqual([e["target"] for e in summary["discovery_errors"]], ["1"])
        self.assertEqual(summary["discovered"], 1)

    def test_discovery_includes_amendments_and_period_bounds(self):
        filings = {9: [("0000000009-25-000001", "2025-01-01", "4/A"), ("0000000009-25-000002", "2025-01-31", "144/A"),
                       ("0000000009-25-000003", "2025-02-01", "5")],
                   7: [("0000000009-25-000001", "2025-01-01", "4/A")]}      # same filing under a second CIK
        summary = update_filings(self.store, FakeClient(filings), ciks=[7, 9], **JANUARY)
        self.assertEqual((summary["discovered"], summary["fetched"], summary["failed"]), (2, 2, 0))
        self.assertEqual(self.sql("SELECT form, filing_date FROM filings ORDER BY filing_date"),
                         [("4/A", "2025-01-01"), ("144/A", "2025-01-31")])

    def test_all_edgar_filers_for_a_period(self):
        index = (b"CIK|Company Name|Form Type|Date Filed|Filename\n"
                 b"11|Jane|4|2025-01-03|edgar/data/11/0000000009-25-000001.txt\n"
                 b"9|Example|4|2025-01-03|edgar/data/9/0000000009-25-000001.txt\n"
                 b"9|Example|4|2025-02-03|edgar/data/9/0000000009-25-000002.txt\n"
                 b"9|Example|10-K|2025-01-03|edgar/data/9/0000000009-25-000003.txt\n")
        summary = update_filings(self.store, FakeClient({}, index=index), **JANUARY)
        self.assertEqual((summary["discovered"], summary["fetched"]), (1, 1))
        self.assertEqual(self.sql("SELECT issuer_cik FROM ownership_rows"), [("0000715072",)])

    def test_every_parser_key_has_a_column(self):
        rows = get_engine("4", client=FakeClient({})).fetch_filing(self.ref)
        self.assertEqual(set(rows[0]), set(COLUMNS))

    def test_failed_replacement_is_retried_without_duplicate_rows(self):
        ref = self.ref
        rows = get_engine("4", client=FakeClient({})).fetch_filing(ref)
        self.store.save(ref, rows)
        self.store.save_failure(ref, "replacement failed")
        self.assertEqual(self.sql("SELECT status, attempts FROM filings"), [("failed", 1)])
        self.assertEqual(self.sql("SELECT shares FROM ownership_rows"), [(35000,)])
        rows[0]["shares"] = "123"
        self.store.save(ref, rows)
        self.store.save(ref, rows)
        self.assertEqual(self.sql("SELECT shares FROM ownership_rows"), [(123,)])
        self.assertEqual(self.sql("SELECT status, attempts, error FROM filings"), [("done", 1, None)])

    def test_failed_save_rolls_back_row_replacement(self):
        ref = self.ref
        self.store.save(ref, [{"accession": ref.accession, "owner_name": "original"}])
        self.store.connection.execute(
            "CREATE TRIGGER reject_bad_owner BEFORE INSERT ON ownership_rows "
            "WHEN NEW.owner_name = 'reject' BEGIN SELECT RAISE(ABORT, 'rejected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save(ref, [{"accession": ref.accession, "owner_name": "first replacement"},
                                  {"accession": ref.accession, "owner_name": "reject"}])
        self.assertEqual(self.sql("SELECT owner_name FROM ownership_rows"), [("original",)])
        self.assertEqual(self.sql("SELECT status FROM filings"), [("done",)])
        with self.assertRaisesRegex(ValueError, "Every row"):
            self.store.save(ref, [{"accession": "another-filing"}])
        self.assertEqual(self.sql("SELECT owner_name FROM ownership_rows"), [("original",)])

    def test_old_database_gains_new_parser_columns_without_losing_rows(self):
        path = Path(self.directory.name) / "old.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE ownership_rows (accession TEXT, shares NUMERIC)")
            connection.execute("INSERT INTO ownership_rows VALUES ('old', 7)")
        with FilingStore(path):
            pass                                    # migrate and close without saving anything
        with sqlite3.connect(path) as plain:        # the migration must have been committed
            self.assertEqual(plain.execute(
                "SELECT accession, shares, parse_warnings FROM ownership_rows").fetchall(), [("old", 7, None)])
        with FilingStore(path) as migrated:
            ref = FilingRef(9, "new", "4", "2025-01-03", "", "", "form4.xml")
            migrated.save(ref, [{"accession": "new", "parse_status": "needs_review",
                                 "parse_warnings": '["recovered_xml"]'}])
        with FilingStore(path) as reopened:
            self.assertEqual(reopened.connection.execute(
                "SELECT parse_status, parse_warnings FROM ownership_rows WHERE accession = 'new'").fetchall(),
                [("needs_review", '["recovered_xml"]')])

    def test_failed_schema_migration_rolls_back_all_new_columns_and_closes_connection(self):
        path = Path(self.directory.name) / "migration_failure.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE ownership_rows (accession TEXT, shares NUMERIC)")
            connection.execute("INSERT INTO ownership_rows VALUES ('old', 7)")
        alterations = []
        def authorize(action, first, second, database, source):
            if action == sqlite3.SQLITE_ALTER_TABLE:
                alterations.append(second)
                if len(alterations) == 2:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(authorize)
        with patch("shared.sec.ingest.storage.sqlite3.connect", return_value=connection):
            with self.assertRaises(sqlite3.DatabaseError):
                FilingStore(path)
        self.assertEqual(len(alterations), 2)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        with sqlite3.connect(path) as reopened:
            self.assertEqual([row[1] for row in reopened.execute("PRAGMA table_info(ownership_rows)")],
                             ["accession", "shares"])
            self.assertEqual(reopened.execute("SELECT * FROM ownership_rows").fetchall(), [("old", 7)])

    def test_unknown_parser_fields_are_rejected_before_replacing_stored_rows(self):
        ref = self.ref
        self.store.save(ref, [{"accession": ref.accession, "shares": "7"}])
        with self.assertRaisesRegex(ValueError, "Unknown parser fields: future_parser_field"):
            self.store.save(ref, [{"accession": ref.accession, "shares": "99", "future_parser_field": "keep me"}])
        self.assertEqual(self.sql("SELECT shares FROM ownership_rows"), [(7,)])
        self.assertEqual(self.sql("SELECT status, attempts, error FROM filings"), [("done", 0, None)])


if __name__ == "__main__":
    unittest.main()
