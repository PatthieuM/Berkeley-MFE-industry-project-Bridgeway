"""Exercise canonical parser semantics and offline error handling."""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
if not __package__:
    sys.path.insert(0, str(HERE.parents[3]))

from shared.sec.form4_parser import formats as F
from shared.sec.form4_parser import engines as E
from shared.sec.form4_parser import client as C
from shared.sec.form4_parser.discovery import FilingRef, list_insider_filings, list_index_filings
from shared.sec.form4_parser.scrape import _summary


class TestFormats(unittest.TestCase):
    """Check economic rows and metadata across the preserved SEC fixtures."""

    def _parse(self, filename):
        path = HERE / filename
        return F.parse_filing(path.read_bytes(), path.name)

    def test_modern_xml(self):
        rows, fmt, status = self._parse("rnst_2026_form4.xml")
        self.assertEqual((fmt, status), (F.FMT_XML, F.STATUS_OK))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["transaction_code"], "S")
        self.assertEqual(rows[0]["shares"], "35000")
        self.assertEqual(rows[0]["transaction_date"], "2026-08-03")
        self.assertEqual(rows[0]["shares_owned_following"], "211458")
        self.assertEqual(rows[0]["row_kind"], "transaction")
        self.assertEqual(rows[0]["parse_warnings"], "")

    def test_rendered_html_matches_xml(self):
        xml_rows, _, _ = self._parse("rnst_2026_form4.xml")
        html_rows, fmt, status = self._parse("rnst_2026_form4_rendered.html")
        self.assertEqual((fmt, status), (F.FMT_RENDERED_HTML, F.STATUS_OK_LEGACY_TABLE))
        self.assertEqual(len(html_rows), len(xml_rows))
        for key in ("issuer_cik", "issuer_ticker", "owner_cik", "owner_name", "is_director",
                    "table", "row_kind", "security_title", "transaction_code", "transaction_date",
                    "shares", "price_per_share", "acquired_disposed", "shares_owned_following",
                    "direct_or_indirect"):
            self.assertEqual(html_rows[0][key], xml_rows[0][key], key)

    def test_legacy_text_and_bare_text(self):
        payload = (HERE / "rnst_2003_form4_legacy.txt").read_bytes()
        bare = payload.split(b"<TEXT>", 1)[1].split(b"</TEXT>", 1)[0]
        for content, expected_format in ((payload, F.FMT_SUBMISSION), (bare, F.FMT_LEGACY_TXT)):
            rows, fmt, status = F.parse_filing(content)
            self.assertEqual((fmt, status), (expected_format, F.STATUS_OK_LEGACY_TABLE))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual((row["table"], row["security_title"]), ("derivative", "Phantom Stock"))
            self.assertEqual((row["shares"], row["price_per_share"]), ("12.27", "40.75"))
            self.assertEqual(row["footnote_ids"], "F1;F2")
            self.assertIn("deferred compensation plan", row["footnotes"])
            self.assertEqual(set(json.loads(row["filing_footnotes"])), {"1", "2"})

    def test_pipe_table_retains_transaction_and_holding(self):
        rows, _, status = self._parse("rnst_2003_form4_pipe_table.txt")
        self.assertEqual(status, F.STATUS_OK_LEGACY_TABLE)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["row_kind"] for r in rows], ["transaction", "holding"])
        self.assertEqual((rows[0]["transaction_code"], rows[0]["shares"]), ("G", "200"))
        self.assertEqual((rows[1]["shares_owned_following"], rows[1]["direct_or_indirect"]), ("5772", "I"))
        self.assertEqual(rows[1]["nature_of_ownership"], "Spouse")

    def test_wrapped_derivative_rows_and_officer_title(self):
        rows, _, status = self._parse("rnst_2003_form4_wrapped_rows.txt")
        self.assertEqual(status, F.STATUS_OK_LEGACY_TABLE)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["transaction_date"] for r in rows], ["2002-01-01", "2003-01-01"])
        self.assertTrue(all(r["security_title"] == "Employee Stock Option (Right to buy)" for r in rows))
        self.assertTrue(all(r["shares"] == "3500" for r in rows))
        self.assertTrue(all(r["officer_title"] == "Executive Vice President" for r in rows))
        self.assertEqual(rows[1]["shares_owned_following"], "7000")
        self.assertIn("ten (10) years", rows[0]["footnotes"])

    def test_wrapped_non_derivative_title_is_one_row(self):
        payload = (HERE / "rnst_2003_form4_pipe_table.txt").read_text().replace(
            "|Common Stock       | 01/03/2003 |",
            "|Common             |            |\n|Stock              | 01/03/2003 |", 1)
        rows, _, status = F.parse_filing(payload.encode())
        self.assertEqual(status, F.STATUS_OK_LEGACY_TABLE)
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["security_title"], rows[0]["shares"]), ("Common Stock", "200"))
        self.assertEqual(rows[1]["row_kind"], "holding")

    def test_submission_metadata_and_core_rows(self):
        cases = (
            ("rnst_2003_complete_submission.txt", F.FMT_SUBMISSION, F.STATUS_OK_LEGACY_TABLE,
             "0001192028", "2003-01-03T13:48:14-05:00", "12.27"),
            ("rnst_2026_submission.txt", F.FMT_XML, F.STATUS_OK,
             "0001192026", "2026-08-03T16:14:51-04:00", "35000"),
        )
        for filename, expected_fmt, expected_status, owner, acceptance, shares in cases:
            rows, fmt, status = self._parse(filename)
            self.assertEqual((fmt, status), (expected_fmt, expected_status))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["owner_cik"], owner)
            self.assertEqual(rows[0]["acceptance_datetime"], acceptance)
            self.assertEqual(rows[0]["shares"], shares)

    def test_canonical_schema_keys(self):
        for path in HERE.glob("rnst_*"):
            rows, _, _ = F.parse_filing(path.read_bytes(), path.name)
            for row in rows:
                self.assertEqual(set(row), set(F.CANON_FIELDS), path.name)

    def test_html_breaks_footnotes_and_officer_title(self):
        payload = (HERE / "rnst_2026_form4_rendered.html").read_text()
        payload = payload.replace(">Common Stock</span>", ">Common<br>Stock</span>")
        payload = payload.replace(">43.88</span>", ">43.88</span><sup>1</sup>")
        payload = payload.replace("<b>Explanation of Responses:</b></td></tr>",
                                  "<b>Explanation of Responses:</b></td></tr><tr><td>1.</td><td>Weighted average price.</td></tr>")
        payload = payload.replace('<tr><td align="center"></td><td class="MedSmallFormText">Officer',
                                  '<tr><td align="center"><span class="FormData">X</span></td><td class="MedSmallFormText">Officer')
        payload = payload.replace('<td width="35%" align="left" style="color: blue"></td>',
                                  '<td width="35%" align="left" style="color: blue">Chief Executive Officer</td>', 1)
        rows, _, status = F.parse_filing(payload.encode())
        self.assertEqual(status, F.STATUS_OK_LEGACY_TABLE)
        self.assertEqual(rows[0]["security_title"], "Common Stock")
        self.assertEqual(rows[0]["price_per_share"], "43.88")
        self.assertEqual(rows[0]["footnote_ids"], "F1")
        self.assertEqual(rows[0]["footnotes"], "F1: Weighted average price.")
        self.assertEqual(rows[0]["officer_title"], "Chief Executive Officer")
        self.assertTrue(rows[0]["is_officer"])

    def test_missing_rendered_footnote_requires_review(self):
        payload = (HERE / "rnst_2026_form4_rendered.html").read_bytes().replace(
            b">43.88</span>", b">43.88</span><sup>(1)</sup>")
        rows, _, status = F.parse_filing(payload)
        self.assertEqual(status, F.STATUS_REVIEW)
        self.assertIn("Referenced footnote F1", rows[0]["parse_warnings"])

    def test_partial_html_rows_require_review(self):
        payload = (HERE / "rnst_2026_form4_rendered.html").read_bytes().replace(
            b"</tbody>", b"<tr><td>Unparsed security</td><td>100</td></tr></tbody>", 1)
        rows, _, status = F.parse_filing(payload)
        self.assertEqual(status, F.STATUS_REVIEW)
        self.assertEqual(len(rows), 1)
        self.assertIn("fewer than 11 cells", rows[0]["parse_warnings"])

    @unittest.skipIf(F.etree is None, "XML recovery requires lxml")
    def test_xml_recovery_cannot_claim_complete_success(self):
        payload = (HERE / "rnst_2026_form4.xml").read_bytes()
        payload = payload.replace(b"</nonDerivativeTransaction>", b"", 1).replace(
            b"</nonDerivativeTable>", b"<nonDerivativeTransaction><securityTitle><value>Second row</value></securityTitle></nonDerivativeTransaction></nonDerivativeTable>")
        rows, _, status = F.parse_filing(payload)
        self.assertEqual(status, F.STATUS_REVIEW)
        self.assertTrue(rows)
        self.assertIn("Malformed XML was recovered", json.loads(rows[0]["parse_warnings"])[0])
        self.assertGreater(len(json.loads(rows[0]["parse_warnings"])), 1)

    def test_stdlib_xml_fallback_is_strict(self):
        import xml.etree.ElementTree as stdlib_etree
        with patch.object(F, "etree", None), patch.object(F, "_stdlib_etree", stdlib_etree, create=True):
            rows, _, status = self._parse("rnst_2026_form4.xml")
            self.assertEqual(status, F.STATUS_OK)
            self.assertEqual(rows[0]["shares"], "35000")
            rows, _, status = F.parse_filing(b"<ownershipDocument><broken></ownershipDocument>")
            self.assertEqual(status, F.STATUS_ERROR)
            self.assertTrue(rows[0]["parse_warnings"])

    def test_empty_ownership_retains_metadata(self):
        payload = b"""<ownershipDocument><documentType>3</documentType><issuer><issuerCik>9</issuerCik></issuer>
        <reportingOwner><reportingOwnerId><rptOwnerCik>11</rptOwnerCik><rptOwnerName>Jane</rptOwnerName></reportingOwnerId></reportingOwner>
        <remarks>No securities owned.</remarks></ownershipDocument>"""
        rows, _, status = F.parse_filing(payload)
        self.assertEqual(status, F.STATUS_EMPTY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), set(F.CANON_FIELDS))
        self.assertEqual((rows[0]["issuer_cik"], rows[0]["owner_cik"]), ("9", "11"))
        self.assertEqual(rows[0]["row_kind"], "no_rows")
        self.assertEqual(rows[0]["remarks"], "No securities owned.")

    def test_form144_amendment_and_empty_metadata(self):
        for security in (b"", b"<securitiesInformation><noOfUnitsSold>20</noOfUnitsSold></securitiesInformation>"):
            payload = b"""<edgarSubmission><submissionType>144/A</submissionType><issuerInfo><issuerCik>9</issuerCik>
            <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Jane</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold></issuerInfo>""" + security + b"</edgarSubmission>"
            rows, fmt, status = F.parse_filing(payload)
            self.assertEqual(fmt, F.FMT_144)
            self.assertEqual(status, F.STATUS_OK if security else F.STATUS_EMPTY)
            self.assertEqual((rows[0]["form"], rows[0]["document_type"]), ("144/A", "144/A"))
            self.assertEqual((rows[0]["owner_name"], rows[0]["issuer_cik"]), ("Jane", "9"))

    def test_pdf_is_unsupported_without_assuming_a_scan(self):
        rows, fmt, status = F.parse_filing(b"%PDF-1.4 text or images")
        self.assertEqual((fmt, status), ("pdf", "unsupported_pdf"))
        self.assertEqual(rows[0]["row_kind"], "unsupported_pdf")
        self.assertIn("text extraction or OCR", rows[0]["parse_warnings"])


class TestEngineFailures(unittest.TestCase):
    """Exercise engine fallbacks and summary reporting without HTTP."""

    def setUp(self):
        self.ref = FilingRef(715072, "0001192026-26-000006", "4", "2026-08-03", "2026-08-03", "", "primary.xml")
        self.xml = (HERE / "rnst_2026_form4.xml").read_bytes()

    def test_mixed_filing_has_no_no_transaction_flag(self):
        payload = self.xml.replace(b"</nonDerivativeTable>", b"""<nonDerivativeHolding><securityTitle><value>Other Class</value></securityTitle>
        <postTransactionAmounts><sharesOwnedFollowingTransaction><value>10</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
        </nonDerivativeHolding></nonDerivativeTable>""")
        client = Mock()
        client.get_bytes.return_value = payload
        rows = E.Form4Engine(client).fetch_filing(self.ref)
        self.assertEqual([r["row_kind"] for r in rows], ["transaction", "holding"])
        self.assertTrue(all(not r["form_flag"] for r in rows))

    def test_empty_filing_preserves_header_and_can_be_summarized(self):
        payload = self.xml[:self.xml.index(b"    <nonDerivativeTable>")] + b"</ownershipDocument>"
        client = Mock()
        client.get_bytes.return_value = payload
        engine = E.Form4Engine(client)
        rows = engine.fetch_filing(self.ref)
        self.assertEqual(rows[0]["owner_cik"], "0001192026")
        self.assertEqual(rows[0]["parse_status"], F.STATUS_EMPTY)
        self.assertEqual(rows[0]["form_flag"], "no_transaction_on_form4")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            _summary("4", engine.to_dataframe(rows))
        self.assertIn("insiders=1", output.getvalue())

    @unittest.skipIf(F.etree is None, "XML recovery requires lxml")
    def test_valid_primary_replaces_recovered_submission(self):
        client = Mock()
        client.get_bytes.side_effect = [self.xml.replace(b"</nonDerivativeTransaction>", b"", 1), self.xml]
        rows = E.Form4Engine(client).fetch_filing(self.ref)
        self.assertEqual(rows[0]["parse_status"], F.STATUS_OK)
        self.assertEqual(rows[0]["source_url"], self.ref.document_url)
        self.assertEqual(rows[0]["parse_warnings"], "")

    @unittest.skipIf(F.etree is None, "XML recovery requires lxml")
    def test_failed_primary_cannot_replace_recovered_submission(self):
        client = Mock()
        client.get_bytes.side_effect = [self.xml.replace(b"</nonDerivativeTransaction>", b"", 1), b"unrecognized"]
        rows = E.Form4Engine(client).fetch_filing(self.ref)
        self.assertEqual(rows[0]["parse_status"], F.STATUS_REVIEW)
        self.assertEqual(rows[0]["source_url"], self.ref.submission_url)
        self.assertEqual(rows[0]["shares"], "35000")

    def test_fetch_failure_keeps_canonical_provenance(self):
        client = Mock()
        client.get_bytes.side_effect = OSError("fixture unavailable")
        engine = E.Form4Engine(client)
        with patch.object(E, "list_insider_filings", return_value=[self.ref]):
            rows = engine.scrape_rows(715072)
        self.assertEqual(len(rows), 1)
        self.assertTrue(set(F.CANON_FIELDS).issubset(rows[0]))
        self.assertEqual(rows[0]["row_kind"], "fetch_error")
        self.assertEqual(rows[0]["submission_url"], self.ref.submission_url)
        self.assertTrue(rows[0]["owner_row_id"])
        self.assertTrue(rows[0]["parse_warnings"])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            _summary("4", engine.to_dataframe(rows))
        self.assertIn("insiders=0", output.getvalue())
        self.assertIn("error", output.getvalue())

    def test_summary_handles_historical_sparse_stubs(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            _summary("4", pd.DataFrame([{"accession": "a", "parse_status": "error"}]))
        self.assertIn("insiders=0", output.getvalue())
        self.assertIn("unknown", output.getvalue())


class TestOfflineClient(unittest.TestCase):
    """Verify cache and retry policies using fake archives and HTTP sessions."""

    def setUp(self):
        self.identity = patch.dict("os.environ", {"SEC_USER_AGENT": "Offline Tests tests@example.com"})
        self.identity.start()
        self.addCleanup(self.identity.stop)

    def test_offline_discovery_replays_refresh_requests(self):
        client = C.EdgarClient(offline=True)
        client.session.get = Mock(side_effect=AssertionError("No network requests allowed"))
        client.archive = Mock()
        client.archive.cached.return_value = json.dumps({"filings": {"recent": {
            "form": ["4"], "accessionNumber": ["0000000009-25-000001"],
            "filingDate": ["2025-01-03"], "primaryDocument": ["a.xml"]}}}).encode()
        self.assertEqual(len(list_insider_filings(client, 9)), 1)
        client.archive.cached.return_value = b"9|Example|4|2025-01-03|edgar/data/9/0000000009-25-000001.txt"
        self.assertEqual(len(list_index_filings(client, year=2025, quarter=1)), 1)
        client.session.get.assert_not_called()
        client.archive.cached.return_value = None
        with self.assertRaises(FileNotFoundError):
            client.get_json("https://data.sec.gov/missing.json")

    def test_final_http_attempt_does_not_sleep(self):
        response = requests.Response()
        response.status_code = 429
        for failure in (response, requests.ConnectionError("offline test")):
            client = C.EdgarClient(max_retries=1)
            client.session.get = Mock(side_effect=failure) if isinstance(failure, Exception) else Mock(return_value=failure)
            with patch.object(client, "_throttle"), patch.object(C.time, "sleep") as sleep:
                with self.assertRaises(RuntimeError):
                    client.get_bytes("https://example.test/filing")
                sleep.assert_not_called()

    def test_retry_wait_occurs_only_between_attempts(self):
        client = C.EdgarClient(max_retries=2)
        client.session.get = Mock(side_effect=requests.ConnectionError("offline test"))
        with patch.object(client, "_throttle"), patch.object(C.time, "sleep") as sleep:
            with self.assertRaises(RuntimeError):
                client.get_bytes("https://example.test/filing")
            sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
