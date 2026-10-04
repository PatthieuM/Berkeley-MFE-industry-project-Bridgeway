"""Test owner attribution, provenance, and discovery with synthetic filings."""

import unittest

from shared.sec.form4_parser.discovery import FilingRef, list_filings_for_ciks, list_index_filings, parse_master_index
from shared.sec.form4_parser.engines import Form4Engine
from shared.sec.form4_parser.formats import parse_filing


def ownership_xml(form="4", joint=True):
    """Return synthetic ownership XML for one transaction and optional co-owner."""
    second = """<reportingOwner><reportingOwnerId><rptOwnerCik>22</rptOwnerCik>
      <rptOwnerName>Example Family Trust</rptOwnerName></reportingOwnerId>
      <reportingOwnerRelationship><isDirector>0</isDirector><isOfficer>0</isOfficer>
      <isTenPercentOwner>1</isTenPercentOwner></reportingOwnerRelationship></reportingOwner>""" if joint else ""
    return f"""<ownershipDocument><schemaVersion>X0508</schemaVersion><documentType>{form}</documentType>
      <periodOfReport>2025-01-02</periodOfReport><dateOfOriginalSubmission>2025-01-03</dateOfOriginalSubmission>
      <issuer><issuerCik>9</issuerCik><issuerName>Example Biotech</issuerName><issuerTradingSymbol>EX</issuerTradingSymbol></issuer>
      <reportingOwner><reportingOwnerId><rptOwnerCik>11</rptOwnerCik><rptOwnerName>Jane Example</rptOwnerName></reportingOwnerId>
      <reportingOwnerRelationship><isDirector>1</isDirector><isOfficer>1</isOfficer><officerTitle>Chief Executive Officer</officerTitle>
      <isTenPercentOwner>0</isTenPercentOwner></reportingOwnerRelationship></reportingOwner>{second}
      <nonDerivativeTable><nonDerivativeTransaction><securityTitle><value>Class A Common Stock</value></securityTitle>
      <transactionDate><value>2025-01-02</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>3</value><footnoteId id="F1"/></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
      <postTransactionAmounts><sharesOwnedFollowingTransaction><value>100</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
      <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
      </nonDerivativeTransaction></nonDerivativeTable><footnotes><footnote id="F1">Weighted average price.</footnote></footnotes>
      <remarks>Only the price of one row is corrected.</remarks></ownershipDocument>""".encode()


class NormalizationTests(unittest.TestCase):
    """Verify canonical ownership rows and filing-level provenance."""
    def test_joint_owners_have_distinct_roles_shared_economic_id(self):
        rows, _, status = parse_filing(ownership_xml())
        self.assertEqual(status, "ok")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["owner_cik"] for r in rows], ["11", "22"])
        self.assertEqual([r["is_officer"] for r in rows], [True, False])
        self.assertEqual([r["officer_title"] for r in rows], ["Chief Executive Officer", ""])
        self.assertEqual(len({r["economic_row_id"] for r in rows}), 1)
        self.assertEqual(len({r["owner_row_id"] for r in rows}), 2)
        self.assertTrue(all(r["owner_attribution_status"] == "joint_owner_unallocated" for r in rows))
        self.assertTrue(all(r["is_natural_person"] is None for r in rows))
        economic = {r["economic_row_id"]: r for r in rows}
        self.assertEqual(sum(float(r["derived_transaction_value"]) for r in economic.values()), 30)
        self.assertTrue(all(r["reported_transaction_value"] == "" for r in rows))

    def test_submission_native_path_preserves_notes_and_eastern_time(self):
        envelope = b"<SEC-DOCUMENT>\n<ACCEPTANCE-DATETIME>20250103163000\n<DOCUMENT>\n<TYPE>4\n<TEXT>" + ownership_xml()
        rows, fmt, status = parse_filing(envelope)
        self.assertEqual((fmt, status), ("ownership_xml", "ok"))
        self.assertEqual(rows[0]["acceptance_datetime"], "2025-01-03T16:30:00-05:00")
        self.assertEqual(rows[0]["original_submission_date"], "2025-01-03")
        self.assertIn("Weighted average", rows[0]["footnotes"])
        self.assertEqual(rows[0]["remarks"], "Only the price of one row is corrected.")

    def test_derivative_holdings_preserve_form_and_kind(self):
        for form in ("3", "4", "5", "4/A"):
            payload = (
                ownership_xml(form, joint=False)
                .replace(b"nonDerivativeTransaction", b"derivativeHolding")
                .replace(b"nonDerivativeTable", b"derivativeTable")
            )
            rows, _, status = parse_filing(payload)
            self.assertEqual(status, "ok")
            self.assertEqual(rows[0]["form"], form)
            self.assertTrue(rows[0]["is_derivative"])
            self.assertEqual(rows[0]["row_kind"], "holding")

    def test_form144_is_proposal_and_embedded_xml_parses(self):
        payload = b"""<SEC-DOCUMENT><DOCUMENT><TYPE>144<TEXT><edgarSubmission><submissionType>144</submissionType>
        <issuerInfo><issuerCik>9</issuerCik><issuerName>Example</issuerName>
        <nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold>Jane Example</nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold></issuerInfo>
        <securitiesInformation><securitiesClassTitle>Common Stock</securitiesClassTitle><noOfUnitsSold>20</noOfUnitsSold>
        <aggregateMarketValue>60</aggregateMarketValue><approxSaleDate>01/04/2025</approxSaleDate></securitiesInformation></edgarSubmission>"""
        rows, fmt, status = parse_filing(payload)
        self.assertEqual((fmt, status), ("form144_xml", "ok"))
        self.assertEqual(rows[0]["row_kind"], "proposed_sale")
        self.assertEqual(rows[0]["sec144_aggregate_market_value"], "60")

    def test_engine_actual_source_and_ids_are_independent_of_source_bytes(self):
        ref = FilingRef(9, "0000000009-25-000001", "4", "2025-01-03", "2025-01-02", "", "primary.xml")
        class Client:
            """Serve only the primary-document fixture."""
            def get_bytes(self, url):
                """Return fixture XML after simulating a submission miss."""
                if url == ref.submission_url:
                    raise OSError("fixture source unavailable")
                return ownership_xml()
        rows = Form4Engine(Client()).fetch_filing(ref)
        self.assertEqual(rows[0]["source_url"], ref.document_url)
        self.assertEqual(rows[0]["xml_url"], ref.document_url)
        self.assertEqual(rows[0]["acceptance_source"], "missing")
        class SubmissionClient:
            """Serve a complete-submission fixture."""
            def get_bytes(self, url):
                """Return an ownership document with an acceptance header."""
                return b"<SEC-DOCUMENT><ACCEPTANCE-DATETIME>20250103163000<DOCUMENT><TYPE>4<TEXT>" + ownership_xml()
        full = Form4Engine(SubmissionClient()).fetch_filing(ref)
        self.assertEqual(full[0]["source_url"], ref.submission_url)
        self.assertEqual(full[0]["owner_row_id"], rows[0]["owner_row_id"])

    def test_unrecognized_filing_still_has_error_row(self):
        ref = FilingRef(9, "0000000009-25-000001", "4", "2025-01-03", "", "", "bad.txt")
        class Client:
            """Serve an unrecognized filing fixture."""
            def get_bytes(self, url):
                """Return bytes that do not match a filing format."""
                return b"not a filing"
        rows = Form4Engine(Client()).fetch_filing(ref)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["parse_status"], "error")


class DiscoveryTests(unittest.TestCase):
    """Verify master-index and submissions-history discovery behavior."""
    INDEX = b"""Description header
CIK|Company Name|Form Type|Date Filed|Filename
11|Jane|4/A|2025-01-03|edgar/data/11/0000000009-25-000001.txt
9|Example|8-K|2025-01-03|edgar/data/9/0000000009-25-000002.txt
9|Example|10-K|2025-01-03|edgar/data/9/0000000009-25-000003.txt
"""

    def test_universal_index_forms_and_original_archive_path(self):
        refs = parse_master_index(self.INDEX, forms=("4", "8-K"))
        self.assertEqual([r.form for r in refs], ["4/A", "8-K"])
        self.assertEqual(refs[0].cik, 11)
        self.assertEqual(refs[0].submission_url, "https://www.sec.gov/Archives/edgar/data/11/0000000009-25-000001.txt")
        class Client:
            """Serve a master-index fixture without caching."""
            def get_bytes(inner, url, use_cache=True):
                """Validate the daily-index URL and return its fixture."""
                self.assertFalse(use_cache)
                self.assertIn("daily-index/2025/QTR1/master.20250103.idx", url)
                return self.INDEX
        self.assertEqual(len(list_index_filings(Client(), year=2025, day="2025-01-03")), 1)

    def test_cik_history_shards_deduplicated_and_date_bounded(self):
        recent = {
            "form": ["4"],
            "accessionNumber": ["0000000009-25-000001"],
            "filingDate": ["2025-01-03"],
            "primaryDocument": ["a.xml"],
        }
        older = {
            "form": ["4", "4"],
            "accessionNumber": ["0000000009-25-000001", "0000000009-24-000001"],
            "filingDate": ["2025-01-03", "2024-01-03"],
        }
        class Client:
            """Serve recent and historical submissions fixtures."""
            def get_json(self, url):
                """Return the history shard or its containing submissions page."""
                return (
                    older
                    if url.endswith("older.json")
                    else {"filings": {"recent": recent, "files": [{"name": "older.json"}]}}
                )
        refs = list_filings_for_ciks(Client(), [9, "0000000009"], start_date="2025-01-01", end_date="2025-12-31")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].accession, "0000000009-25-000001")


if __name__ == "__main__":
    unittest.main()
