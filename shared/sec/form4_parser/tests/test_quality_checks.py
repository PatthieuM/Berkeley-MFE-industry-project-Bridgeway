"""Verify ownership and price sanity checks with synthetic offline tables."""

import sys
import unittest
from pathlib import Path

import pandas as pd

if __package__:
    from shared.sec.form4_parser import quality_checks as QC
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import quality_checks as QC


def _row(**kw):
    """Return a minimal non-derivative Form 4 row, with ``kw`` overriding any field."""
    base = {"table": "non_derivative", "owner_cik": "1", "issuer_cik": "9",
            "issuer_ticker": "T", "owner_name": "Jane", "security_title": "Common Stock",
            "direct_or_indirect": "D", "form": "4", "accession": "a", "filing_url": "",
            "source_format": "ownership_xml", "footnotes": "", "acceptance_datetime": "",
            "sequence_in_filing": "1"}
    base.update(kw)
    return base


class TestContinuity(unittest.TestCase):
    """Cover ownership-balance continuity and split adjustments."""

    def test_clean_sequence_ok(self):
        df = pd.DataFrame([
            _row(transaction_date="2026-01-01", acquired_disposed="A", shares="100", shares_owned_following="100"),
            _row(transaction_date="2026-02-01", acquired_disposed="D", shares="40", shares_owned_following="60"),
            _row(transaction_date="2026-03-01", acquired_disposed="A", shares="10", shares_owned_following="70"),
        ])
        res = QC.check_ownership_continuity(df)
        self.assertEqual(res.iloc[0]["continuity_status"], QC.CONT_FIRST)
        self.assertEqual(res.iloc[1]["continuity_status"], QC.CONT_OK)
        self.assertEqual(res.iloc[2]["continuity_status"], QC.CONT_OK)

    def test_broken_balance_flagged(self):
        # 100 - 40 should be 60, but filing reports 999 -> flag.
        df = pd.DataFrame([
            _row(transaction_date="2026-01-01", acquired_disposed="A", shares="100", shares_owned_following="100"),
            _row(transaction_date="2026-02-01", acquired_disposed="D", shares="40", shares_owned_following="999"),
        ])
        res = QC.check_ownership_continuity(df)
        self.assertEqual(res.iloc[1]["continuity_status"], QC.CONT_FLAG)
        self.assertEqual(res.iloc[1]["position_difference"], 939)

    def test_split_factor_applied(self):
        # 4:1 split: 100 -> 400, then -0 stays 400.
        df = pd.DataFrame([
            _row(transaction_date="2026-01-01", acquired_disposed="A", shares="100", shares_owned_following="100"),
            _row(transaction_date="2026-06-01", acquired_disposed="A", shares="0", shares_owned_following="400"),
        ])
        without = QC.check_ownership_continuity(df)
        self.assertEqual(without.iloc[1]["continuity_status"], QC.CONT_FLAG)  # unadjusted -> mismatch
        with_split = QC.check_ownership_continuity(df, pd.DataFrame([
            {"issuer_cik": "9", "effective_date": "2026-05-01", "split_factor": 4}]))
        self.assertEqual(with_split.iloc[1]["continuity_status"], QC.CONT_OK)


class TestPrice(unittest.TestCase):
    """Cover transaction-price checks against daily market ranges."""

    def _lookup(self, feed):
        """Wrap a {date: prices} dict as the (ticker, date) lookup the price check expects."""
        return lambda tkr, d: feed.get(pd.Timestamp(d.date()))

    def test_inside_and_outside(self):
        df = pd.DataFrame([
            {"issuer_ticker": "T", "owner_name": "x", "transaction_date": "2026-01-05",
             "transaction_code": "S", "shares": "10", "price_per_share": "50"},
            {"issuer_ticker": "T", "owner_name": "x", "transaction_date": "2026-01-05",
             "transaction_code": "S", "shares": "10", "price_per_share": "999"},
        ])
        feed = {pd.Timestamp("2026-01-05"): {"low": "49", "high": "51", "close": "50"}}
        res = QC.check_transaction_prices(df, price_lookup=self._lookup(feed))
        self.assertEqual(res.iloc[0]["price_status"], QC.PRICE_EXACT)
        self.assertEqual(res.iloc[1]["price_status"], QC.PRICE_OUTSIDE)
        self.assertEqual(res.iloc[0]["derived_transaction_value"], 500)
        self.assertTrue(pd.isna(res.iloc[0]["reported_transaction_value"]))

    def test_non_market_code_and_no_feed(self):
        df = pd.DataFrame([
            {"issuer_ticker": "T", "owner_name": "x", "transaction_date": "2026-01-05",
             "transaction_code": "M", "shares": "10", "price_per_share": "0"},
            {"issuer_ticker": "T", "owner_name": "x", "transaction_date": "2026-01-05",
             "transaction_code": "S", "shares": "10", "price_per_share": "50"},
        ])
        res = QC.check_transaction_prices(df, price_lookup=None)
        self.assertEqual(res.iloc[0]["price_status"], QC.PRICE_NA)
        self.assertEqual(res.iloc[1]["price_status"], QC.PRICE_NONE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
