"""Legacy continuity exports and the corrected version 2 price check.

The default continuity function remains frozen for the published audit.
``check_ownership_continuity_v2`` is an explicit opt-in for a separate rerun.
"""

from __future__ import annotations

import pandas as pd

if __package__:
    from .src import quality_checks as _legacy
    from .src import quality_checks_v2 as _v2
else:  # scrape.py also supports direct execution from this directory.
    from src import quality_checks as _legacy
    from src import quality_checks_v2 as _v2

CONTINUITY_OUTPUT_COLUMNS = _legacy.CONTINUITY_OUTPUT_COLUMNS
PRICE_OUTPUT_COLUMNS = _v2.PRICE_OUTPUT_COLUMNS
check_ownership_continuity = _legacy.check_ownership_continuity
resolve_amendments = _legacy.resolve_amendments
check_ownership_continuity_v2 = _v2.check_ownership_continuity
resolve_amendments_v2 = _v2.resolve_amendments

CONT_OK = "PASS"
CONT_FLAG = "FLAG_REVIEW_UNADJUSTED"
CONT_FIRST = "BASELINE_NO_PRIOR"
CONT_NOT_TESTED = "NOT_TESTABLE_ROW_KIND"
PRICE_EXACT = "PASS_EXACT_DAY_RANGE"
PRICE_ADJACENT = "PASS_ADJACENT_DAY_RANGE"
PRICE_WEEKLY = "PASS_WEEKLY_RANGE"
PRICE_OUTSIDE = "FLAG_OUTSIDE_WEEKLY_RANGE"
PRICE_NONE = "NO_MARKET_DATA"
PRICE_NA = "NOT_MARKET_COMPARABLE"
CONTINUITY_COLUMNS = CONTINUITY_OUTPUT_COLUMNS
PRICE_COLUMNS = PRICE_OUTPUT_COLUMNS


def check_transaction_prices(df, market_data=None, *, price_lookup=None, **kwargs):
    """Version 2 price check, with the historical lookup callback as an adapter.

    ``price_status`` aliases ``price_check_status`` for the old CLI. Derived
    shares-times-price is named ``derived_transaction_value``, never reported.
    The callback receives a ticker and a calendar date. Its results, including
    missing observations, are cached once per security/date within this call.
    """
    if price_lookup is not None and market_data is not None:
        raise ValueError("Supply market_data or price_lookup, not both")
    if price_lookup is not None:
        observations = {}
        queried = set()
        span = kwargs.get("max_adjacent_calendar_days", 7)
        if span < 0:
            raise ValueError("max_adjacent_calendar_days must be nonnegative")
        records = df.to_dict("records")
        security_column = kwargs.get("transaction_security_col")
        if security_column:
            _v2._resolve_column(df, security_column, (), "transaction security")
        callback_securities = [
            _v2._value(row, security_column) if security_column
            else _v2._value(row, "issuer_ticker", "ticker")
            for row in records
        ]
        securities = [
            _v2._market_identifier(value, security_column) if security_column
            else _v2._normalized_text(value)
            for value in callback_securities
        ]
        for row, ticker, callback_security in zip(records, securities, callback_securities):
            day = _v2._date(row.get("transaction_date"))
            if not ticker or pd.isna(day):
                continue
            week_start = day - pd.Timedelta(days=day.weekday())
            start = min(day - pd.Timedelta(days=span), week_start)
            end = max(day + pd.Timedelta(days=span), week_start + pd.Timedelta(days=6))
            for query_day in pd.date_range(start, end):
                key = (ticker, query_day)
                if key not in queried:
                    queried.add(key)
                    observation = price_lookup(callback_security, query_day)
                    if observation is not None and len(observation):
                        observations[key] = {
                            **observation, "_lookup_security": ticker,
                            kwargs.get("market_date_col") or "date": query_day,
                        }
        df = df.assign(_lookup_security=securities)
        kwargs["transaction_security_col"] = "_lookup_security"
        kwargs["market_security_col"] = "_lookup_security"
        market_data = pd.DataFrame(observations.values())
    result = _v2.check_transaction_prices(df, market_data, **kwargs)
    result["price_status"] = result["price_check_status"]
    return result
