"""Pull CRSP daily data for every PERMNO in the validated biotech universe (2003-2025).

The script reads the universe intervals, writes annual stock Parquet files plus XBI
and calendar files under ``BRIDGEWAY_DATA``, and records a JSON coverage report.

Run from the submission root:
    python shared/market/pull_universe_prices.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

SUBMISSION = Path(__file__).resolve().parents[2]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

from shared.config import DATA_ROOT
from shared.universe.wrds_cloud import open_wrds_connection

UNIVERSE = DATA_ROOT / "universe/biotech_universe_intervals.parquet"
WANTED = ["permno", "permco", "dlycaldt", "ticker", "dlyret", "dlyretx", "dlyprc", "dlyclose", "dlyopen",
          "dlyhigh", "dlylow", "dlyvol", "shrout", "dlycap", "dlydelflg", "dlyprevprc", "dlyprevcap",
          "dlyretmissflg", "dlycumfacpr", "dlycumfacshr"]
OUT = DATA_ROOT / "crsp/daily"
MARKET_OUT = DATA_ROOT / "crsp/market"


def main() -> int:
    """Query WRDS for universe and XBI history, write files, and report coverage."""
    permnos = sorted(pd.read_parquet(UNIVERSE, columns=["permno"]).permno.dropna().astype(int).unique().tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"permnos_requested": len(permnos), "years": {}}
    with open_wrds_connection() as connection:
        cols = pd.read_sql(
            "SELECT column_name FROM information_schema.columns WHERE table_schema='crsp' AND table_name='dsf_v2'",
            connection,
        ).column_name.str.lower().tolist()
        select = [c for c in WANTED if c in cols]
        report["columns"] = select
        for year in range(2003, 2026):
            frame = pd.read_sql(
                f"SELECT {', '.join(select)} FROM crsp.dsf_v2 "
                "WHERE permno = ANY(%(p)s) AND dlycaldt BETWEEN %(a)s AND %(b)s",
                connection,
                params={"p": permnos, "a": f"{year}-01-01", "b": f"{year}-12-31"},
            )
            frame.to_parquet(OUT / f"daily_{year}.parquet", index=False)
            report["years"][year] = {"rows": len(frame), "permnos": int(frame.permno.nunique())}
            print(f"{year}: {len(frame):,} rows, {frame.permno.nunique()} securities", flush=True)
        xbi = pd.read_sql(
            "SELECT permno, dlycaldt, ticker, dlyret, dlyclose FROM crsp.dsf_v2 "
            "WHERE upper(ticker)='XBI' AND dlycaldt BETWEEN '2003-01-01' AND '2025-12-31' "
            "ORDER BY dlycaldt",
            connection,
        )
        if xbi.empty:
            raise RuntimeError("CRSP dsf_v2 returned no XBI rows")
        xbi.to_parquet(MARKET_OUT / "xbi_daily.parquet", index=False)
        pd.DataFrame({"date": pd.to_datetime(xbi["dlycaldt"]).drop_duplicates().sort_values()}).to_parquet(
            MARKET_OUT / "trading_calendar.parquet", index=False
        )
    report["permnos_found"] = int(
        pd.concat(
            pd.read_parquet(p, columns=["permno"]) for p in OUT.glob("daily_*.parquet")
        ).permno.nunique()
    )
    report["xbi_rows"] = len(xbi)
    (MARKET_OUT / "universe_prices_report.json").write_text(json.dumps(report, indent=2))
    print(f"Done: {report['permnos_found']} of {len(permnos)} securities found.")
    return 0


if __name__ == "__main__":
    import argparse

    # Parsing first means ``--help`` prints usage instead of opening a WRDS session.
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
