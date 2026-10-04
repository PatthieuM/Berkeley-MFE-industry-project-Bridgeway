"""Define shared pipeline conventions and paths to SEC, universe, market, and result files.

Downstream event construction and evaluation modules consume ``Config`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Tuple

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
# Default layout: <project>/code (this repository) next to <project>/data.
DEFAULT_DATA_ROOT = SUBMISSION_DIR.parent / "data"
DATA_ROOT = Path(os.environ.get("BRIDGEWAY_DATA", DEFAULT_DATA_ROOT)).expanduser().resolve()


@dataclass(frozen=True)
class Config:
    """Store immutable research conventions and paths rooted at ``BRIDGEWAY_DATA``."""

    workspace: Path = DATA_ROOT
    project_dir: Path = SUBMISSION_DIR / "signals/smallcap_purchases"
    results_dir: Path = SUBMISSION_DIR / "signals/smallcap_purchases/results"
    universe_path: Path = DATA_ROOT / "universe/biotech_universe_intervals.parquet"
    crosswalk_path: Path = DATA_ROOT / "universe/identifier_crosswalk.parquet"
    universe_manifest_path: Path = DATA_ROOT / "universe/manifest.json"
    normalized_sec_dir: Path = DATA_ROOT / "sec/normalized"
    v6_market_dir: Path = DATA_ROOT / "crsp/market"
    universe_price_dir: Path = DATA_ROOT / "crsp/daily"
    study_start: str = "2009-01-01"
    study_end: str = "2025-12-31"
    breakpoint_history_start: str = "2006-01-01"
    transaction_date_end: str = "2026-06-30"
    horizons: Tuple[int, ...] = (1, 2, 3, 5, 10, 21, 42, 63, 126)
    path_horizon: int = 63
    benchmark: str = "XBI"
    entry_rule: str = "close of first exchange session strictly after filing date"
    small_cap_rule: str = "lagged-cap lower tercile; year Y cutoff uses purchase events filed in prior years only"
    turnover_sessions: int = 60
    turnover_min_valid: int = 40
    clustering: Tuple[str, str] = ("issuer_cik", "filing_month")
    price_source: str = "universe_full"
    periods: Tuple[Tuple[str, int, int], ...] = (
        ("2009-2025", 2009, 2025),
        ("2009-2019", 2009, 2019),
        ("2020-2025", 2020, 2025),
    )

    def with_price_source(self, source: str) -> "Config":
        """Return a copy configured for a validated daily-price source."""
        if source not in {"universe_full", "v6_cache"}:
            raise ValueError("price_source must be 'universe_full' or 'v6_cache'")
        return replace(self, price_source=source)


DEFAULT_CONFIG = Config()
