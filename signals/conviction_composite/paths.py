"""Locate the data the conviction-composite scripts read and write.

Everything is rooted at ``BRIDGEWAY_DATA`` (see ``shared/config.py``); each
location can also be overridden with its own environment variable.

Inputs
    CONVICTION_HANDOFF_DIR  The 2026-09-16 ``long_history`` folder, holding
                            ``analysis/event_results.parquet`` and
                            ``market/{daily_<year>,etf_daily,cik_links}.parquet``.
    Universe                ``universe/biotech_universe_intervals.parquet``, the
                            same strict-GICS universe the other signals use.
    CRSP_EXTRA_DIR          Optional extra ``daily_<year>.parquet`` panels (e.g. the
                            GICS names missing from the handoff extract).

Outputs
    CONVICTION_OUTPUT_DIR   Crawled events for the full-GICS run. Row-level data
                            stay under ``BRIDGEWAY_DATA``, never in this repository.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SUBMISSION = Path(__file__).resolve().parents[2]
if str(SUBMISSION) not in sys.path:
    sys.path.insert(0, str(SUBMISSION))

from shared.config import DATA_ROOT  # noqa: E402


def _dir(env: str, default: Path) -> Path:
    """Return the directory named by environment variable ``env``, or ``default``."""
    return Path(os.environ.get(env, default)).expanduser().resolve()


HANDOFF = _dir("CONVICTION_HANDOFF_DIR", DATA_ROOT / "handoff_2026-09-16/long_history")
ANALYSIS = HANDOFF / "analysis"
MKT = HANDOFF / "market"
EVENTS = ANALYSIS / "event_results.parquet"
UNIVERSE = DATA_ROOT / "universe/biotech_universe_intervals.parquet"
OUTPUT = _dir("CONVICTION_OUTPUT_DIR", DATA_ROOT / "outputs/conviction")
