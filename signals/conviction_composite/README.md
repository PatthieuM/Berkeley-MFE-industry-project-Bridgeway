# Conviction composite

Scores each open-market insider purchase from 0 to 4 (small cap, large stake, breadth, seniority) and tests the score with a monthly walk-forward against XBI. `prepare_inputs.py` feeds the full-GICS run from the shared data layer.

**These scripts do not run from this branch alone.** Their main input, the SIC 2834/2836 event table, is produced by code that lives in the team repository, not here. The sections below list every input, where it comes from, and what can be rebuilt on this branch.

## Scripts

| Script | Purpose |
|---|---|
| `conviction_signal.py` | Conviction score and its train/test evaluation |
| `firm_month_signal.py` | Variant aggregated to firm-months, with a monthly-rebalanced long/short book |
| `backtest.py` | Calendar-time portfolio and robustness statistics |
| `crsp_reprice.py` | Prices the events on CRSP (with delisting returns) |
| `walkforward_conviction.py` | Walk-forward on the SIC 2834/2836 event set |
| `walkforward_gics_biotech.py` | Same test, keeping only events dated while the firm was in the GICS biotech universe |
| `walkforward_full_gics.py` | Same test on the full GICS biotech universe (handoff events plus the firms missing from it) |
| `scrape_gics_events.py` | EDGAR crawl of Forms 4 and 5 for the missing firms (network; needs `SEC_USER_AGENT`) |
| `prepare_inputs.py` | Offline alternative to the crawl and to a separate CRSP pull (see below) |
| `paths.py` | Input and output locations, all under `BRIDGEWAY_DATA` |

## Inputs

| Input | Used by | Source | On this branch? |
|---|---|---|---|
| `analysis/event_results.parquet` (SIC 2834/2836 events with roles, holding fraction, cluster flags) | all three walk-forwards | Team repository, `main`: the earlier 20-year study, `2026-09-16/research_long/` (`prepare.py`, `analyze.py`) | **No.** Neither the table nor the code that builds it |
| `market/daily_<year>.parquet`, `etf_daily.parquet`, `cik_links.parquet` for the SIC firms | all three walk-forwards | Same 2026-09-16 `long_history` folder | **No** |
| SIC 2834/2836 universe (759 firms) | defines the event table above | Same 2026-09-16 pipeline | **No.** `shared/universe/universe.py` builds the GICS universe; SIC is only its fallback when GICS is missing |
| `universe/biotech_universe_intervals.parquet` | GICS runs | `shared/universe/universe.py` | Yes |
| Events of the universe firms missing from the handoff table (`gics_missing_events.parquet`) | `walkforward_full_gics.py` | `scrape_gics_events.py` (crawl) or `prepare_inputs.py` (offline) | Yes |
| CRSP panels for those firms, with a `date` column | `walkforward_full_gics.py` via `CRSP_EXTRA_DIR` | Team repository, branch `task3-signals-crsp-debias`: `src/fetch_crsp_prices.py`; or `prepare_inputs.py` | Yes, through `prepare_inputs.py` |

Place the 2026-09-16 `long_history` folder at `<BRIDGEWAY_DATA>/handoff_2026-09-16/long_history`, or point `CONVICTION_HANDOFF_DIR` at it. It is licensed row-level data and is not in the SharePoint data package.

## `prepare_inputs.py`

Builds the two full-GICS inputs offline from the shared data layer:

1. `<CONVICTION_OUTPUT_DIR>/crsp_extra/daily_<year>.parquet`: the shared CRSP pull (`shared/market/pull_universe_prices.py`) with `dlycaldt` renamed to `date`.
2. `<CONVICTION_OUTPUT_DIR>/gics_missing_events.parquet`: the table the crawl would write, built from the normalized SEC quarterly data sets and restricted to dates on which the firm was in the biotech universe.

Differences from the crawl: original Form 4 filings only (no Form 5), and `signal_date` is the SEC filing date because the data sets carry no acceptance time.

```bash
python signals/conviction_composite/prepare_inputs.py
CRSP_EXTRA_DIR=<printed folder> python signals/conviction_composite/walkforward_full_gics.py
```

## Results and reproducibility

No result files are stored here. The presentation (`docs/presentation_2026-10-01.pdf`) reports three runs:

| Run | Firms | L/S alpha per year | t |
|---|---|---|---|
| SIC 2834/2836 | 759 | +12.8% | 2.80 |
| SIC and GICS intersection | 495 | +11.3% | 1.88 |
| Full GICS | 875 | +7.2% | 1.55 |

The first two need only the handoff folder. The full-GICS figures on the slides were produced with missing-firm events built as in `prepare_inputs.py` but **without** the universe-interval filter on the added events; about 40% of those added rows fell outside the firm's universe interval.

With the filter applied (`prepare_inputs.py` as committed, run on 2026-10-02): 824 firms, L/S alpha +8.6% per year, t = 1.59; high-conviction event-level mean +0.45% per quarter versus XBI, p = 0.89. The conclusion is unchanged: the score is not confirmed on the full GICS universe.

All figures are gross of trading costs.
