# Berkeley MFE industry project - Bridgeway

A seven-week team project of the Berkeley MFE program with Bridgeway Capital. The question: do public insider filings (SEC Forms 3, 4 and 5) predict the returns of U.S. biotech stocks?

The repository holds the full pipeline, from SEC filings to tested signals. It contains code, documentation and aggregate results only; licensed CRSP/Compustat data and row-level SEC data are not included.

Presentation slides (October 1, 2026): [`docs/presentation_2026-10-01.pdf`](docs/presentation_2026-10-01.pdf).

## Repository layout

```
README.md, INSTALL.md       this overview; conda setup step by step
environment.yml             conda environment (requirements.txt for pip)
docs/                       slides, methodology, data dictionary, decisions, research log
results/                    aggregate summaries (SUMMARY.md and CSV tables)
shared/                     the common pipeline
  config.py                   every data path, in one place
  sec/                        step 1: SEC insider filings
  universe/                   step 2: point-in-time biotech universe
  market/                     step 3: CRSP prices
  events/                     step 4: events
  engine/                     steps 4 and 5: returns, statistics, quality checks
  tests/                      step 7: test suite
signals/                    step 6: the four signals of the presentation
  multi_insider_purchases/
  smallcap_purchases/
  seller_silence/
  conviction_composite/
```

## The pipeline, step by step

### 1. Collect SEC insider filings (`shared/sec/`)

- `acquire_normalize.py` downloads the SEC quarterly Insider Transactions Data Sets (Forms 3, 4, 5; 2006Q1 onward) and normalizes their seven tables: submissions, reporting owners, non-derivative and derivative transactions and holdings, footnotes.
- `form4_parser/` reads filings one by one from EDGAR: Forms 3, 4, 5 and 144, in XML, legacy HTML and plain-text formats. Unreadable filings are kept with an error status, never dropped silently.
- `ingest/` fetches a day, week, month or date range of filings for chosen stocks, the universe or every filer, and stores them in a SQLite database.
- `form144.py` discovers and parses Form 144. The parser is tested; Form 144 filings have not been collected.

### 2. Build the universe (`shared/universe/`)

`universe.py` builds a point-in-time universe of U.S. biotech common stocks from CRSP, Compustat and their link table: GICS 35201010, with SIC 2836 as a fallback when GICS is missing; NYSE, NYSE American and Nasdaq; firms that later delisted are kept for the period they were eligible. The result is 953 securities, 2003–2025. `universe_v5.py` is a revised builder; see `shared/universe/README.md`.

### 3. Pull prices (`shared/market/`)

`pull_universe_prices.py` pulls CRSP daily data through WRDS for every universe security, 2003–2025: total returns including delisting returns, daily low and high, market capitalization, split factors, plus the XBI benchmark and the trading calendar.

### 4. Build events and returns (`shared/events/`, `shared/engine/evaluate.py`)

The information date is the SEC filing date, and entry is the close of the next exchange session. Returns are measured in excess of a benchmark over the same sessions, at 1, 2, 3, 5, 10, 21, 42, 63 and 126 sessions. Standard errors are clustered by company and by filing month.

### 5. Check data quality (`shared/engine/`)

Checks flag rows; they never delete or correct them.

- `continuity.py` audits whether an owner's consecutive filings chain together: previous position + acquired − disposed = reported position, adjusted for splits. On 961,584 rows of the biotech universe, 68% of comparable non-derivative rows reconcile exactly. A break means a missing link (shares held through a trust, option exercises reported in the derivative table, gifts filed later on Form 5), not a wrong trade. Breakdown: `results/continuity_summary.csv`.
- `price_check.py` compares reported trade prices with CRSP. Across all universe trades, 2006–2025, the reported price lies within the transaction day's low–high range for 90.5% of purchases and 98.5% of sales, and within the surrounding five-session range for 94.5% and 99.4%. Purchases outside the range are mostly about 10% below the low, which is consistent with participation in discounted share offerings (not verified trade by trade). Output: `results/price_check_summary.csv`.
- `sanity.py` attaches transaction-level flags (price range, scale errors, late filings, duplicates, very large trades) used in the robustness tables.

### 6. Test signals (`signals/`)

The four signals presented on October 1, 2026. Each folder has its own README.

| Signal | Folder | Definition | Result |
|---|---|---|---|
| Multi-insider purchases | `multi_insider_purchases/` | At least two insiders buy within seven days; train, validation and test periods; equal-weight biotech benchmark | Test 2020–2025, 3 sessions: +0.90 pp; t = 2.30; N = 722 |
| Small-cap purchases | `smallcap_purchases/` | An insider buys in a company in the bottom third by market capitalization; XBI benchmark | 2009–2025, 5 sessions: +1.46 pp; t = 4.02; N = 3,524 |
| Seller silence | `seller_silence/` | An insider who sold in the same month for two years does not sell this year, compared with routine sellers who do; XBI benchmark | 3 months: +0.91 pp (t = 0.80); 6 months: +1.33 pp (t = 0.67); not significant |
| Conviction composite | `conviction_composite/` | Score of 0 to 4 per purchase (small cap, large stake, breadth, seniority); monthly walk-forward long/short against XBI | SIC 2834/2836: alpha +12.8% per year, t = 2.80. Full GICS universe: +7.2% per year, t = 1.55; not confirmed |

Notes:

- The small-cap folder also reports two baselines used on the slides: all purchases (+0.80 pp over 5 sessions) and all sales (−0.07 pp).
- Result files use short codes: S1 small-cap purchases, S2 all purchases, S3 all sales, S4 seller silence, S5 multi-insider purchases.
- The conviction-composite scripts cannot be run from this repository alone: their main input, the SIC 2834/2836 event table, and the code that builds it are not here. `signals/conviction_composite/README.md` lists every input and its source.
- The event case studies in the slides (FDA approvals, financing filings) come from a separate SEC, FDA and ClinicalTrials.gov data store that is not in this repository.

### 7. Run the tests

```bash
python shared/tests/run_tests.py     # 14 checks; 11 run without the data folder
python -m pytest shared              # parser, universe and ingest unit tests, offline
```

## Installation

Set up with conda: `conda env create -f environment.yml`, then `conda activate bridgeway`. Full steps (environment variables, WRDS access, verifying the install, rebuilding the data) are in [`INSTALL.md`](INSTALL.md).

## Data layout

The code expects the data next to this repository (`../data`), or wherever `BRIDGEWAY_DATA` points:

```
data/
  universe/       point-in-time biotech universe (shared/universe/universe.py)
  sec/raw/        SEC quarterly Forms 3/4/5 ZIP archives, 2006Q1-2026Q2
  sec/normalized/ Form 4 purchases and sales used by the signals
  sec/all_forms/  all Forms 3/4/5 tables
  crsp/daily/     CRSP daily prices of the universe securities, 2003-2025
  crsp/market/    XBI/IBB, trading calendar, legacy cache used by the regression test
  compustat/      historical GICS, CRSP-Compustat links, 2025-2026 daily prices
  outputs/        ownership continuity rows, event exports
```

See `docs/data_dictionary.md` for every file and column, and `docs/without_wrds.md` for what can be done without a WRDS account.

## Reproduce the result tables

```bash
python -m signals.smallcap_purchases.report      # small-cap purchases, baselines, seller silence
python -m shared.engine.price_check              # price-range check
```

The report writes aggregate-only tables (`results_table.csv`, `ssn_results.csv`, `conditioning.csv`, `RESULTS.md`). The notebook `signals/smallcap_purchases/insider_signals.ipynb` is saved with its outputs. The multi-insider folder stores its own saved results and report; do not rerun its script to check the setup (see its README).

## Limitations

- All results are before spreads, market impact, borrow constraints and other trading costs.
- The small-cap purchase result is in-sample: the rule was specified on 2009–2025 data. Filings from 2026 have not been examined and are the out-of-sample test.
- The conviction composite is significant on the SIC 2834/2836 set but not on the full GICS universe, and it cannot be rerun from this repository alone.
- The SEC quarterly data sets start in 2006, which limits designs that need a long owner history.
- Form 144 filings have not been collected.
- The SEC update command is run by hand: there is no scheduler, and market data are not updated incrementally.
- Historical GICS is effective-dated rather than publication-vintage, and some CIK intervals rely on a current-identifier fallback.
