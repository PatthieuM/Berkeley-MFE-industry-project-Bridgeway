# Multi-insider purchases in biotech

Executed 2026-09-24. The selected configuration uses a seven-calendar-day window, at least two buyers, and a three-session mean excess return. The saved test estimate is **+0.90 pp**, with a two-sided p-value of **.031**. See [REPORT.md](REPORT.md) for a readable presentation of the saved results.

## Scope

The signal identifies filing days when several eligible insiders have disclosed purchases of the same issuer within a short window. It measures subsequent returns relative to an issuer-excluded, daily equal-weight biotech benchmark.

| Design choice | Project requirement (notes file not included) | Implementation |
|---|---|---|
| Biotech universe | Project notes, line 115: “We'll do this for Biotech stocks only” | Effective-dated GICS 35201010, with SIC 2836 fallback, in the existing v4 universe |
| Benchmark | Project notes, lines 117–121: equal-weight biotech is the preferred benchmark | Exclude all securities of the event issuer; require 99% daily coverage; XBI is descriptive |
| Entry | Project notes, line 123: “use the close of the 3/24, the day after filing date” | First session strictly after the filing date; returns start the next session |
| Horizons | Project notes, line 123: 1, 2, 3, 5, 10 trading days, one month, one quarter | 1, 2, 3, 5, 10, 21, 63 sessions; 21 and 63 approximate a month and quarter |
| Purchases and multiple insiders | Project notes, lines 73–77 and 130 | Eligible code-P purchases and distinct natural-person buyers |
| Baseline and time stability | Project notes, lines 93–95 and 136 | A small signal family, calendar stages, and descriptive period halves |

The signal uses a purchase-count window without the size, role and stake components of the conviction composite. The primary question is whether these purchase events beat biotech. The descriptive clustered-minus-comparison-arm result addresses whether buyer clustering adds information over other purchase days.

## Literature and adaptations

| Source | Relevance and limits |
|---|---|
| [Alldredge & Blank (2019)](https://onlinelibrary.wiley.com/doi/pdf/10.1111/jfir.12172) | Motivates multiple-insider purchases. The abstract reports abnormal returns above 2% in the following month. Seven- and fourteen-day filing windows are this study's choices; the exact paper window has not been verified. |
| Lakonishok & Lee (2001); Jeng, Metrick & Zeckhauser (2003) | Historical evidence is stronger for purchases than sales. A size filter is omitted to limit this study's search, rather than inferred to be unnecessary from those papers. |
| [Cohen, Malloy & Pomorski (2012)](https://dash.harvard.edu/server/api/core/bitstreams/7312037e-2b77-6bd4-e053-0100007fdf3b/content) | Routine classification requires prior trading history. The −82 bp monthly number in Table II is the opportunistic-minus-routine sales coefficient difference, not a standalone sales alpha. |
| Ravina & Sapienza (2010) | Independent directors' purchases also predict abnormal returns; this does not establish that one role always dominates. No role filter is used here. |
| [Biggerstaff, Cicero & Wintoki (2020)](https://mbwintoki.github.io/files/21_JCF_2020_Wintoki.pdf) | Motivates trade-pattern analysis and dependence across firms and time. Its return tests cluster by firm and month; this study's issuer/quarter inference is an adaptation. |
| Harvey, Liu & Zhu (2016); Bailey et al. (2014); McLean & Pontiff (2016) | Motivate controlling specification search and examining performance across periods. A t-statistic above 3 is a factor-discovery warning from that literature, not a universal cutoff for this event study. |
| [Hu (2024)](https://theses.lib.polyu.edu.hk/handle/200/13251) | Finds reduced insider profits in pharmaceutical firms affected by FDAAA disclosure, especially for non-routine trades. This does not establish the size of this filing-based signal's return. |

For eligible purchase rows from 2009 onward, 91.9% lack the history needed for the frozen routine label. That coverage is too sparse for a routine/non-routine grid dimension here.

## Frozen inputs

The four inputs live in `frozen/` inside this folder: `events.parquet`, `purchases.parquet`, `classification.py` and `protocol.json`. The folder is git-ignored because the two parquet files hold licensed CRSP-derived data. `run_split.py` checks each file's SHA-256 against the values pinned in the script and recorded in `results.json`, and stops on any mismatch. The files are never regenerated; if they are missing, the author restores them from a private backup.

They were taken unchanged from the archive `bridgewayLocal_frozen_2026-09-24.tar.gz` (SHA-256 `7e7d284dff704aaf2b66eaa975df135c54d246b72940c77edef0f51da6ee51e4`), the workspace of the earlier study, which the author keeps outside the repository.

- Events contain outcome records at seven horizons, including unavailable or censored returns, for eligible purchase company-days from 2007-04-16 through 2025-12-31. Window/buyer relabeling reuses those records; it does not price new returns. Delisting returns are included where CRSP provides them, and a position is carried as cash after a verified final payout.
- Eligibility follows `protocol.json:14`: original common-stock code P; sole identifiable natural-person officer/director; direct or indirect holdings; affirmative financing/exercise rows excluded; unknown venue retained. Code P can include private purchases; retained unknown venue does not establish open-market execution.
- A company-day is excluded if its seven- or fourteen-day window includes multiple security classes.
- Effective-dated GICS history can include back-dated reclassifications. Closing-price microstructure and daily rebalancing can affect both event and benchmark returns; the sign of bias in their difference has not been established.

## Signal family and stages

On filing date *d*, at least *k* distinct eligible buyers must have publicly filed purchases in the inclusive calendar window **[d − (w − 1), d]**. A later filing never changes an earlier solo day's label. Events are filing company-days, not only the first crossing of the buyer threshold.

| Parameter | Values | Interpretation |
|---|---|---|
| Window, calendar days | 7, 14 | One- and two-week windows |
| Minimum buyers | 2, 3 | At least two or three distinct eligible buyers |
| Horizon, sessions | 1, 2, 3, 5, 10, 21, 63 | The requested short horizons plus month/quarter approximations |
| Estimator | Mean; mean clipped at ±100 pp (`wins100`) | Unclipped and capped excess-return means |

The 2 × 2 × 7 × 2 grid contains 56 configurations. Universe, eligibility, filing-date entry, benchmark and spacing rules are fixed. Within each issuer and arm, retain the first event, then the next only after *h* sessions. Spacing is selected before checking return availability; the outcome must end within the stage. For k=3, the arm stored as `solo` includes days with one or two buyers, so it means “below the buyer threshold.”

Train-period notes: the 10-session horizon (C01) and the ±100 pp winsorization were chosen with prior exposure to 2009–2025 results. The −100 pp cap does not bind in the training clustered sample (minimum −91.7 pp).

| Stage | Signal dates | Purchase days | Clustered k≥2 / k≥3, w=7, before spacing | Issuers | Quarters |
|---|---|---:|---:|---:|---:|
| Train | 2009-01-01 to 2016-12-31 | 2,778 | 890 / 367 | 325 | 32 |
| Validation | 2017-01-01 to 2019-12-31 | 1,155 | 385 / 172 | 260 | 12 |
| Test | 2020-01-01 to 2025-12-31 | 3,122 | 968 / 379 | 454 | 24 |

Training outcomes end by 2016-12-31 and validation outcomes by 2019-12-31.

## Selection and inference

1. Fit all 56 training configurations. The mean is an OLS intercept. The reported SE is the maximum of two-way issuer/quarter CR1, issuer CR1, quarter CR1 and HC1; degrees of freedom are min(issuers, quarters) − 1. Report Holm p-values over all 56 rows. The three distinct training results with the largest t-statistics proceed to validation, with the original stable tie order.
2. Refit those candidates on validation. **Select the largest validation t with t > 0 and one-sided p < .10.** If none qualifies, stop without a test-period statistic.
3. Fit the selected configuration on the test stage. **Pass requires a positive estimate and two-sided p < .05.** Descriptive checks do not alter this decision.
4. Report issuer-pairs bootstrap intervals (1,999 draws, seed 20260916), issuer omissions, largest issuer event share and the two period halves, 2020–2022 and 2023–2025. Also report the issuer × circular two-quarter product-weight interval as a separate dependence sensitivity.
5. Report clustered-minus-below-threshold purchases, XBI, seven horizons and C01 (7, 2, 10, mean) descriptively. The C01 row is produced only if validation selects a configuration.

The max-of-four SE is an ad hoc safeguard. On training, the two-way estimate is below HC1 in 38 of 56 rows, by up to 52%. This comparison does not prove that smaller estimates are errors or that the resulting intervals have nominal finite-sample coverage. The validation screen uses only 12 time clusters. Screening three correlated candidates does not have a universal false-pass probability or power independent of their dependence and effect sizes.

Long outcome windows can cross calendar-quarter boundaries, and quarter clustering does not model every source of dependence across quarters. [Cameron–Gelbach–Miller's PSD correction](https://cameron.econ.ucdavis.edu/research/JBESpaper2009version.pdf) sets negative eigenvalues to zero; it does not enlarge an already positive scalar variance.

Returns are event-level excess returns before costs. A constant additional per-event cost equal to the mean would erase the estimated excess over a costless benchmark. Portfolio turnover, spreads and market impact are not modeled. Reported purchase prices are a price-level proxy, not observed entry quotes.

## C01 and implementation checks

C01 remains a descriptive reference. Its earlier all-horizon work and this grid have different selection and multiplicity procedures.

Honesty note for the reviewer: the author has previously seen full-sample results at all horizons (C01). This analysis demonstrates the procedure and the signal's temporal stability; a future run requires separately frozen inputs, hashes, stage dates and reporting periods.

The run notes record the executed script's SHA-256, `727b856b87539a1731d6fd4f0457dcc5cd66c95b310bfc52ec9c78da84a62a98`, at execution. The file was edited afterward and before its first commit; the exact executed source was not retained, so those intervening edits cannot now be independently checked. The calculation code matches the earliest committed version. Later changes cover labels, documentation, formatting and input loading. The historical notes record a teammate's independent validation-stage reproduction byte-for-byte and a leakage check that blanked every outcome dated 2017 or later and reproduced the training ledger to 8e-15.

On 2026-10-03 the input loader was changed to read the four files from `frozen/` instead of extracting them from the archive; the four pinned input hashes are unchanged. A run of the current script writes `results.json` and `RESULTS.md` without the archive hash, so their bytes differ from the saved files even when every number is identical.

`results.json`, `RESULTS.md` and `train_ledger.csv` are the files written by the run. `train_ledger.csv` is byte-identical to the run's output; `results.json` and `RESULTS.md` differ from it only in the run title and the decision wording, relabelled on 2026-09-29 to match the script's labels.

## Files and commands

| File | Purpose |
|---|---|
| `run_split.py` | The analysis script; reads `frozen/` and overwrites the three output files if executed, so do not run it to check the setup |
| `frozen/` | The four inputs, git-ignored; see Frozen inputs |
| `results.json`, `train_ledger.csv`, `RESULTS.md` | Saved results and ledger from the run |
| `render_report.py`, `REPORT.md` | Readable rendering of the saved JSON; runs no analysis |
| `test_revision.py` | Saved-file hashes, saved-result rendering, overwrite protection, and the input check with valid, missing and corrupted inputs |

Run these commands from this folder:

```sh
PYTHONDONTWRITEBYTECODE=1 python render_report.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest test_revision
```

The run recorded Python 3.12.2, pandas 2.2.2, numpy 1.26.4, statsmodels 0.14.2 and scipy 1.13.1; it did not record the parquet engine; PyArrow 25.0.0 reads the inputs.

A future test needs new inputs frozen with their hashes, the same signal family, thresholds, bootstrap seed and draw count, the validation stage first, then the test stage on the new sample only. The code that built the inputs is inside the archived workspace the author keeps.
