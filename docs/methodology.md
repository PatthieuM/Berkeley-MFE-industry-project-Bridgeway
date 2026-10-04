# Methodology

## Sample start, timing, and horizons

The common S1–S4 reporting period starts in 2009. The SEC quarterly Insider Transactions Data Sets begin in 2006Q1, providing 2006–2008 purchase history for S1's first annual size cutoff.

The event date is the public SEC filing date. Entry is the close of the first exchange session strictly after filing. An *h*-session return runs from that entry close to the close *h* exchange sessions later. S1–S4 are reported at 1, 2, 3, 5, 10, 21, 42, 63, and 126 sessions. Their pre-registered headlines are 5 sessions for S1–S3 and 63/126 sessions for S4 seller silence; the other rows are descriptive. S5 uses its separate seven-horizon grid.

For S1–S4, missing prices, invalid total-return levels, and paths containing a delisting flag produce missing outcomes. The engine keeps the scheduled entry session and leaves incomplete outcomes missing. S5 follows its frozen return construction, which retains available delisting returns.

## Universe, events, and benchmark

The historical universe contains U.S. common shares meeting the strict biotech definition at the relevant date, including firms that subsequently delisted. Membership uses historical GICS 35201010 with labelled SIC fallback and effective-dated CRSP/Compustat identifiers. Historical GICS is effective-dated but not publication-vintage; current-CIK fallback intervals remain a limitation.

S1–S3 use original Form 4 filings, exclude amendments, and create one observation per issuer × filing date × direction. Only non-derivative transaction codes P and S enter these signals. S1 restricts the S2 purchase baseline to the lowest third by lagged market capitalization, using cutoffs estimated from eligible purchases filed in earlier years. S4 is built at owner-month level and freezes availability on the second U.S. federal business day after month-end.

Code F covers payment of an exercise price or tax liability by delivering or withholding securities and is excluded. Code S covers open-market or private sales. The available fields do not reliably identify whether a sale was tax-motivated. See the [SEC Form 4 transaction-code instructions](https://www.sec.gov/files/form4.pdf).

S1–S4 use XBI total return over the stock's exact entry and exit sessions. S5 uses an issuer-excluded equal-weight biotech benchmark and a separate [research design](../signals/multi_insider_purchases/README.md).

## Inference and conditioning

S1–S3 report means, medians, hit rates, t-based confidence intervals, and two-sided p-values with issuer × filing-month two-way clustered standard errors. S4 compares firm-months containing a silent routine seller (SSN) with firm-months containing continued routine selling and no silent routine seller (SSS-only). It clusters by issuer × signal month.

The conditioning table is descriptive. Buyer roles are exclusive: CEO/CFO/COO/chair/president first, then other officers, directors only, and 10% owners only. Purchase-dollar terciles use reported transaction value, with shares × price as fallback. Each year's thresholds use earlier years only.

S3 seller classes apply the Cohen–Malloy–Pomorski rule at each year start. A seller must have transactions in each of the previous three calendar years, known by that year start. A shared transaction month across all three years gives the routine label; no shared month gives the opportunistic label. Insufficient history remains unclassified. An issuer-day is opportunistic if any seller is opportunistic, otherwise routine if any seller is routine, otherwise unclassified.

## Research separation

The [research log](research_log.md) separates exploratory work, chronological validation and the unified-branch replication. It records rejected hypotheses and incomplete tests. S5's procedure and existing C01 note are in its [README](../signals/multi_insider_purchases/README.md).

## Sanity checks

Executable checks cover next-session timing, aligned benchmark endpoints, two-way clustered inference, seller-silence availability, prior-years-only breakpoints, the frozen v6 regression, and full-universe price coverage.

Transaction sanity checks retain rows and attach flags for direction inconsistencies, negative or long filing lags, duplicated economic transactions, reported prices outside the CRSP daily range, likely 10× price-scale errors, extreme returns, missing paths, delistings, and unusually large trades. A large trade is flagged when reported value exceeds either 1% of lagged market capitalization or trailing 20-session average dollar volume.

Ownership continuity is separately audited across derivative and non-derivative transactions and holdings, using owner × issuer × security × ownership-form groups and CRSP share-factor adjustments. In [continuity_summary.csv](../results/continuity_summary.csv), 289,380 of 961,584 rows (30.09%) are consistent, 18,358 (1.91%) are first observations, 129,697 (13.49%) are inconsistent, and 524,149 (54.51%) are not comparable. Non-comparable rows lack enough information to compare the previous position, intervening transactions and reported position. Continuity status is a diagnostic and does not filter the signals.

These published counts use the retained legacy continuity implementation. Version 2 identifier, split and date handling is available separately; its results require a new audit and comparison before replacing the published summary.
