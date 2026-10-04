# Signal summary

S1–S4 use the historical biotech universe, next-session-close entry, and matched XBI total returns. S1–S3 inference is clustered by issuer and filing month; S4 contrasts seller-silence and continued-selling firm-months with issuer × signal-month clustering. All four are reported at 1, 2, 3, 5, 10, 21, 42, 63, and 126 sessions.

| Signal | Test | Result | Interpretation | Source |
|---|---|---|---|---|
| Multi-insider purchases (S5) | Its own chronological test: 2020–2025, 3 sessions, issuer-excluded equal-weight biotech benchmark | +0.90 pp; SE 0.39; t=2.30; p=.031; 95% CI [+0.09,+1.71]; N=722 | Meets its stated p-value and sign rule; clustered-minus-solo secondary is +0.49 pp, p=.29 | [S5 saved results](../signals/multi_insider_purchases/RESULTS.md) |
| Small-cap purchases (S1) | 2009–2025, 5 sessions | N=3,524; +1.459 pp; t=4.02; p=0.0000816 | Positive headline result | [S1–S3 report](../signals/smallcap_purchases/results/RESULTS.md) |
| Seller silence (S4) | 2009–2025, 63 and 126 sessions; SSN minus SSS-only | 63: N=2,805/1,460; +0.91 pp; t=0.80; p=.427. 126: N=2,648/1,394; +1.33 pp; t=0.67; p=.505 | Positive estimates; neither statistically significant | [S4 results](../signals/smallcap_purchases/results/ssn_results.csv) |
| Conviction composite | Monthly walk-forward 2007–2025, long/short of high-conviction purchases against sales, XBI benchmark | SIC 2834/2836 (759 firms): alpha +12.8% per year; t=2.80. Full GICS universe (875 firms): +7.2% per year; t=1.55 | Significant on the SIC set; not confirmed on the full GICS universe | Presentation slides; inputs and caveats in [conviction README](../signals/conviction_composite/README.md) |

Baselines reported with the small-cap signal, 2009–2025, 5 sessions: all purchases (S2) N=11,280; +0.798 pp; t=4.56. All sales (S3) N=30,080; −0.066 pp; t=−0.93.

For S4, the two N values count firm-months with silent routine sellers (SSN) and continued routine selling without silent routine sellers (SSS-only), respectively.

S5 is reported separately: its eligibility, benchmark, stage split, standard-error rule, and horizon selection are specific to its own design. The stated numbers are quoted from its source report.

The SEC data layer covers Forms 3, 4, and 5 and preserves submissions, owners, non-derivative transactions and holdings, derivative transactions and holdings, and footnotes. Ownership continuity audits 961,584 rows; for non-derivative shares, 68% of comparable rows reconcile exactly (see the [README](../README.md) and [continuity summary](continuity_summary.csv)). The Form 144 parser is included and sample-tested, but Form 144 filings are not collected and no signal uses them.

## Limitations and possible extensions

Pre-2006 owner history is unavailable in the quarterly SEC datasets. Useful extensions include Form 144 collection, earnings and clinical-trial conditioning, scheduled data refreshes, integration of incremental SQLite ingestion with the research pipeline, and explicit transaction-cost modelling. Effective-dated GICS is not publication-vintage, and S1–S4 return windows with unavailable data after delisting remain missing. S5 retains available delisting returns under its frozen construction.
