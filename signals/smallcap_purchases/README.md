# S1 — small-cap insider purchases

S1 is the subset of mapped code-P purchase issuer-days whose lagged market cap is in the lower tercile. The cutoff for year Y is estimated only from eligible purchase events filed in earlier years, beginning in 2006. Market features end before entry. Entry is the first session close strictly after the filing date; outcomes are matched to XBI over identical exchange sessions.

The official result is +1.459 percentage points over five sessions (N=3,524; issuer × filing-month clustered t=4.02, p=0.0000816) for 2009–2025. The same engine also reports S2 (all purchases) and S3 (sales) as baselines.

From `submission/`:

```bash
python -m signals.smallcap_purchases.report
```

The checked-in notebook contains cells only and no outputs. It calls the shared modules and is intended for a fresh execution by the reviewer.
