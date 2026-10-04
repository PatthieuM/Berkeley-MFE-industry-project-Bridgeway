# Market data acquisition

`pull_universe_prices.py` requests CRSP `dsf_v2` rows for every historical universe PERMNO, separately pulls XBI by ticker, and writes an XBI-derived exchange-session calendar. Output paths are rooted at `BRIDGEWAY_DATA`; no licensed data are written into `submission/`.

From `submission/`:

```bash
python shared/market/pull_universe_prices.py
```

This requires an authenticated WRDS Cloud connection and Duo approval. Expected outputs are annual `daily_<year>.parquet` files, `xbi_daily.parquet`, `trading_calendar.parquet`, and an aggregate pull report under the external data root.
