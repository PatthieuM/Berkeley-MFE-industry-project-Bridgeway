# SEC filing crawl and updates

Fetches Forms 3, 4, 5 and 144 from EDGAR for a period, parses them with `shared/sec/form4_parser`, and stores them in a SQLite database.

Set `SEC_USER_AGENT` to your organization and contact email. Run the commands from the repository root:

```bash
export SEC_USER_AGENT="Organization Name contact@example.com"
python -m shared.sec.ingest.updates --universe --period week                              # biotech universe, last 7 days
python -m shared.sec.ingest.updates --tickers RNST --start 2003-01-01 --end 2005-12-31    # named stocks, any dates
python -m shared.sec.ingest.updates --all --period week                                   # every filer on EDGAR
```

- **Targets:** choose `--ciks`, `--tickers`, `--universe` or `--all`. `--tickers` resolves current ticker symbols. `--all` searches the EDGAR quarterly indexes; the current-quarter index includes filings through the previous business day.
- **Dates:** use `--period day|week|month` or `--start YYYY-MM-DD`. `--end YYYY-MM-DD` defaults to today. Dates are inclusive filing dates.
- **Reruns:** completed filings are skipped. Fetch and storage failures are retried by later runs covering the filing date.
- **Parser flags:** rows with status `error`, `empty`, `needs_review` or `unsupported_pdf` are retained. These filings are marked done and are not retried automatically. `flagged` counts successfully stored filings with at least one flagged row; it is included in `fetched`.
- **Exit codes:** completed crawls exit 1 for fetch, storage or discovery failures, otherwise 0. Parser flags alone do not change the exit code. Command-line usage and date-validation errors exit 2. HTTP 401/403 aborts with a traceback.

The database is `sec/insider_filings.sqlite` under `BRIDGEWAY_DATA`, or under `../data` beside the repository when that variable is unset.

| Table | Content |
|---|---|
| `filings` | One row per attempted filing: accession, CIK, form, filing date, status, failed-attempt count (`attempts`), error |
| `ownership_rows` | Parser rows, including diagnostic placeholders. See `row_kind` and `parse_status` |

Notes for queries:

- Use `ownership_rows.issuer_cik` to select a stock. `filings.cik` is the CIK the filing was found under, which can be the reporting owner.
- A joint filing repeats each line once per owner. Count each `economic_row_id` once when summing shares or values.
- A numeric column keeps the filing's raw text when the value is not a plain number, as in some pre-2004 text filings. Add `typeof(column) != 'text'` to aggregates.
- Inspect `parse_status` and `parse_warnings` before analysis; successful storage does not guarantee successful parsing.
- A successful save replaces a filing's rows in one transaction. A failed replacement keeps the old rows.
- Opening an older database adds missing parser columns. Existing rows have `NULL` in those columns.

Tests need no network: `python -m unittest shared.sec.ingest.tests.test_ingest`.
