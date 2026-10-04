# Historical biotech universe

`universe.py` retains the version 4 membership logic used for the published study. Its manifest now correctly records that current-CIK fallback links are not excluded by the included event mapper. Saved universe files are unchanged. From the repository root, use an explicit external output directory:

```bash
python -m shared.universe.universe --help
python -m shared.universe.universe \
  --output-dir "${BRIDGEWAY_DATA:-../data}/universe"
```

Licensed outputs must stay outside this repository. The explicit command above uses `BRIDGEWAY_DATA`, defaulting to `../data`; the legacy module's bare default is retained for compatibility, so supply `--output-dir`.

## Version 5 for future rebuilds

`universe_v5.py` uses rule `crsp-us-common-10-11-v5`. It rejects fractional identifiers, malformed dates and conflicting classification overlaps, treats zero SIC as missing, and checks source lineage, complete universe views and crosswalk values. It retains the version 4 policy of using PERMNO-only classification history only when the exact PERMNO/GVKEY history key is absent. GICS remains effective-dated and can contain retrospective revisions.

```bash
python -m shared.universe.universe_v5 --dry-run
python -m shared.universe.universe_v5 --output-dir "${BRIDGEWAY_DATA:-../data}/universe_v5"
```

The default version 5 destination is `universe_v5/` under the configured data root. Publication writes an immutable directory under `.universe_v5.versions/` and atomically switches the `universe_v5` symlink. `--overwrite` updates only a pointer produced by this version; older directories and saved versions are retained. Existing physical directories cannot be replaced. A fresh installation can use a pointer named `universe` to keep `Config.universe_path` unchanged; the published study's current data location is not switched automatically.

Version 5 has synthetic validation and publication tests. It has not rebuilt the licensed universe or replaced any published result. A new WRDS extraction and comparison are required before adopting it.

## WRDS connection

The existing `wrds_cloud.py` helper is unchanged. It reads exact WRDS entries from `.pgpass`, chooses the last matching entry, selects Duo option 1, accepts a first-use host key and limits its tunnel to one hour.

The opt-in `wrds_cloud_v2.py` uses the first matching pgpass entry, supports wildcard fields and `PGPASSFILE`, requires restrictive pgpass permissions, verifies SSH host keys and keeps the tunnel open until context exit. Before first use, obtain the expected server fingerprint from WRDS and verify it when enrolling the host key in SSH's `known_hosts`. Do not accept an unverified key just to bypass an error. Confirm the Duo option with the account configuration; `--duo-option` defaults to 1.

```bash
python -m shared.universe.wrds_cloud_v2 --help
python -m shared.universe.wrds_cloud_v2 --username YOUR_WRDS_USERNAME
# After verifying the opt-in connector, use it with a future rebuild:
python -m shared.universe.universe_v5 --connector v2 --dry-run
```

The new connector has mocked authentication tests; no live login was performed. Run a real connection check before adopting it. The universe builder defaults to the existing connector.

Offline tests: `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider shared/universe/tests`.
