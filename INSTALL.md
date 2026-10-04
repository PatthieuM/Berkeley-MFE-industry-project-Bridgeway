# Installation with conda

These steps set up everything needed to run this package: the tests, the signal reports, the notebook, and (optionally) the WRDS and SEC data pulls.

## 1. Prerequisites

- **Conda.** [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is enough; Anaconda also works. Check with `conda --version`.
- **Git.**
- **Optional:** a WRDS account with CRSP and Compustat access, needed only to rebuild the universe or re-pull prices (step 6).

Conda older than 23.10 uses a slow solver by default that can hang for many minutes on this environment. Check with `conda config --show solver`; if it says `classic`, switch to the fast solver once:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

## 2. Get the code

The code expects its data folder **next to** the repository, so clone it into a project folder:

```bash
mkdir bridgeway && cd bridgeway
git clone https://github.com/PatthieuM/Berkeley-MFE-industry-project-Bridgeway.git code
cd code
```

This gives:

```
bridgeway/
  code/    this repository
  data/    licensed and SEC data (you create or receive this; see step 5)
```

To keep the data elsewhere, set `BRIDGEWAY_DATA` (step 4).

## 3. Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate bridgeway
```

This installs Python 3.12.2 and dependencies from conda-forge. Python, numpy, pandas, scipy and statsmodels match the versions recorded in the S5 results. PyArrow 25.0.0 is a verified reader for the pinned parquet inputs; the historical run did not record its PyArrow version. Matching these versions supports reproduction but does not by itself prove numerical equivalence.

Register the environment as a Jupyter kernel so the notebook can use it:

```bash
python -m ipykernel install --user --name bridgeway --display-name "Python (bridgeway)"
```

Later, after pulling changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

For pip, use Python 3.12.2 and `pip install -r requirements.txt`; the file applies `constraints-research.txt` to the numerical packages and parquet reader. Conda also installs the notebook kernel and test runner and remains the supported setup.

## 4. Environment variables

Set these once inside the environment; conda restores them on every `conda activate bridgeway`:

```bash
conda env config vars set SEC_USER_AGENT="Your Name your.email@berkeley.edu"
conda env config vars set BRIDGEWAY_DATA="$HOME/bridgeway/data"
conda env config vars set WRDS_USERNAME="your_wrds_username"
conda activate bridgeway
```

| Variable | Needed for | Notes |
|---|---|---|
| `SEC_USER_AGENT` | Any SEC/EDGAR download | SEC requires a name and contact email; requests are refused without it |
| `BRIDGEWAY_DATA` | Everything that reads data | Optional; defaults to `../data` next to the repository |
| `WRDS_USERNAME` | WRDS pulls only | Optional; otherwise inferred from `~/.pgpass` |

Check with `conda env config vars list`.

## 5. Verify the install

```bash
python shared/sec/acquire_normalize.py --self-test
python -m pytest shared/sec/form4_parser/tests
python shared/tests/run_tests.py
```

- The first two are fully offline and should pass on a fresh install.
- `run_tests.py` runs offline unit checks plus regression checks on local data. Without the data folder it reports **11/14 passed**: the three data-dependent checks (`no_lookahead`, `v6_regression`, `universe_full_coverage`) fail with `FileNotFoundError`. That is expected until the data is in place; with the data, all 14 should pass.

## 6. Data

Licensed CRSP/Compustat rows and row-level SEC data are **not** in this repository. Either get the prepared `data/` folder from a teammate with WRDS access, or rebuild it. The expected layout is in the README ("Data layout").

### WRDS access (only to rebuild)

The WRDS scripts connect through an SSH tunnel and read your password from `~/.pgpass`. Open `~/.pgpass` in a text editor (typing the password into a shell command would save it in your shell history) and add one line:

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD
```

Then restrict the file to your user:

```bash
chmod 600 ~/.pgpass
```

The tunnel uses `ssh` through `pexpect`, so the WRDS scripts run on macOS and Linux. On Windows, run them from WSL.

Test the connection (approve the Duo push when prompted):

```bash
python shared/universe/wrds_cloud.py
```

### Rebuild order

```bash
# 1. Point-in-time biotech universe (WRDS)
python -m shared.universe.universe --output-dir "${BRIDGEWAY_DATA:-../data}/universe"

# 2. CRSP daily prices, XBI and trading calendar for the universe (WRDS)
python shared/market/pull_universe_prices.py

# 3. SEC quarterly Forms 3/4/5 datasets, one quarter at a time (public, needs SEC_USER_AGENT)
python shared/sec/acquire_normalize.py --year 2024 --quarter 1

# 4. Lossless normalization of all downloaded quarters (all seven SEC tables)
python shared/sec/acquire_normalize.py --normalize-all

# 5. Ownership-continuity audit
python -m shared.engine.continuity
```

Repeat step 3 for every quarter from 2006Q1 onward. To rebuild without WRDS, see `docs/without_wrds.md`.

## 7. Run the research

```bash
python -m signals.smallcap_purchases.report       # S1-S4 tables and RESULTS.md
jupyter lab signals/smallcap_purchases/insider_signals.ipynb   # or open it in VS Code
```

Pick the **Python (bridgeway)** kernel in the notebook. Jupyter Lab is not in the environment; install it with `conda install -c conda-forge jupyterlab` if you want it, or use VS Code's notebook support.

S5 (`signals/multi_insider_purchases/run_split.py`) reads a frozen archive that is kept locally and not committed; see its README.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `conda env create` hangs on "Solving environment" | Switch to the libmamba solver (step 1) |
| `CondaValueError: prefix already exists` | `conda env remove -n bridgeway`, then create again |
| SEC requests return 403 or the script stops asking for `SEC_USER_AGENT` | Set `SEC_USER_AGENT` (step 4) and reactivate the environment |
| `FileNotFoundError` under `data/` | Data folder missing or `BRIDGEWAY_DATA` points to the wrong place |
| WRDS connection times out | Approve the Duo push; check `~/.pgpass` has mode 600 |
| Notebook uses the wrong Python | Select the "Python (bridgeway)" kernel |

To remove everything: `conda env remove -n bridgeway`.
