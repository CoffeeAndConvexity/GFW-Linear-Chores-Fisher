# Chores-CEEI

This repository contains implementations and numerical experiments for computing a competitive equilibrium with equal incomes (CEEI) when every item is a chore. It compares three approaches:

- **GFW**: a greedy Frank-Wolfe method using a linear-minimization oracle.
- **EPM**: an exterior-point method using a quadratic-minimization oracle.
- **COMB**: a combinatorial price/allocation method whose balance-allocation step is solved with CVXPY.

The experiments report the number of oracle calls/iterations, running time, and the fraction of instances solved. Some runs also compare utilitarian, egalitarian, and Nash welfare. The approximate-equilibrium threshold is `1e-2` by default (`a1` in the CSV columns), while the exact threshold (`e`) is `1e-8`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `code/gfw.py` | GFW algorithm and its Gurobi LP oracle |
| `code/epm.py` | EPM algorithm and its Gurobi/CVXPY QP oracle |
| `code/combinatorial.py` | Combinatorial algorithm |
| `code/run_simulation.py` | Configurable synthetic-distribution and welfare experiments |
| `code/run_simulation_epsilon.py` | Fixed-size approximation-threshold experiment |
| `code/run_bidding.py` | Experiments using the bidding data |
| `code/plot_*.py` | Individual plotting commands |
| `code/reproduce_figures.py` | Regenerates all 24 files currently in `figures/` |
| `data/` | Raw bidding input and committed experiment summaries |
| `figures/` | Paper-ready plots generated from the CSV summaries |
| `diagram/` | Explanatory geometry/GFW diagrams and their source scripts |

The repository includes the data needed by the scripts and the resulting figures. Regenerating plots from the committed CSVs does not require Gurobi. Re-running the numerical optimization does require a working Gurobi installation and license, so a clone is not fully executable offline without that external solver license.

## Setup

Python 3.10 or newer is recommended. The repository was verified with Python 3.12.2, NumPy 1.26.4, SciPy 1.13.1, pandas 2.2.2, Matplotlib 3.8.4, CVXPY 1.6.0, and Gurobi 11.0.3.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure Gurobi using a normal `gurobi.lic` file or another mechanism supported by Gurobi. For an ephemeral WLS environment, the code also recognizes these environment variables:

```bash
export GUROBI_WLSACCESSID='...'
export GUROBI_WLSSECRET='...'
export GUROBI_LICENSEID='...'
```

Do not put license credentials in source files or commit them to Git.

## Reproduce the committed figures

From the repository root, regenerate all 24 experiment figures from the committed CSVs with:

```bash
python code/reproduce_figures.py --output-dir figures
```

Use a temporary output directory if you only want to check reproducibility without replacing tracked artifacts:

```bash
python code/reproduce_figures.py --output-dir /tmp/chores-ceei-figures
```

The generated filenames match the current contents of `figures/`. Numerical values come directly from the committed CSVs, so plotting is deterministic apart from PDF metadata and minor rendering differences between Matplotlib/font versions.

Individual plot commands are also available:

```bash
python code/plot_simulation.py \
  data/uniform_5x5.10x10....30x30_100.csv \
  --num-seeds 100 --output-dir figures

python code/plot_welfare.py \
  --csv data/uniform_5x5.10x10....80x80_welfare_10.csv \
  --output-dir figures

python code/plot_simulation_epsilon.py \
  --distribution uniform --size 300x300 --num-seeds 100 \
  --epsilons 0.5 0.05 0.005 0.0005 \
  --variant approx --xmode epsilon --xscale log --yscale log \
  --output-dir figures
```

The two rectangular multi-distribution plots use `plot_simulation.py --group`; see `python code/plot_simulation.py --help` or `code/reproduce_figures.py` for the exact input groups.

## Run numerical experiments

Start with a small smoke test because the complete experiments can be expensive:

```bash
python code/run_simulation.py \
  --sizes 2x2 5x5 \
  --distributions uniform \
  --num-seeds 2 \
  --algorithms GFW EPM COMB \
  --save-dir results
```

Every synthetic instance uses an all-ones budget vector. Seeds are the integers from zero through `num_seeds - 1`. Available distributions are:

- `uniform`: continuous uniform draws on `[0, 1)`.
- `randint`: integer draws from 1 through 1000.
- `randint10`: integer draws from 1 through 10.
- `lognormal`: NumPy's standard log-normal draw.
- `truncnormal`: SciPy truncated-normal draws with bounds `1e-3` and `10` in standardized units.
- `exponential`: NumPy's standard exponential draw.
- `uniformha`: the uniform experiment with the historical high-accuracy EPM settings (`FeasibilityTol=1e-9`, `OptimalityTol=1e-9`, and `BarCorrectors=1000`).

For example, the 100-seed, three-algorithm comparison represented by the `5x5` through `30x30` CSVs can be rerun with:

```bash
python code/run_simulation.py \
  --sizes 5x5 10x10 15x15 20x20 25x25 30x30 \
  --distributions uniform exponential lognormal truncnormal randint \
  --num-seeds 100 \
  --algorithms GFW EPM COMB \
  --save-dir results
```

The threshold experiment at `300x300` is:

```bash
python code/run_simulation_epsilon.py \
  --size 300x300 \
  --epsilons 0.5 0.05 0.005 0.0005 \
  --distributions uniform lognormal truncnormal \
  --num-seeds 100 \
  --algorithms GFW EPM \
  --save-dir results
```

The bidding-data runs are:

```bash
python code/run_bidding.py \
  --data data/bidding-data.csv \
  --sizes 5 50 100 150 200 250 300 \
  --num-seeds 100 \
  --variant both \
  --save-dir results
```

`data/bidding-data.csv` contains 667 anonymized bidders and submissions numbered through 526. The categorical bids are converted to disutilities as `yes=1`, `maybe=3`, `no response=5`, `no=7`, and `conflict=7M+1`; the noise variant adds seeded Gaussian noise with standard deviation `0.2` and clips values below `1e-3`.

Run `python code/run_simulation.py --help`, `python code/run_simulation_epsilon.py --help`, or `python code/run_bidding.py --help` for all command-line options. Write new runs to a separate directory if you want to preserve the committed timing data.

## Conducted experiments in `data/`

The filename convention is generally:

```text
<distribution>_<first-size>.<second-size>....<last-size>_<number-of-seeds>.csv
```

The committed data covers:

| Experiment | Sizes / thresholds | Seeds | Algorithms represented |
| --- | --- | ---: | --- |
| Core synthetic comparison | `5x5, 10x10, ..., 30x30`; uniform, exponential, log-normal, truncated-normal, and integer data | 100 | GFW, EPM, COMB |
| Large square scaling | `5x5, 50x50, ..., 300x300` for the five main distributions | 100 | GFW, EPM |
| High-accuracy EPM | Same large-square sizes, `uniformha` | 100 | GFW, EPM |
| Rectangular scaling | `10x100, 500x100, ..., 3000x100` and the transposed family | 100 | GFW |
| Threshold sensitivity | `300x300` at epsilon `0.5, 0.05, 0.005, 0.0005` for uniform, log-normal, and truncated-normal; partial exponential/integer runs are also retained | 100 | GFW, EPM |
| Supplemental welfare/scaling | Five-step square sizes through `50x50` or `80x80`, plus dense `3x3` through `50x50` sweeps | 1 or 10 | GFW, EPM, COMB |
| Bidding data | `5x5, 50x50, ..., 300x300`, with and without noise | 100 | GFW, EPM |
| GFW vs. simplex pivots | `20x20` and `60x60` | 10 | GFW and the simplex prototype |

Some CSVs have blank COMB or EPM fields because those algorithms were intentionally omitted or did not solve instances at that scale. Runtime columns are machine- and solver-version-dependent, so a rerun should reproduce the experiment design and qualitative plots but not identical wall-clock measurements.

## CSV columns

Simulation CSVs contain one row per problem size. Important column families are:

- `iteration_<algorithm>_a1` and `iteration_<algorithm>_e`: mean iterations/oracle calls for approximate and exact CEEI.
- `...-std`: sample standard deviation over the included seeds.
- `solved_<algorithm>_a1` and `solved_<algorithm>_e`: number of solved seeds, not a ratio. Plotting divides these values by the requested seed count.
- `runningtime_<algorithm>_a1` and `runningtime_<algorithm>_e`: mean seconds to the corresponding stopping condition.

Welfare CSVs contain `utilitarian`, `egalitarian`, and `nash` values for exact solutions. Their summaries use instances on which all three compared algorithms returned an exact equilibrium; `distant_prices` is the mean sum of pairwise Euclidean distances among their price vectors.

## Geometry diagrams

The scripts in `diagram/` reproduce the three committed PNGs when run from that directory:

```bash
cd diagram
python Graph_Bounded.py
python Graph_Unbounded.py
python Graph_GFW.py
```

`Graph_GFW.py` uses Gurobi and LaTeX-rendered labels, so it needs the configured solver license and a LaTeX installation. The other two use NumPy, SciPy, Matplotlib, and CVXPY.
