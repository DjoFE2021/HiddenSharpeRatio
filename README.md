# RMT

Random-matrix-theory confidence bounds on the maximal Sharpe ratio of a large set of assets.

This library implements the methods of *The Hidden Sharpe Ratio* by Chernov,
Kelly, Malamud and Schwab (see [Citation](#citation)).

Given a `T x P` matrix of excess returns (for example managed portfolios), the
target is the maximal squared Sharpe ratio `SR^2 = mu' Sigma^{-1} mu` attainable
from the `P` assets. Its in-sample estimate is strongly biased when `P` is
comparable to, or larger than, `T`. The library computes ridge-regularised
statistics on a grid of penalties `z`, removes their bias with random matrix
theory, and reports:

- **lower bounds** on `SR^2`, from a geometric extension of the series
  `SR^2 = B_0 + B_1 + B_2 + ...`;
- **upper bounds** on `SR^2`, under a no-good-deal condition (eigen-directions
  with small variance cannot carry a large share of `SR^2`);
- **confidence bounds** from a CLT for the estimated statistics, pointwise in
  `z` and simultaneously over the whole grid.

The mathematical setting is summarised in the docstring of `RMT.rmt`
(`src/RMT/rmt/__init__.py`).

## Installation

From the repository root:

```bash
pip install -e .                  # core: numpy (>= 2.0), scipy, pandas
pip install -e ".[portfolios]"    # adds PyTorch, needed by RMT.portfolios
pip install -e ".[test]"          # adds pytest, to run the test suite
```

or, with conda, `conda env create -f environment.yml`. Python 3.10 or later.

On Intel Macs, use the conda environment if you need `RMT.portfolios`: the
newest PyTorch that pip installs there (2.2) does not work with NumPy 2, which
the library requires.

## Quick start

```python
import numpy as np
from RMT.rmt.final_stat import setup_grs
from RMT.utils.constants import Z_EFF_GRID

# 30 years of monthly returns on 200 assets; the true annualised SR^2 is 37.5
rng = np.random.default_rng(0)
R = rng.normal(0.005, 0.04, size=(360, 200))
T, P = R.shape

z_grid = (P / T) * Z_EFF_GRID     # ridge penalties, in units of the average eigenvalue
grs = setup_grs(R, z_grid)        # computes everything

lbs = grs.get_all_lbs(T, annualizer=12)   # one row per grid point
ubs = grs.get_all_ubs(T, annualizer=12)

print(lbs[["full_lb2_inf", "full_lb2_inf_clt"]])   # lower bound and its 97.5% version
print(ubs[["u_2", "u_2_clt"]])                     # upper bound and its 97.5% version
print(lbs["muti_lb2"].iloc[0], ubs["muti_ub2"].iloc[0])  # simultaneous over the grid
```

All values are annualised squared Sharpe ratios (take square roots for Sharpe
ratios). With `annualizer=1` they are per period.

### Main output columns

| Column | Meaning |
| --- | --- |
| `grs_0`, `grs_1`, `grs_2` | GRS statistics `B_0`, `B_1`, `B_2` |
| `full_lb{J}_{k}`, `full_lb{J}_inf` | lower bound of order `J` (1 or 2) with `k` extra geometric terms, or infinitely many |
| `u_{J}` | upper bound of order `J` (0, 1 or 2) at the optimal cut-off |
| `full_ub{J}_{k}`, `full_ub{J}_inf` | upper bound at the cut-off implied by the geometric extension |
| `..._clt` | the same bound minus (lower) or plus (upper) `n_sigma = 1.96` standard errors |
| `muti_lb{J}`, `muti_ub{J}` | bounds valid simultaneously over the whole grid (`muti` = multi-z) |

The docstrings of `GRS.get_all_lbs` and `GRS.get_all_ubs` list every column.

### Managed portfolios from stock-level data

`RMT.portfolios` builds the return matrix from a panel of characteristics and
returns indexed by `(date, id)`:

```python
from RMT.portfolios.preprocess import rank_characteristics
from RMT.portfolios.random_feature_portfolios import build_random_features_portfolios
from RMT.rmt.final_stat import setup_grs
from RMT.utils.constants import Z_EFF_GRID

# panel: DataFrame indexed by (date, id), with return column "r_1" and characteristic columns
panel = rank_characteristics(panel, id_cols=["r_1"])
managed = build_random_features_portfolios(panel, ret_col="r_1", num_seeds=10, n_random_feat_per_seed=100)

T, P = managed.shape
grs = setup_grs(managed.values, (P / T) * Z_EFF_GRID)
```

## Conventions

- Returns are used as given; the sample mean is handled inside the statistics.
- `z_grid` is expressed in units of the average eigenvalue `tr(R'R / T) / P`,
  which makes the results invariant to the scale of the returns.
- Keep the grid small (a few points): the simultaneous bounds draw 1,000,000
  normal vectors of the grid's dimension.

## Package layout

```
src/RMT/
├── rmt/                    GRS statistics, their CLT and the bounds
│   ├── final_stat.py       setup_grs (entry point) and convenience wrappers
│   ├── grs.py              GRS class, assembled from the mixins below
│   ├── grs_theta_mixin.py  theta(z) and B_0, B_1, B_2
│   ├── grs_sigma_mixin.py  pointwise CLT
│   ├── grs_multi_z_mixin.py  joint CLT across the grid, simultaneous bounds
│   ├── grs_bounds_mixin.py lower and upper bounds, result tables
│   ├── resolvent.py        Stieltjes transform and bias term xi(z)
│   ├── ZStar.py            effective ridge Z*(z)
│   └── grs_utils.py        truncated geometric series
├── linalg/                 EigenSystem, covariance repair
├── portfolios/             random-feature portfolios and residualisation (PyTorch), ranks
└── utils/                  constants (Z_EFF_GRID, ...)
tests/                      pytest suite
```

## Tests

```bash
pytest -q
```

The suite takes a few minutes; it runs against `src/` without installing the package.

## Citation

If you use this library, please cite:

> Chernov, Kelly, Malamud and Schwab. *The Hidden Sharpe Ratio*. Working paper.

```bibtex
@unpublished{ChernovKellyMalamudSchwab_HiddenSharpeRatio,
  author = {Chernov and Kelly and Malamud and Schwab},
  title  = {The Hidden Sharpe Ratio},
  note   = {Working paper}
}
```

## License

MIT, see [LICENSE](LICENSE).
