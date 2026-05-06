"""Post-hoc transaction-cost sensitivity analysis.

The cost model is the standard proportional brokerage formulation:

    r^net_t = r^gross_t - c * TO_t

where c is the one-way cost fraction (e.g. 0.0005 for 5 bps) and TO_t is
the drift-adjusted daily one-way L1 turnover as defined in turnover.py.

Because the gross NAV paths and weight matrices are persisted for every
(strategy, test_year) pair under ``artifacts_Ricky/results/rolling/``,
no retraining is required. The analysis is purely post-hoc.

Correctness notes
-----------------
* ``weights_full`` passed to ``apply_cost`` must include at least one row
  dated before the start of the calendar test year. This ensures that the
  drift weight on the first trading day of the year is computed from the
  actual prior-day position rather than from a cold start.
* The first trading day of each test year is excluded from cost charging
  (it is the renormalisation anchor, so its gross return is by definition
  zero). All subsequent days are charged c * TO_t.
* Cash earns zero return; turnover.py already handles the CASH column by
  assigning zero return to it (see turnover.py:44-45).
* This analysis does NOT retrain the PPO agent. The policy was optimised
  under a zero-cost assumption; a cost-aware agent would reduce turnover
  further. Our results therefore represent a conservative lower bound on
  DRL net-of-cost performance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import compute_performance_metrics
from .turnover import compute_turnover

COST_GRID_BPS: list[int] = [0, 1, 2, 5, 10, 20, 50]
INITIAL_NAV: float = 100_000.0
YEARS: list[int] = list(range(2011, 2021))
STRATS: list[str] = ["drl", "mvo", "ew"]


def apply_cost(
    nav_gross: pd.Series,
    weights_full: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: int,
    year: int,
) -> pd.Series:
    """Apply a proportional one-way cost and return the net-of-cost NAV.

    Parameters
    ----------
    nav_gross:
        Gross NAV series renormalised so ``nav_gross.iloc[0] == INITIAL_NAV``,
        already trimmed to calendar year ``year``.
    weights_full:
        Full post-rebalancing weight matrix for this (strategy, year) window,
        **including the pre-roll burn-in rows** dated before ``year``. This
        ensures the first-day-of-year turnover is computed from the actual
        prior position.
    prices:
        Market-wide price panel (assets + SPY + VIX). Passed to
        ``compute_turnover`` for the drift calculation.
    cost_bps:
        One-way cost in basis points. Pass 0 to return an unchanged copy.
    year:
        Calendar year of the test window (integer).

    Returns
    -------
    pd.Series
        Net-of-cost NAV with the same first date as ``nav_gross`` and the
        same starting value (INITIAL_NAV).
    """
    if cost_bps == 0:
        return nav_gross.copy()

    c = cost_bps / 10_000.0

    # Compute turnover on the full window so that year-boundary drift is
    # derived from the actual end-of-burn-in position, then trim to year.
    turnover_full = compute_turnover(weights_full, prices)
    turnover_yr = turnover_full.loc[turnover_full.index.year == year]

    # Gross daily returns; first element (renormalisation anchor) is NaN.
    gross_ret = nav_gross.pct_change()

    # Align: charge cost only on days where both a gross return and a
    # turnover observation exist (drops the first anchor day automatically).
    common_idx = gross_ret.dropna().index.intersection(turnover_yr.index)
    if common_idx.empty:
        return nav_gross.copy()

    net_ret = gross_ret.loc[common_idx] - c * turnover_yr.loc[common_idx]

    # Reconstruct net NAV from INITIAL_NAV on the first anchor date.
    first_date = nav_gross.index[0]
    net_vals = np.concatenate(
        [[INITIAL_NAV], INITIAL_NAV * (1.0 + net_ret).cumprod().values]
    )
    net_idx = pd.Index([first_date]).append(common_idx)
    return pd.Series(net_vals, index=net_idx, name="nav")


def run_tc_sensitivity(
    base_dir: Path,
    cost_grid_bps: Optional[list[int]] = None,
    years: Optional[list[int]] = None,
    strategies: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run transaction-cost sensitivity over a grid of cost levels.

    Parameters
    ----------
    base_dir:
        Root artifact directory, e.g. ``Path('artifacts_Ricky')``.
    cost_grid_bps:
        One-way costs to evaluate, in basis points.
        Defaults to ``COST_GRID_BPS = [0, 1, 2, 5, 10, 20, 50]``.
    years:
        Test years to include (default: 2011–2020).
    strategies:
        Strategy names to include (default: drl, mvo, ew).

    Returns
    -------
    detail : pd.DataFrame
        Long-format table, one row per (strategy, test_year, cost_bps).
        Columns: strategy, test_year, cost_bps, plus all metrics from
        ``compute_performance_metrics``.
    summary : pd.DataFrame
        Cross-window means: one row per (strategy, cost_bps), columns
        prefixed with ``mean_``.
    """
    if cost_grid_bps is None:
        cost_grid_bps = COST_GRID_BPS
    if years is None:
        years = YEARS
    if strategies is None:
        strategies = STRATS

    roll_dir = base_dir / "results" / "rolling"
    prices = pd.read_csv(
        base_dir / "data" / "market_data.csv",
        index_col=0,
        parse_dates=True,
    )

    rows: list[dict] = []
    for strat in strategies:
        for year in years:
            raw_nav = pd.read_csv(
                roll_dir / str(year) / strat / "nav.csv",
                index_col=0,
                parse_dates=True,
            ).squeeze("columns")

            nav_yr = raw_nav.loc[raw_nav.index.year == year].astype(float)
            if nav_yr.empty:
                continue
            nav_yr = nav_yr / float(nav_yr.iloc[0]) * INITIAL_NAV

            weights_full = pd.read_csv(
                roll_dir / str(year) / strat / "weights.csv",
                index_col=0,
                parse_dates=True,
            )

            for cost_bps in cost_grid_bps:
                net_nav = apply_cost(nav_yr, weights_full, prices, cost_bps, year)
                rows.append({
                    "strategy": strat,
                    "test_year": year,
                    "cost_bps": cost_bps,
                    **compute_performance_metrics(net_nav),
                })

    detail = pd.DataFrame(rows)

    metric_cols = [
        c for c in detail.columns
        if c not in ("strategy", "test_year", "cost_bps")
    ]
    summary = (
        detail.groupby(["strategy", "cost_bps"])[metric_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"mean_{c}" for c in metric_cols})
    )
    return detail, summary


def compute_breakeven(summary: pd.DataFrame) -> dict:
    """Find the cost level at which DRL's Sharpe advantage over EW vanishes.

    Linearly interpolates between the two bracketing cost levels to produce
    a fractional-bps estimate. Returns NaN if DRL never surpasses EW in the
    evaluated range.
    """
    drl = summary[summary.strategy == "drl"].set_index("cost_bps")["mean_sharpe_ratio"]
    ew  = summary[summary.strategy == "ew"].set_index("cost_bps")["mean_sharpe_ratio"]
    mvo = summary[summary.strategy == "mvo"].set_index("cost_bps")["mean_sharpe_ratio"]

    edge_drl_ew  = drl - ew
    edge_drl_mvo = drl - mvo

    def crossover_bps(edge: pd.Series) -> float:
        costs = sorted(edge.index)
        for i in range(len(costs) - 1):
            c0, c1 = costs[i], costs[i + 1]
            e0, e1 = float(edge.loc[c0]), float(edge.loc[c1])
            if e0 >= 0 and e1 < 0:
                # Linear interpolation.
                return float(c0 + (c1 - c0) * e0 / (e0 - e1))
        return float("nan")

    return {
        "drl_vs_ew_breakeven_bps":  crossover_bps(edge_drl_ew),
        "drl_vs_mvo_breakeven_bps": crossover_bps(edge_drl_mvo),
        "drl_vs_ew_edge_at_0bps":   float(edge_drl_ew.get(0, float("nan"))),
        "drl_vs_mvo_edge_at_0bps":  float(edge_drl_mvo.get(0, float("nan"))),
        "drl_vs_ew_edge_at_5bps":   float(edge_drl_ew.get(5, float("nan"))),
        "drl_vs_mvo_edge_at_5bps":  float(edge_drl_mvo.get(5, float("nan"))),
        "drl_vs_ew_edge_at_10bps":  float(edge_drl_ew.get(10, float("nan"))),
        "drl_vs_mvo_edge_at_10bps": float(edge_drl_mvo.get(10, float("nan"))),
    }


def main(base_dir: Optional[Path] = None) -> None:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1] / "artifacts_Ricky"

    print("Running transaction-cost sensitivity analysis...")
    detail, summary = run_tc_sensitivity(base_dir)

    out_dir = base_dir / "results"
    detail.to_csv(out_dir / "tc_sensitivity.csv", index=False)
    summary.to_csv(out_dir / "tc_sensitivity_summary.csv", index=False)
    print(f"  tc_sensitivity.csv         ({len(detail)} rows)")
    print(f"  tc_sensitivity_summary.csv ({len(summary)} rows)")

    be = compute_breakeven(summary)

    print("\nMean Sharpe across 10 windows, by strategy and cost (bps):")
    piv = summary.pivot(index="strategy", columns="cost_bps", values="mean_sharpe_ratio")
    print(piv.to_string(float_format="{:.4f}".format))

    print("\nDRL mean Sharpe advantage over EW and MVO by cost level:")
    drl_s = summary[summary.strategy == "drl"].set_index("cost_bps")["mean_sharpe_ratio"]
    ew_s  = summary[summary.strategy == "ew"].set_index("cost_bps")["mean_sharpe_ratio"]
    mvo_s = summary[summary.strategy == "mvo"].set_index("cost_bps")["mean_sharpe_ratio"]
    print(f"  {'bps':>4}  {'DRL':>8}  {'vs EW':>8}  {'vs MVO':>8}")
    for c in sorted(summary["cost_bps"].unique()):
        c = int(c)
        print(
            f"  {c:>4}  {float(drl_s.get(c,float('nan'))):>8.4f}"
            f"  {float(drl_s.get(c,float('nan'))-ew_s.get(c,float('nan'))):>+8.4f}"
            f"  {float(drl_s.get(c,float('nan'))-mvo_s.get(c,float('nan'))):>+8.4f}"
        )

    print(f"\nBreak-even costs (DRL Sharpe edge → 0):")
    print(f"  vs EW:  {be['drl_vs_ew_breakeven_bps']:.1f} bps")
    print(f"  vs MVO: {be['drl_vs_mvo_breakeven_bps']:.1f} bps  (not meaningful if DRL always > MVO)")


if __name__ == "__main__":
    main()
