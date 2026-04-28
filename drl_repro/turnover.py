from __future__ import annotations

import numpy as np
import pandas as pd


def compute_turnover(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Compute daily one-way L1 portfolio turnover with price-drift correction.

    At each step t the portfolio weight drifts passively with asset prices before
    the strategy rebalances. Turnover measures the fraction of the portfolio that
    must actually be traded:

        TO_t = 0.5 * sum_i | w_{i,t} - w_tilde_{i,t} |

    where w_tilde_{i,t} is the weight after price changes but before rebalancing:

        w_tilde_{i,t} = w_{i,t-1} * (1 + r_{i,t}) / sum_j( w_{j,t-1} * (1 + r_{j,t}) )

    and r_{i,t} is the simple return of asset i from t-1 to t.
    CASH is treated as earning zero return.

    This is the standard Grinold-Kahn definition: TO_t in [0, 1] where 1.0 means a
    complete portfolio rotation. It maps directly to proportional transaction costs:
    daily drag ≈ cost_per_unit × TO_t.

    Args:
        weights: Post-rebalancing target weights, shape (T, N).
                 Index must be a DatetimeIndex of trading days.
                 Columns: asset tickers, optionally including 'CASH'.
        prices:  Asset prices. Must contain all non-CASH columns in ``weights``.
                 Index is a DatetimeIndex (may span a wider date range).

    Returns:
        Series of daily one-way L1 turnover indexed to weights.index[1:].
    """
    asset_cols = [c for c in weights.columns if c != "CASH"]
    has_cash = "CASH" in weights.columns

    # Simple returns aligned to the weights dates
    rets = prices[asset_cols].pct_change().reindex(weights.index)
    if has_cash:
        rets = rets.copy()
        rets["CASH"] = 0.0
    rets = rets[weights.columns]  # enforce same column order as weights

    w = weights.to_numpy(dtype=float)   # (T, N)
    r = rets.to_numpy(dtype=float)      # (T, N)

    # Drift: apply t's returns to t-1's target weights
    growth = w[:-1] * (1.0 + r[1:])                            # (T-1, N)
    row_sums = growth.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)          # guard div-by-zero
    w_tilde = growth / row_sums                                 # (T-1, N)

    l1_to = 0.5 * np.abs(w[1:] - w_tilde).sum(axis=1)         # (T-1,)

    return pd.Series(l1_to, index=weights.index[1:], name="l1_turnover")
