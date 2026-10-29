
from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "stochastic_backtest_2019.csv"

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

STRATEGIES = [
    "deterministic",
    "risk_neutral",
    "cvar_025",
    "cvar_100",
]

if len(df) != 356:
    raise ValueError("Expected 356 trading days")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates")

profit_cols = [f"{s}_profit" for s in STRATEGIES]

if df[profit_cols].isna().any().any():
    raise ValueError("Missing daily profits")

print("--- STOCHASTIC ANALYSIS 2019 ---")
print("Evaluated days:", len(df))


# -------------------------------------------
# 1. Overall realized performance
# -------------------------------------------

print("\n--- REALIZED PERFORMANCE ---")

for name in STRATEGIES:
    profits = df[f"{name}_profit"]

    tail_n = math.ceil(0.05 * len(profits))

    print(f"\n{name}")
    print(f"Total profit: {profits.sum():.2f} EUR")
    print(f"Daily average: {profits.mean():.3f} EUR")
    print(f"Loss days: {(profits < 0).sum()}")
    print(f"Worst day: {profits.min():.2f} EUR")
    print(
        f"Worst 5% mean: "
        f"{profits.nsmallest(tail_n).mean():.2f} EUR"
    )


# -------------------------------------------
# 2. Paired block bootstrap
# -------------------------------------------

def bootstrap_ci(
    differences,
    block_size=14,
    n_bootstrap=10000,
    seed=42,
):
    values = np.asarray(differences, dtype=float)
    n = len(values)

    rng = np.random.default_rng(seed)

    n_blocks = math.ceil(n / block_size)
    max_start = n - block_size + 1

    starts = rng.integers(
        0, max_start, size=(n_bootstrap, n_blocks)
    )

    offsets = np.arange(block_size)

    indices = (
        starts[:, :, None]
        + offsets[None, None, :]
    )

    indices = indices.reshape(
        n_bootstrap, -1
    )[:, :n]

    means = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        means, [0.025, 0.975]
    )

    return lower, upper


print("\n--- 14-DAY BLOCK BOOTSTRAP ---")

baseline = df["deterministic_profit"]

for name in STRATEGIES[1:]:

    differences = df[f"{name}_profit"] - baseline

    lower, upper = bootstrap_ci(differences)

    print(f"\n{name} vs deterministic")
    print(
        f"Mean difference: "
        f"{differences.mean():+.3f} EUR/day"
    )
    print(
        f"Total difference: "
        f"{differences.sum():+.2f} EUR"
    )
    print(
        f"95% bootstrap CI: "
        f"[{lower:.3f}, {upper:.3f}] EUR/day"
    )
    print(
        f"Winning days: {(differences > 1e-6).sum()}"
    )
    print(
        f"Losing days: {(differences < -1e-6).sum()}"
    )
    print(
        f"Equal days: {(differences.abs() <= 1e-6).sum()}"
    )

    if lower > 0:
        print("CI entirely above zero")
    elif upper < 0:
        print("CI entirely below zero")
    else:
        print("CI includes zero")


# -------------------------------------------
# 3. Monthly performance
# -------------------------------------------

monthly = df.groupby(
    df["date"].dt.month
)[profit_cols].sum()

for name in STRATEGIES[1:]:
    monthly[f"{name}_advantage"] = (
        monthly[f"{name}_profit"]
        - monthly["deterministic_profit"]
    )

print("\n--- MONTHLY PERFORMANCE ---")
print(monthly.round(2).to_string())


# -------------------------------------------
# 4. Estimated vs realized tail summaries
# -------------------------------------------

print("\n--- TAIL PERFORMANCE ---")

for name in STRATEGIES:

    estimated = df[
        f"{name}_estimated_tail"
    ].mean()

    actual_profits = df[f"{name}_profit"]

    tail_n = math.ceil(
        0.05 * len(actual_profits)
    )

    realized = (
        actual_profits.nsmallest(tail_n).mean()
    )

    print(f"\n{name}")
    print(
        f"Mean daily estimated scenario tail: "
        f"{estimated:.2f} EUR"
    )
    print(
        f"Realized worst 5% of days: "
        f"{realized:.2f} EUR"
    )

print(
    "\nNote: Estimated scenario tails and "
    "realized annual tails are different "
    "statistical quantities."
)

# -------------------------------------------
# 5. Save monthly results
# -------------------------------------------

OUTPUT = (
    ROOT / "results"
    / "stochastic_monthly_2019.csv"
)

monthly.to_csv(OUTPUT)

print("\nSaved:", OUTPUT)
print("Stochastic analysis completed.")
