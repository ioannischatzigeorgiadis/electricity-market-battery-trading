
from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "milp_backtest_2020.csv"
OUTPUT = ROOT / "results" / "milp_monthly_2020.csv"

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

STRATEGIES = [
    "fixed_profit",
    "naive_profit",
    "ml_profit",
    "oracle_profit",
]

required = ["date"] + STRATEGIES

if not set(required).issubset(df.columns):
    raise ValueError("Missing required columns")

if len(df) != 357:
    raise ValueError(f"Expected 357 days, got {len(df)}")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates")

if not np.isfinite(
    df[STRATEGIES].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid profit values")


def moving_block_bootstrap(
    values,
    block_size=14,
    n_bootstrap=10000,
    seed=42
):
    values = np.asarray(values, dtype=float)
    n = len(values)

    rng = np.random.default_rng(seed)

    n_blocks = math.ceil(n / block_size)

    starts = rng.integers(
        0,
        n - block_size + 1,
        size=(n_bootstrap, n_blocks)
    )

    indices = (
        starts[:, :, None]
        + np.arange(block_size)
    ).reshape(n_bootstrap, -1)[:, :n]

    means = values[indices].mean(axis=1)

    return np.quantile(
        means,
        [0.025, 0.975]
    )


print("--- 2020 MILP ECONOMIC ANALYSIS ---")
print("Evaluated days:", len(df))

print("\n--- PERFORMANCE BY STRATEGY ---")

tail_n = math.ceil(0.05 * len(df))

for name in STRATEGIES:
    profits = df[name]

    print(f"\n{name}")
    print(f"Total: {profits.sum():.2f} EUR")
    print(f"Daily mean: {profits.mean():.3f} EUR")
    print(f"Loss days: {(profits < 0).sum()}")
    print(f"Worst day: {profits.min():.2f} EUR")
    print(
        "Worst 5% mean:",
        f"{profits.nsmallest(tail_n).mean():.2f} EUR"
    )


print("\n--- PAIRED BOOTSTRAP ---")

comparisons = [
    ("fixed_profit", "ML vs Fixed"),
    ("naive_profit", "ML vs Naive"),
]

for baseline, label in comparisons:

    diff = df["ml_profit"] - df[baseline]

    lower, upper = moving_block_bootstrap(diff)

    print(f"\n{label}")
    print(f"Total advantage: {diff.sum():+.2f} EUR")
    print(f"Daily advantage: {diff.mean():+.3f} EUR")
    print(f"Winning days: {(diff > 1e-5).sum()}")
    print(f"Losing days: {(diff < -1e-5).sum()}")
    print(f"Equal days: {(diff.abs() <= 1e-5).sum()}")

    print(
        "95% bootstrap CI:",
        f"[{lower:.3f}, {upper:.3f}] EUR/day"
    )


print("\n--- MONTHLY PERFORMANCE ---")

monthly = df.groupby(
    df["date"].dt.strftime("%Y-%m")
)[STRATEGIES].sum()

monthly["ml_vs_fixed"] = (
    monthly["ml_profit"]
    - monthly["fixed_profit"]
)

monthly["ml_vs_naive"] = (
    monthly["ml_profit"]
    - monthly["naive_profit"]
)

print(monthly.round(2).to_string())

print(
    "\nMonths ML beats Fixed:",
    int((monthly["ml_vs_fixed"] > 0).sum()),
    "of",
    len(monthly)
)

print(
    "Months ML beats Naive:",
    int((monthly["ml_vs_naive"] > 0).sum()),
    "of",
    len(monthly)
)

monthly.to_csv(OUTPUT)

print("\nSaved:", OUTPUT)
print("2020 economic analysis completed.")
