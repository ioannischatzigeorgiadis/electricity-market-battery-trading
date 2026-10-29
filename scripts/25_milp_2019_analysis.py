
from pathlib import Path
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "milp_backtest_2019.csv"
MONTHLY_OUTPUT = ROOT / "results" / "milp_monthly_2019.csv"

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

strategies = [
    "fixed_profit",
    "naive_profit",
    "ml_profit",
    "oracle_profit",
]

print("--- 2019 MILP STATISTICAL ANALYSIS ---")
print("Evaluated days:", len(df))

if len(df) != 356:
    raise ValueError("Expected 356 evaluated days")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates found")

if df[strategies].isna().any().any():
    raise ValueError("Missing daily profits")

# 1. Overall economic performance
print("\n--- ECONOMIC PERFORMANCE ---")

for col in strategies:
    profits = df[col]

    tail_n = math.ceil(0.05 * len(profits))
    worst_5_mean = profits.nsmallest(tail_n).mean()

    print(f"\n{col}")
    print(f"Total profit: {profits.sum():.2f} EUR")
    print(f"Average daily profit: {profits.mean():.2f} EUR")
    print(f"Loss days: {(profits < 0).sum()}")
    print(f"Worst day: {profits.min():.2f} EUR")
    print(f"Worst 5% mean: {worst_5_mean:.2f} EUR")

# 2. Paired strategy comparisons
df["ml_vs_fixed"] = (
    df["ml_profit"] - df["fixed_profit"]
)

df["ml_vs_naive"] = (
    df["ml_profit"] - df["naive_profit"]
)

print("\n--- DAILY COMPARISONS ---")

for col in ["ml_vs_fixed", "ml_vs_naive"]:
    differences = df[col]

    print(f"\n{col}")
    print(f"Mean advantage: {differences.mean():.3f} EUR/day")
    print(f"Total advantage: {differences.sum():.2f} EUR")
    print(f"ML winning days: {(differences > 1e-6).sum()}")
    print(f"ML losing days: {(differences < -1e-6).sum()}")
    print(f"Equal days: {(differences.abs() <= 1e-6).sum()}")

# 3. Monthly performance
print("\n--- MONTHLY PERFORMANCE ---")

monthly = df.groupby(df["date"].dt.month).agg(
    days=("date", "size"),
    fixed=("fixed_profit", "sum"),
    naive=("naive_profit", "sum"),
    ml=("ml_profit", "sum"),
    oracle=("oracle_profit", "sum"),
)

monthly["ml_advantage"] = monthly["ml"] - monthly["fixed"]

print(monthly.round(2).to_string())

monthly.to_csv(MONTHLY_OUTPUT)

# 4. Moving-block bootstrap
# Resample paired daily profit differences in 14-day blocks.

def block_bootstrap_ci(
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
        0,
        max_start,
        size=(n_bootstrap, n_blocks)
    )

    offsets = np.arange(block_size)

    indices = (
        starts[:, :, None]
        + offsets[None, None, :]
    )

    indices = indices.reshape(n_bootstrap, -1)[:, :n]

    bootstrap_means = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975]
    )

    return lower, upper


print("\n--- 14-DAY BLOCK BOOTSTRAP ---")

for col in ["ml_vs_fixed", "ml_vs_naive"]:
    lower, upper = block_bootstrap_ci(df[col])

    print(f"\n{col}")
    print(f"Observed mean: {df[col].mean():.3f} EUR/day")
    print(f"95% bootstrap CI: [{lower:.3f}, {upper:.3f}]")

    if lower > 0:
        print("CI entirely above zero")
    elif upper < 0:
        print("CI entirely below zero")
    else:
        print("CI includes zero")

# 5. Oracle capture
capture = (
    df["ml_profit"].sum()
    / df["oracle_profit"].sum()
) * 100

print("\n--- ORACLE CAPTURE ---")
print(f"ML Oracle Capture: {capture:.2f}%")

print("\nSaved monthly analysis:", MONTHLY_OUTPUT)
print("2019 statistical analysis completed.")
