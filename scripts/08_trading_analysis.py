
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "trading_comparison_2016_2017.csv"

# Load daily realized trading profits
df = pd.read_csv(DATA, parse_dates=["date"])

strategies = [
    "fixed_profit",
    "naive_profit",
    "ml_profit",
    "oracle_profit"
]

# Check the input data
if df[strategies].isna().any().any():
    raise ValueError("Missing trading profits")

if df["date"].duplicated().any():
    raise ValueError("Duplicate trading dates")

# 1. Yearly profitability
df["year"] = df["date"].dt.year

yearly = df.groupby("year")[strategies].sum()

print("\n--- YEARLY PROFITS (EUR) ---")
print(yearly.round(2).to_string())

# 2. Daily performance distribution
print("\n--- RISK METRICS ---")

worst_n = max(1, int(np.ceil(len(df) * 0.05)))

for strategy in strategies:
    profits = df[strategy]

    print(f"\n{strategy}")
    print(f"Mean daily profit: {profits.mean():.2f}")
    print(f"Worst daily profit: {profits.min():.2f}")
    print(f"Loss-making days: {(profits < 0).sum()}")
    print(f"Average worst 5%: {profits.nsmallest(worst_n).mean():.2f}")

# 3. Paired comparison: ML vs Fixed
df["ml_advantage"] = df["ml_profit"] - df["fixed_profit"]

print("\n--- ML VS FIXED ---")
print("Days ML outperforms Fixed:",
      (df["ml_advantage"] > 0).sum())

print("Days Fixed outperforms ML:",
      (df["ml_advantage"] < 0).sum())

print("Average daily improvement:",
      round(df["ml_advantage"].mean(), 2))

print("\nTop 5 ML improvements:")
print(
    df.nlargest(5, "ml_advantage")[
        ["date", "ml_advantage"]
    ].to_string(index=False)
)
