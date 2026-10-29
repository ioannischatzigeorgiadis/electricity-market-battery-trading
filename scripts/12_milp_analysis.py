
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "milp_backtest_2016_2017.csv"

# Load MILP backtest results
df = pd.read_csv(DATA, parse_dates=["date"])
df = df.sort_values("date")

strategies = [
    "fixed_profit",
    "naive_profit",
    "ml_profit",
    "oracle_profit"
]

# Verify dataset
assert len(df) == 731
assert not df["date"].duplicated().any()
assert not df[strategies].isna().any().any()

# 1. Yearly profitability
df["year"] = df["date"].dt.year

yearly = df.groupby("year")[strategies].sum()

print("\n--- YEARLY MILP PROFITS ---")
print(yearly.round(2).to_string())

# 2. Risk analysis
print("\n--- DAILY RISK METRICS ---")

worst_n = int(np.ceil(len(df) * 0.05))

for strategy in strategies:

    profits = df[strategy]

    print(f"\n{strategy}")
    print(f"Worst daily profit: {profits.min():.2f}")
    print(f"Loss-making days: {(profits < 0).sum()}")
    print(
        f"Average worst 5%: "
        f"{profits.nsmallest(worst_n).mean():.2f}"
    )

# 3. Paired economic comparison
difference = (
    df["ml_profit"] - df["fixed_profit"]
).to_numpy()

print("\n--- ML VS FIXED ---")
print(f"Total advantage: {difference.sum():.2f}")
print(f"Average daily advantage: {difference.mean():.2f}")
print(f"ML better days: {(difference > 1e-8).sum()}")
print(f"Fixed better days: {(difference < -1e-8).sum()}")

# 4. Circular block bootstrap
rng = np.random.default_rng(42)

n = len(difference)
block_size = 14
bootstrap_means = []

for _ in range(3000):

    starts = rng.integers(
        0, n,
        size=int(np.ceil(n / block_size))
    )

    indices = (
        starts[:, None] + np.arange(block_size)
    ) % n

    sample = difference[indices.ravel()[:n]]

    bootstrap_means.append(sample.mean())

ci = np.percentile(bootstrap_means, [2.5, 97.5])

print("\n--- BOOTSTRAP CONFIDENCE INTERVAL ---")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}] EUR/day")
