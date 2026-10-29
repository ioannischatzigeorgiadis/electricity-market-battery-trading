
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "results" / "trading_comparison_2016_2017.csv",
    parse_dates=["date"]
).sort_values("date")

# Daily paired profit differences
df["difference"] = df["ml_profit"] - df["fixed_profit"]

differences = df["difference"].to_numpy()

n = len(differences)

print("--- OBSERVED ML ADVANTAGE ---")
print(f"Total improvement: {differences.sum():.2f} EUR")
print(f"Mean daily improvement: {differences.mean():.2f} EUR")

# Moving block bootstrap with circular wrapping
rng = np.random.default_rng(42)

n_bootstrap = 3000
block_size = 14

bootstrap_means = np.empty(n_bootstrap)

for b in range(n_bootstrap):

    n_blocks = int(np.ceil(n / block_size))

    starts = rng.integers(0, n, size=n_blocks)

    indices = (
        starts[:, None] + np.arange(block_size)
    ) % n

    indices = indices.ravel()[:n]

    bootstrap_means[b] = differences[indices].mean()

# 95% bootstrap confidence interval
ci_low, ci_high = np.percentile(
    bootstrap_means, [2.5, 97.5]
)

print("\n--- BLOCK BOOTSTRAP ---")
print(f"Block size: {block_size} days")
print(f"Bootstrap samples: {n_bootstrap}")
print(f"95% CI for mean daily improvement:")
print(f"[{ci_low:.3f}, {ci_high:.3f}] EUR/day")

# Check dependence on extreme positive days
print("\n--- OUTLIER SENSITIVITY ---")

for k in [3, 10, 20]:

    remaining = df.drop(
        df.nlargest(k, "difference").index
    )

    total = remaining["difference"].sum()

    print(
        f"Removing top {k} ML advantage days: "
        f"{total:.2f} EUR total advantage"
    )
