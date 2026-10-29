
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Read realized stochastic backtest
df = pd.read_csv(
    OUT / "stochastic_backtest_2016_2017.csv",
    parse_dates=["date"]
)

# Read previous deterministic benchmarks
benchmark = pd.read_csv(
    OUT / "milp_backtest_2016_2017.csv",
    parse_dates=["date"]
)

df = df.merge(
    benchmark[["date", "fixed_profit", "ml_profit"]],
    on="date",
    validate="one_to_one"
).sort_values("date").reset_index(drop=True)

assert len(df) == 731
assert not df["date"].duplicated().any()

strategies = {
    "Fixed": "fixed_profit",
    "Deterministic ML": "deterministic_profit",
    "Risk-Neutral": "risk_neutral_profit",
    "CVaR 0.25": "cvar_025_profit",
    "CVaR 1.00": "cvar_100_profit"
}

assert not df[list(strategies.values())].isna().any().any()

# The deterministic result must match previous MILP backtest
assert np.allclose(
    df["deterministic_profit"],
    df["ml_profit"],
    atol=1e-4
)

print("\n--- REALIZED RISK ANALYSIS ---")

worst_n = int(np.ceil(0.05 * len(df)))

print("Days evaluated:", len(df))
print("Worst-tail days:", worst_n)

for name, column in strategies.items():

    profit = df[column].to_numpy()

    worst_tail = np.sort(profit)[:worst_n]

    print(f"\n{name}")
    print(f"Total profit: {profit.sum():.2f} EUR")
    print(f"Average daily profit: {profit.mean():.2f} EUR")
    print(f"Worst day: {profit.min():.2f} EUR")
    print(f"Loss-making days: {(profit < -1e-6).sum()}")

    print(
        f"Realized worst 5% mean: "
        f"{worst_tail.mean():.2f} EUR"
    )

# Performance by year
df["year"] = df["date"].dt.year

print("\n--- YEARLY PERFORMANCE ---")

yearly = df.groupby("year")[
    list(strategies.values())
].sum()

print(yearly.round(2).to_string())

# Paired moving-block bootstrap
print("\n--- STATISTICAL COMPARISON ---")

baseline = df["deterministic_profit"].to_numpy()

rng = np.random.default_rng(42)

B = 3000
BLOCK = 14
n = len(df)

for name in ["Risk-Neutral", "CVaR 0.25", "CVaR 1.00"]:

    profit = df[strategies[name]].to_numpy()

    difference = profit - baseline

    bootstrap_means = np.empty(B)

    for b in range(B):

        starts = rng.integers(
            0, n,
            size=int(np.ceil(n / BLOCK))
        )

        indices = (
            starts[:, None] + np.arange(BLOCK)
        ) % n

        sample = difference[indices.ravel()[:n]]

        bootstrap_means[b] = sample.mean()

    ci = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    print(f"\n{name} vs Deterministic ML")

    print(
        f"Total profit difference: "
        f"{difference.sum():.2f} EUR"
    )

    print(
        f"Average daily difference: "
        f"{difference.mean():.3f} EUR"
    )

    print(
        f"95% bootstrap CI: "
        f"[{ci[0]:.3f}, {ci[1]:.3f}] EUR/day"
    )

print("\nAnalysis complete.")
