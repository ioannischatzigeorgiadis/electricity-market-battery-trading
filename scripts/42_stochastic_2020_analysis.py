
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results"
    / "stochastic_backtest_2020_with_throughput.csv"
)

OUTPUT = (
    ROOT / "results"
    / "stochastic_analysis_2020.csv"
)

MONTHLY_OUTPUT = (
    ROOT / "results"
    / "stochastic_monthly_2020.csv"
)

STRATEGIES = [
    "risk_neutral",
    "cvar_025",
    "cvar_100",
]

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

required = ["date", "deterministic_profit"] + [
    f"{name}_profit" for name in STRATEGIES
]

if not set(required).issubset(df.columns):
    raise ValueError("Missing required columns")

if len(df) != 357:
    raise ValueError(
        f"Expected 357 days, got {len(df)}"
    )

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates")

if not (df["date"].dt.year == 2020).all():
    raise ValueError("Incorrect evaluation year")

if not np.isfinite(
    df[required[1:]].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid profit values")


def block_bootstrap(
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

    bootstrap_means = values[indices].mean(axis=1)

    return np.quantile(
        bootstrap_means,
        [0.025, 0.975]
    )


print("--- STOCHASTIC 2020 STATISTICAL ANALYSIS ---")
print("Evaluated days:", len(df))
print("Evaluation year:", df["date"].dt.year.unique())

base = df["deterministic_profit"]

print(
    "Deterministic total:",
    f"{base.sum():.2f} EUR"
)

if abs(base.sum() - 18859.17) > 0.10:
    raise ValueError("Unexpected deterministic total")


summary = []

for name in STRATEGIES:

    profits = df[f"{name}_profit"]
    difference = profits - base

    lower, upper = block_bootstrap(difference)

    tail_n = math.ceil(0.05 * len(df))

    print(f"\n--- {name.upper()} ---")

    print(f"Total profit: {profits.sum():.2f} EUR")
    print(f"Total advantage: {difference.sum():+.2f} EUR")
    print(
        f"Daily advantage: "
        f"{difference.mean():+.3f} EUR"
    )

    print(
        f"Winning days: "
        f"{(difference > 1e-5).sum()}"
    )

    print(
        f"Losing days: "
        f"{(difference < -1e-5).sum()}"
    )

    print(
        f"Equal days: "
        f"{(difference.abs() <= 1e-5).sum()}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{lower:.3f}, {upper:.3f}] EUR/day"
    )

    print(
        f"Loss days: "
        f"{(profits < 0).sum()}"
    )

    print(
        f"Worst day: {profits.min():.2f} EUR"
    )

    print(
        f"Worst 5% mean: "
        f"{profits.nsmallest(tail_n).mean():.2f} EUR"
    )

    summary.append({
        "strategy": name,
        "total_profit": profits.sum(),
        "total_advantage": difference.sum(),
        "daily_advantage": difference.mean(),
        "ci_lower": lower,
        "ci_upper": upper,
        "winning_days": int(
            (difference > 1e-5).sum()
        ),
        "losing_days": int(
            (difference < -1e-5).sum()
        ),
        "equal_days": int(
            (difference.abs() <= 1e-5).sum()
        ),
        "loss_days": int((profits < 0).sum()),
        "worst_day": profits.min(),
        "worst_5pct_mean": (
            profits.nsmallest(tail_n).mean()
        ),
    })


# Monthly comparison against deterministic
monthly_df = df[["date"]].copy()

for name in STRATEGIES:
    monthly_df[name] = (
        df[f"{name}_profit"]
        - df["deterministic_profit"]
    )

monthly = monthly_df.groupby(
    monthly_df["date"].dt.strftime("%Y-%m")
)[STRATEGIES].sum()

print("\n--- MONTHLY ADVANTAGE VS DETERMINISTIC ---")
print(monthly.round(2).to_string())

for name in STRATEGIES:
    print(
        f"{name} positive months:",
        int((monthly[name] > 0).sum()),
        "of",
        len(monthly)
    )


pd.DataFrame(summary).to_csv(
    OUTPUT, index=False
)

monthly.to_csv(MONTHLY_OUTPUT)

print("\nSaved:", OUTPUT)
print("Saved:", MONTHLY_OUTPUT)
print("Stochastic statistical analysis completed.")
