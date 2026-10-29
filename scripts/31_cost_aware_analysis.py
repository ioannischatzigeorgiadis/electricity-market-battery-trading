
from pathlib import Path
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results" / "cost_aware_backtest_2019.csv"
OUTPUT = ROOT / "results" / "cost_aware_analysis_2019.csv"

df = pd.read_csv(INPUT, parse_dates=["date"])

required = [
    "date", "scenario", "old_net",
    "cost_aware_net", "charged_mwh",
    "discharged_mwh"
]

if not set(required).issubset(df.columns):
    raise ValueError("Missing required columns")

if df[["date", "scenario"]].duplicated().any():
    raise ValueError("Duplicate scenario/day combinations")

print("--- COST-AWARE STATISTICAL ANALYSIS ---")


def block_bootstrap(values, block_size=14,
                    n_bootstrap=10000, seed=42):

    values = np.asarray(values, dtype=float)
    n = len(values)

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_size)

    starts = rng.integers(
        0, n - block_size + 1,
        size=(n_bootstrap, n_blocks)
    )

    offsets = np.arange(block_size)

    indices = (
        starts[:, :, None] + offsets
    ).reshape(n_bootstrap, -1)[:, :n]

    means = values[indices].mean(axis=1)

    return np.quantile(means, [0.025, 0.975])


summary = []

for scenario, group in df.groupby("scenario"):

    group = group.sort_values("date").copy()

    if len(group) != 356:
        raise ValueError(
            f"Unexpected day count: {scenario}"
        )

    group["difference"] = (
        group["cost_aware_net"] - group["old_net"]
    )

    diff = group["difference"]
    lower, upper = block_bootstrap(diff)

    print(f"\n--- {scenario.upper()} ---")

    print(f"Old total: {group['old_net'].sum():.2f} EUR")
    print(
        f"Cost-aware total: "
        f"{group['cost_aware_net'].sum():.2f} EUR"
    )

    print(f"Total advantage: {diff.sum():+.2f} EUR")
    print(f"Mean advantage: {diff.mean():+.3f} EUR/day")
    print(f"Winning days: {(diff > 1e-5).sum()}")
    print(f"Losing days: {(diff < -1e-5).sum()}")
    print(f"Equal days: {(diff.abs() <= 1e-5).sum()}")

    print(
        f"95% bootstrap CI: "
        f"[{lower:.3f}, {upper:.3f}] EUR/day"
    )

    print(
        "Old loss days:",
        (group["old_net"] < 0).sum()
    )

    print(
        "Cost-aware loss days:",
        (group["cost_aware_net"] < 0).sum()
    )

    tail_n = math.ceil(0.05 * len(group))

    print(
        "Old worst 5% mean:",
        round(group["old_net"].nsmallest(tail_n).mean(), 2)
    )

    print(
        "Cost-aware worst 5% mean:",
        round(
            group["cost_aware_net"]
            .nsmallest(tail_n).mean(), 2
        )
    )

    monthly = group.groupby(
        group["date"].dt.month
    )["difference"].sum()

    print("\nMonthly advantage:")
    print(monthly.round(2).to_string())

    summary.append({
        "scenario": scenario,
        "old_total": group["old_net"].sum(),
        "cost_aware_total": group["cost_aware_net"].sum(),
        "total_advantage": diff.sum(),
        "mean_advantage": diff.mean(),
        "winning_days": int((diff > 1e-5).sum()),
        "losing_days": int((diff < -1e-5).sum()),
        "ci_lower": lower,
        "ci_upper": upper,
    })

pd.DataFrame(summary).to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Cost-aware analysis completed.")
