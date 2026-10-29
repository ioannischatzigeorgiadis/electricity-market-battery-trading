
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Load calibration errors
calibration = pd.read_csv(
    OUT / "ml_error_scenarios_2015.csv",
    index_col=0
).to_numpy(dtype=float).ravel()

q05, q95 = np.quantile(calibration, [0.05, 0.95])

# Load out-of-sample ML predictions
prices = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

errors = prices["actual"] - prices["ml_forecast"]

# Load realized profits
trading = pd.read_csv(
    OUT / "stochastic_backtest_2016_2017.csv",
    parse_dates=["date"]
)

strategies = {
    "Deterministic": "deterministic_profit",
    "Risk-Neutral": "risk_neutral_profit",
    "CVaR 0.25": "cvar_025_profit",
    "CVaR 1.00": "cvar_100_profit"
}

print("\n--- YEARLY ERROR AND RISK DIAGNOSTICS ---")

for year in [2016, 2017]:

    year_errors = errors[
        errors.index.year == year
    ].to_numpy()

    year_trading = trading[
        trading["date"].dt.year == year
    ]

    coverage = np.mean(
        (year_errors >= q05) &
        (year_errors <= q95)
    )

    print(f"\n========== {year} ==========")

    print(f"Forecast MAE: {np.mean(np.abs(year_errors)):.3f}")
    print(f"Error standard deviation: {np.std(year_errors):.3f}")
    print(f"90% interval coverage: {100 * coverage:.2f}%")

    worst_n = int(np.ceil(0.05 * len(year_trading)))

    print("\nRealized strategy risk:")

    for name, column in strategies.items():

        profit = year_trading[column].to_numpy()
        tail = np.sort(profit)[:worst_n]

        print(f"\n{name}")
        print(f"Total profit: {profit.sum():.2f} EUR")
        print(f"Loss-making days: {(profit < -1e-6).sum()}")
        print(f"Worst day: {profit.min():.2f} EUR")
        print(f"Worst 5% mean: {tail.mean():.2f} EUR")

# Examine monthly error coverage
error_frame = pd.DataFrame(
    {
        "error": errors.to_numpy(),
        "abs_error": np.abs(errors.to_numpy()),
        "covered": (
            (errors.to_numpy() >= q05) &
            (errors.to_numpy() <= q95)
        )
    },
    index=errors.index
)

monthly = error_frame.resample("MS").agg(
    coverage=("covered", "mean"),
    mae=("abs_error", "mean"),
    error_std=("error", "std")
)

monthly["coverage"] *= 100

print("\n--- SIX MONTHS WITH LOWEST COVERAGE ---")
print(
    monthly.nsmallest(6, "coverage")
    .round(2)
    .to_string()
)

print("\nDiagnostics complete.")
