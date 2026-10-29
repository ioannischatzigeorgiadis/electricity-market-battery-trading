
from pathlib import Path
from importlib import import_module
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Import our existing battery model
sys.path.insert(0, str(ROOT / "scripts"))
battery = import_module("10_battery_optimization")

# First test only seven days; later change to None
N_DAYS = None

# Read real prices and forecasts
df = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

# Build a fixed schedule from historical training data
historical = pd.read_csv(
    ROOT / "data" / "raw" / "DE.csv",
    index_col=0,
    parse_dates=True
)

historical = historical[
    historical.index.year <= 2015
]

hourly_means = (
    historical.groupby(historical.index.hour)["Price"]
    .mean()
    .reindex(range(24))
)

if hourly_means.isna().any():
    raise ValueError("Missing historical hourly means")

fixed_charge, fixed_discharge, _ = (
    battery.optimize_battery(hourly_means.to_numpy())
)

def realized_profit(actual, charge, discharge):
    return float(
        np.sum(actual * (discharge - charge))
        * battery.DT
    )

def verify_schedule(charge, discharge, soc):
    residual = (
        soc[1:] - soc[:-1]
        - battery.ETA * charge * battery.DT
        + discharge * battery.DT / battery.ETA
    )

    tol = 1e-5

    assert np.max(np.abs(residual)) < tol
    assert np.min(soc) >= -tol
    assert np.max(soc) <= battery.E_MAX + tol
    assert np.max(charge) <= battery.P_MAX + tol
    assert np.max(discharge) <= battery.P_MAX + tol
    assert np.max(np.minimum(charge, discharge)) < tol
    assert abs(soc[0]) < tol
    assert abs(soc[-1]) < tol

# Group prices by delivery day
days = list(df.groupby(df.index.normalize()))

if N_DAYS is not None:
    days = days[:N_DAYS]

results = []

for i, (date, day) in enumerate(days, start=1):

    day = day.sort_index()

    if len(day) != 24 or list(day.index.hour) != list(range(24)):
        raise ValueError(f"Unexpected hourly data: {date}")

    if day[["actual", "ml_forecast", "naive_forecast"]].isna().any().any():
        raise ValueError(f"Missing observations: {date}")

    actual = day["actual"].to_numpy()

    daily_result = {"date": date}

    # Fixed historical-profile schedule
    daily_result["fixed_profit"] = realized_profit(
        actual, fixed_charge, fixed_discharge
    )

    # All other strategies use the same MILP solver
    forecast_sources = {
        "naive": day["naive_forecast"].to_numpy(),
        "ml": day["ml_forecast"].to_numpy(),
        "oracle": actual
    }

    for name, predicted_prices in forecast_sources.items():

        charge, discharge, soc = (
            battery.optimize_battery(predicted_prices)
        )

        verify_schedule(charge, discharge, soc)

        daily_result[f"{name}_profit"] = realized_profit(
            actual, charge, discharge
        )

    results.append(daily_result)

    if i % 50 == 0 or i == len(days):
        print(f"Processed {i}/{len(days)} days", flush=True)

results = pd.DataFrame(results)

# Economic evaluation
print("\n--- MILP BATTERY BACKTEST ---")
print("Days evaluated:", len(results))

oracle_total = results["oracle_profit"].sum()

for name in ["fixed", "naive", "ml", "oracle"]:

    profit = results[f"{name}_profit"]

    print(f"\n{name.upper()}")
    print(f"Total profit: {profit.sum():.2f} EUR")
    print(f"Average daily profit: {profit.mean():.2f} EUR")
    print(f"Loss-making days: {(profit < 0).sum()}")

    if oracle_total > 0:
        print(
            f"Oracle profit capture: "
            f"{100 * profit.sum() / oracle_total:.2f}%"
        )

# Save results
filename = (
    "milp_backtest_2016_2017.csv"
    if N_DAYS is None
    else "milp_smoke_test.csv"
)

results.to_csv(OUT / filename, index=False)

print("\nResults saved:", filename)
