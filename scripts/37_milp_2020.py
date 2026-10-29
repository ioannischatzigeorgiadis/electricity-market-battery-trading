
from pathlib import Path
from importlib import import_module
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "scripts"))

optimize_battery = import_module(
    "10_battery_optimization"
).optimize_battery

PRED_FILE = ROOT / "results" / "ml_predictions_2020.csv"
OLD_FILE = ROOT / "data" / "raw" / "DE.csv"

# First run only 7 days.
# Later, set to None for the complete backtest.
N_DAYS = None

DT = 1.0

print("Loading 2020 ML predictions...")

df = pd.read_csv(PRED_FILE)

timestamps = pd.to_datetime(
    df["local_timestamp"],
    utc=True
)

df.index = pd.DatetimeIndex(
    timestamps
).tz_convert("Europe/Berlin")

df = df.sort_index()

print("Forecast observations:", len(df))

# ----------------------------------------
# Fixed baseline from historical data
# ----------------------------------------

old = pd.read_csv(
    OLD_FILE,
    index_col=0,
    parse_dates=True
)

train = old.loc[
    old.index.year <= 2015
].copy()

hourly_profile = (
    train["Price"]
    .groupby(train.index.hour)
    .mean()
    .reindex(range(24))
)

if hourly_profile.isna().any():
    raise ValueError("Incomplete historical profile")

fixed_prices = hourly_profile.to_numpy(dtype=float)

fixed_ch, fixed_dis, _ = optimize_battery(
    fixed_prices
)

# ----------------------------------------
# Evaluate realized daily profit
# ----------------------------------------

def realized_profit(actual, charge, discharge):
    return float(
        np.sum(
            actual * (discharge - charge) * DT
        )
    )


def optimize_and_evaluate(forecast, actual):
    charge, discharge, soc = optimize_battery(
        forecast
    )

    if not np.isclose(soc[0], 0.0, atol=1e-5):
        raise ValueError("Invalid initial SOC")

    if not np.isclose(soc[-1], 0.0, atol=1e-5):
        raise ValueError("Invalid final SOC")

    return realized_profit(
        actual, charge, discharge
    )


results = []
skipped = []

for day, group in df.groupby(df.index.date):

    group = group.sort_index()

    # Exclude 23-hour and 25-hour DST days
    # in this directly comparable 24-hour test.
    if (
        len(group) != 24
        or sorted(group.index.hour.tolist())
        != list(range(24))
    ):
        skipped.append(str(day))
        continue

    actual = group["actual"].to_numpy(dtype=float)
    ml = group["ml_forecast"].to_numpy(dtype=float)
    naive = group["naive_forecast"].to_numpy(dtype=float)

    fixed_profit = realized_profit(
        actual, fixed_ch, fixed_dis
    )

    naive_profit = optimize_and_evaluate(
        naive, actual
    )

    ml_profit = optimize_and_evaluate(
        ml, actual
    )

    oracle_profit = optimize_and_evaluate(
        actual, actual
    )

    if oracle_profit + 1e-4 < max(
        fixed_profit, naive_profit, ml_profit
    ):
        raise ValueError(
            f"Oracle consistency check failed: {day}"
        )

    results.append({
        "date": str(day),
        "fixed_profit": fixed_profit,
        "naive_profit": naive_profit,
        "ml_profit": ml_profit,
        "oracle_profit": oracle_profit
    })

    if N_DAYS is not None and len(results) >= N_DAYS:
        break

out = pd.DataFrame(results)

if out.empty:
    raise ValueError("No eligible days")

filename = (
    "milp_smoke_2020.csv"
    if N_DAYS is not None
    else "milp_backtest_2020.csv"
)

output = ROOT / "results" / filename

out.to_csv(output, index=False)

print("\n--- 2020 MILP RESULTS ---")
print("Evaluated days:", len(out))
print("Skipped dates:", skipped)

for column in [
    "fixed_profit",
    "naive_profit",
    "ml_profit",
    "oracle_profit"
]:
    print(
        f"{column}: "
        f"{out[column].sum():.2f} EUR"
    )

print(
    "\nML advantage vs Fixed:",
    round(
        out["ml_profit"].sum()
        - out["fixed_profit"].sum(),
        2
    ),
    "EUR"
)

print("\nSaved:", output)
print("MILP evaluation completed.")
