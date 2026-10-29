
from pathlib import Path
from importlib import import_module
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

sys.path.insert(0, str(ROOT / "scripts"))

battery = import_module("10_battery_optimization")
stochastic = import_module("14_stochastic_cvar")

N_DAYS = None
N_SCENARIOS = 100
WINDOW = 365

rng = np.random.default_rng(42)

# Load historical calibration errors
cal = pd.read_csv(
    OUT / "ml_error_scenarios_2015.csv",
    index_col=0,
    parse_dates=True
)

cal.index = cal.index.normalize()
cal.columns = cal.columns.astype(int)
cal = cal.reindex(columns=range(24))

# Load test forecasts
df = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

days = list(df.groupby(df.index.normalize()))

# Construct historical daily residual trajectories
test_errors = {}

for date, day in days:

    day = day.sort_index()

    if len(day) != 24 or list(day.index.hour) != list(range(24)):
        raise ValueError(f"Invalid day: {date}")

    residual = (
        day["actual"] - day["ml_forecast"]
    ).to_numpy(dtype=float)

    test_errors[pd.Timestamp(date)] = residual

test_error_df = pd.DataFrame.from_dict(
    test_errors,
    orient="index",
    columns=range(24)
).sort_index()

# Future residuals are stored but NEVER selected:
# scenario history is restricted separately for each day.
history = pd.concat([cal, test_error_df]).sort_index()

assert not history.index.duplicated().any()
assert np.isfinite(history.to_numpy()).all()

if N_DAYS is not None:
    days = days[:N_DAYS]

results = []

for i, (date, day) in enumerate(days, start=1):

    date = pd.Timestamp(date)

    # Only use information from two or more days ago
    cutoff = date - pd.DateOffset(days=2)
    start = cutoff - pd.DateOffset(days=WINDOW-1)

    past = history.loc[start:cutoff]

    if len(past) < N_SCENARIOS:
        raise ValueError(f"Insufficient past scenarios: {date}")

    # Sample complete 24-hour error trajectories
    indices = rng.choice(
        len(past),
        size=N_SCENARIOS,
        replace=False
    )

    sampled_errors = past.iloc[indices].to_numpy()

    actual = day["actual"].to_numpy(dtype=float)
    forecast = day["ml_forecast"].to_numpy(dtype=float)

    scenarios = forecast[None, :] + sampled_errors

    schedules = {
        "deterministic": battery.optimize_battery(forecast),

        "risk_neutral": stochastic.optimize_stochastic(
            scenarios, risk_weight=0.0
        ),

        "cvar_025": stochastic.optimize_stochastic(
            scenarios, risk_weight=0.25
        ),

        "cvar_100": stochastic.optimize_stochastic(
            scenarios, risk_weight=1.0
        )
    }

    row = {"date": date}

    for name, (charge, discharge, soc) in schedules.items():

        tol = 1e-5

        residual = (
            soc[1:] - soc[:-1]
            - battery.ETA * charge * battery.DT
            + discharge * battery.DT / battery.ETA
        )

        assert np.max(np.abs(residual)) < tol
        assert np.max(np.minimum(charge, discharge)) < tol
        assert soc.min() >= -tol
        assert soc.max() <= battery.E_MAX + tol
        assert charge.max() <= battery.P_MAX + tol
        assert discharge.max() <= battery.P_MAX + tol
        assert abs(soc[0]) < tol
        assert abs(soc[-1]) < tol

        net_power = discharge - charge

        realized = float(
            np.sum(actual * net_power) * battery.DT
        )

        scenario_profit = (
            scenarios @ net_power * battery.DT
        )

        worst_n = max(1, int(round(0.05 * N_SCENARIOS)))

        row[f"{name}_profit"] = realized
        row[f"{name}_scenario_tail"] = (
            np.sort(scenario_profit)[:worst_n].mean()
        )

    results.append(row)

    if i % 25 == 0 or i == len(days):
        print(f"Processed {i}/{len(days)} days", flush=True)

results = pd.DataFrame(results)

print("\n--- ROLLING STOCHASTIC BACKTEST ---")
print("Days evaluated:", len(results))

for name in [
    "deterministic",
    "risk_neutral",
    "cvar_025",
    "cvar_100"
]:

    profit = results[f"{name}_profit"]

    print(f"\n{name.upper()}")
    print(f"Total realized profit: {profit.sum():.2f} EUR")
    print(f"Average daily profit: {profit.mean():.2f} EUR")
    print(f"Loss-making days: {(profit < -1e-6).sum()}")

    print(
        "Average estimated worst 5% scenario profit: "
        f"{results[f'{name}_scenario_tail'].mean():.2f} EUR"
    )

filename = (
    "rolling_stochastic_backtest_2016_2017.csv"
    if N_DAYS is None
    else "rolling_stochastic_smoke_test.csv"
)

results.to_csv(OUT / filename, index=False)

print("\nResults saved:", filename)
