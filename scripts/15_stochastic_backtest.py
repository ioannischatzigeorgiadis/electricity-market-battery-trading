
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

df = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

errors = pd.read_csv(
    OUT / "ml_error_scenarios_2015.csv",
    index_col=0
).to_numpy(dtype=float)

assert errors.shape == (365, 24)
assert np.isfinite(errors).all()

rng = np.random.default_rng(42)

days = list(df.groupby(df.index.normalize()))

if N_DAYS is not None:
    days = days[:N_DAYS]

results = []

for i, (date, day) in enumerate(days, start=1):

    day = day.sort_index()

    if len(day) != 24 or list(day.index.hour) != list(range(24)):
        raise ValueError(f"Invalid delivery day: {date}")

    actual = day["actual"].to_numpy(dtype=float)
    forecast = day["ml_forecast"].to_numpy(dtype=float)

    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError(f"Missing prices: {date}")

    # Draw complete historical daily error trajectories
    indices = rng.choice(
        len(errors),
        size=N_SCENARIOS,
        replace=False
    )

    scenarios = forecast[None, :] + errors[indices]

    # One shared schedule per strategy across its scenarios
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

        residual = (
            soc[1:] - soc[:-1]
            - battery.ETA * charge * battery.DT
            + discharge * battery.DT / battery.ETA
        )

        tol = 1e-5

        assert np.max(np.abs(residual)) < tol
        assert np.max(np.minimum(charge, discharge)) < tol
        assert soc.min() >= -tol
        assert soc.max() <= battery.E_MAX + tol
        assert charge.max() <= battery.P_MAX + tol
        assert discharge.max() <= battery.P_MAX + tol
        assert abs(soc[0]) < tol
        assert abs(soc[-1]) < tol

        net_power = discharge - charge

        # Realized profit based on actual market prices
        realized = float(
            np.sum(actual * net_power) * battery.DT
        )

        # Estimated profit distribution from scenarios
        scenario_profits = (
            scenarios @ net_power * battery.DT
        )

        tail_n = max(
            1,
            int(round(0.05 * N_SCENARIOS))
        )

        tail_profit = np.sort(scenario_profits)[:tail_n].mean()

        row[f"{name}_profit"] = realized
        row[f"{name}_scenario_tail"] = tail_profit

    results.append(row)

    if i % 25 == 0 or i == len(days):
        print(f"Processed {i}/{len(days)} days", flush=True)

results = pd.DataFrame(results)

print("\n--- STOCHASTIC MILP BACKTEST ---")
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
    "stochastic_backtest_2016_2017.csv"
    if N_DAYS is None
    else "stochastic_smoke_test.csv"
)

results.to_csv(OUT / filename, index=False)

print("\nResults saved:", filename)
