
from pathlib import Path
from importlib import import_module
import sys
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

optimize_battery = import_module(
    "10_battery_optimization"
).optimize_battery

optimize_stochastic = import_module(
    "14_stochastic_cvar"
).optimize_stochastic

PRED_FILE = ROOT / "results" / "ml_predictions_2020.csv"
ERROR_FILE = ROOT / "results" / "ml_error_scenarios_2015.csv"
REF_FILE = ROOT / "results" / "milp_backtest_2020.csv"

# First perform a seven-day smoke test.
# Change to None after the first successful run.
N_DAYS = None

N_SCENARIOS = 100
RANDOM_SEED = 42

P_MAX = 1.0
E_MAX = 2.0
ETA = np.sqrt(0.90)
DT = 1.0
TOL = 1e-4


# -------------------------------------------------
# 1. Load historical calibration errors
# -------------------------------------------------

errors_df = pd.read_csv(ERROR_FILE)

hour_cols = [str(h) for h in range(24)]

if not set(hour_cols).issubset(errors_df.columns):
    raise ValueError("Missing hourly error columns")

errors = errors_df[hour_cols].to_numpy(dtype=float)

if errors.shape != (365, 24):
    raise ValueError(
        f"Unexpected error matrix shape: {errors.shape}"
    )

if not np.isfinite(errors).all():
    raise ValueError("Invalid calibration errors")

print("Historical error trajectories:", errors.shape)


# -------------------------------------------------
# 2. Load 2020 ML predictions
# -------------------------------------------------

pred = pd.read_csv(PRED_FILE)

timestamps = pd.to_datetime(
    pred["local_timestamp"],
    utc=True
)

pred.index = pd.DatetimeIndex(
    timestamps
).tz_convert("Europe/Berlin")

pred = pred.sort_index()

required_cols = [
    "actual",
    "ml_forecast",
    "naive_forecast",
]

if pred[required_cols].isna().any().any():
    raise ValueError("Missing forecast values")

print("2020 forecast observations:", len(pred))


# -------------------------------------------------
# 3. Load frozen deterministic reference
# -------------------------------------------------

reference = pd.read_csv(REF_FILE)

if reference["date"].duplicated().any():
    raise ValueError("Duplicate reference dates")

reference = reference.sort_values("date")

if len(reference) != 357:
    raise ValueError("Expected 357 reference days")

print("Reference MILP days:", len(reference))


# -------------------------------------------------
# 4. Profit and physical feasibility functions
# -------------------------------------------------

def realized_profit(prices, charge, discharge):
    return float(
        np.sum(
            prices * (discharge - charge) * DT
        )
    )


def validate_battery(charge, discharge, soc):
    charge = np.asarray(charge, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    soc = np.asarray(soc, dtype=float)

    if (
        charge.shape != (24,)
        or discharge.shape != (24,)
        or soc.shape != (25,)
    ):
        raise ValueError("Unexpected optimizer output shape")

    if not all(
        np.isfinite(v).all()
        for v in (charge, discharge, soc)
    ):
        raise ValueError("Non-finite optimization result")

    if (
        np.min(charge) < -TOL
        or np.min(discharge) < -TOL
        or np.max(charge) > P_MAX + TOL
        or np.max(discharge) > P_MAX + TOL
    ):
        raise ValueError("Power limits violated")

    if (
        np.min(soc) < -TOL
        or np.max(soc) > E_MAX + TOL
    ):
        raise ValueError("SOC limits violated")

    if abs(soc[0]) > TOL or abs(soc[-1]) > TOL:
        raise ValueError("Initial/final SOC violated")

    if np.any(
        (charge > TOL) & (discharge > TOL)
    ):
        raise ValueError(
            "Simultaneous charging and discharging"
        )

    expected_soc = (
        soc[:-1]
        + ETA * charge * DT
        - discharge * DT / ETA
    )

    if not np.allclose(
        soc[1:],
        expected_soc,
        atol=TOL,
        rtol=0
    ):
        raise ValueError("Energy balance violated")


def evaluate_schedule(
    actual,
    scenarios,
    charge,
    discharge,
    soc
):
    validate_battery(charge, discharge, soc)

    actual_profit = realized_profit(
        actual, charge, discharge
    )

    scenario_profits = scenarios @ (
        np.asarray(discharge) - np.asarray(charge)
    ) * DT

    # Worst 5 of 100 scenario outcomes.
    tail_n = math.ceil(
        0.05 * len(scenario_profits)
    )

    estimated_tail = float(
        np.sort(scenario_profits)[:tail_n].mean()
    )

    return actual_profit, estimated_tail


# -------------------------------------------------
# 5. Independent stochastic backtest
# -------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

groups = {
    date: group.sort_index()
    for date, group in pred.groupby(
        pred.index.strftime("%Y-%m-%d")
    )
}

rows = []

print("\n--- 2020 STOCHASTIC MILP BACKTEST ---")

for item in reference.itertuples(index=False):

    date = str(item.date)

    if date not in groups:
        raise ValueError(f"Missing forecast day: {date}")

    group = groups[date]

    # Same eligible 24-hour days as Script 24.
    if (
        len(group) != 24
        or sorted(group.index.hour.tolist())
        != list(range(24))
    ):
        raise ValueError(
            f"Unexpected day length: {date}"
        )

    actual = group["actual"].to_numpy(dtype=float)
    forecast = group["ml_forecast"].to_numpy(dtype=float)

    # Sample complete historical daily error paths.
    indices = rng.choice(
        len(errors),
        size=N_SCENARIOS,
        replace=False
    )

    scenarios = (
        forecast[None, :] + errors[indices, :]
    )

    # Deterministic ML reference.
    ch, dis, soc = optimize_battery(forecast)

    det_profit, det_tail = evaluate_schedule(
        actual, scenarios, ch, dis, soc
    )

    # Ensure consistency with Script 24.
    if not np.isclose(
        det_profit,
        item.ml_profit,
        atol=0.05,
        rtol=0
    ):
        raise ValueError(
            f"Deterministic reference mismatch: {date}"
        )

    
    result = {
        "date": date,
        "deterministic_profit": det_profit,
        "deterministic_estimated_tail": det_tail,
        "deterministic_charged_mwh": float(
            np.sum(ch) * DT
        ),
        "deterministic_discharged_mwh": float(
            np.sum(dis) * DT
        ),
    }


   
    risk_levels = [
        ("risk_neutral", 0.0),
        ("cvar_025", 0.25),
        ("cvar_100", 1.0),
    ]

    for name, risk_weight in risk_levels:

        ch, dis, soc = optimize_stochastic(
            scenarios,
            risk_weight=risk_weight
        )

        profit, estimated_tail = evaluate_schedule(
            actual, scenarios, ch, dis, soc
        )

        # Save realized economic results
        result[f"{name}_profit"] = profit
        result[f"{name}_estimated_tail"] = estimated_tail

        # Save battery energy throughput
        result[f"{name}_charged_mwh"] = float(
            np.sum(ch) * DT
        )

        result[f"{name}_discharged_mwh"] = float(
            np.sum(dis) * DT
        )

    # Save the completed daily result
    rows.append(result)


    if len(rows) % 25 == 0:
        print("Evaluated days:", len(rows))

    if N_DAYS is not None and len(rows) >= N_DAYS:
        break


# -------------------------------------------------
# 6. Save and summarize results
# -------------------------------------------------

out = pd.DataFrame(rows)

required_columns = [
    "deterministic_profit",
    "risk_neutral_profit",
    "cvar_025_profit",
    "cvar_100_profit",
]

missing = [
    col for col in required_columns
    if col not in out.columns
]

if missing:
    raise ValueError(
        f"Missing result columns: {missing}"
    )

print("All strategy result columns verified.")

filename = (
    "stochastic_smoke_2020.csv"
    if N_DAYS is not None
    else "stochastic_backtest_2020_with_throughput.csv"
)

OUTPUT = ROOT / "results" / filename

out.to_csv(OUTPUT, index=False)

strategies = [
    "deterministic",
    "risk_neutral",
    "cvar_025",
    "cvar_100",
]

print("\n--- REALIZED ECONOMIC PERFORMANCE ---")
print("Evaluated days:", len(out))

for name in strategies:

    profits = out[f"{name}_profit"]

    tail_n = math.ceil(0.05 * len(profits))

    print(f"\n{name}")
    print(f"Total profit: {profits.sum():.2f} EUR")
    print(f"Average daily: {profits.mean():.2f} EUR")
    print(f"Loss days: {(profits < 0).sum()}")
    print(f"Worst day: {profits.min():.2f} EUR")

    print(
        "Realized worst 5% mean:",
        f"{profits.nsmallest(tail_n).mean():.2f} EUR"
    )

    print(
        "Mean estimated scenario tail:",
        f"{out[f'{name}_estimated_tail'].mean():.2f} EUR"
    )

print("\n--- ADVANTAGE VS DETERMINISTIC ---")

baseline = out["deterministic_profit"].sum()

for name in strategies[1:]:

    advantage = out[f"{name}_profit"].sum() - baseline

    print(f"{name}: {advantage:+.2f} EUR")

print("\nSaved:", OUTPUT)
print("Stochastic evaluation completed.")
