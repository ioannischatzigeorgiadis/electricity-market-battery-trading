
from pathlib import Path
from importlib import import_module
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

optimize_stochastic = import_module(
    "14_stochastic_cvar"
).optimize_stochastic

ERROR_FILE = ROOT / "results" / "ml_error_scenarios_2015.csv"
PRED_FILE = ROOT / "results" / "ml_predictions_2019.csv"
REF_FILE = ROOT / "results" / "milp_backtest_2019.csv"

OUTPUT = ROOT / "results" / "stochastic_seed_sensitivity_2019.csv"

SEEDS = [7, 42, 2026]
N_SCENARIOS = 100
DT = 1.0

# ------------------------------------------
# 1. Load existing data
# ------------------------------------------

errors_df = pd.read_csv(ERROR_FILE)

errors = errors_df[
    [str(h) for h in range(24)]
].to_numpy(dtype=float)

if errors.shape != (365, 24):
    raise ValueError("Unexpected calibration error shape")

pred = pd.read_csv(PRED_FILE)

utc = pd.to_datetime(
    pred["local_timestamp"],
    utc=True
)

pred.index = pd.DatetimeIndex(
    utc
).tz_convert("Europe/Berlin")

pred = pred.sort_index()

groups = {
    date: group
    for date, group in pred.groupby(
        pred.index.strftime("%Y-%m-%d")
    )
}

reference = pd.read_csv(REF_FILE)
reference = reference.sort_values("date")

if len(reference) != 356:
    raise ValueError("Unexpected number of reference days")

# Use all 356 eligible days of 2019.
selected = reference.copy()

print("--- SCENARIO SENSITIVITY AUDIT ---")
print("Selected days:", len(selected))
print("Random seeds:", SEEDS)
print("Scenarios per day:", N_SCENARIOS)


# ------------------------------------------
# 2. Evaluate different scenario samples
# ------------------------------------------

results = []

RISK_LEVELS = [
    ("risk_neutral", 0.0),
    ("cvar_025", 0.25),
    ("cvar_100", 1.0),
]

for seed in SEEDS:

    print(f"\nRunning seed {seed}...")

    rng = np.random.default_rng(seed)

    for item in selected.itertuples(index=False):

        date = str(item.date)
        group = groups[date].sort_index()

        if len(group) != 24:
            raise ValueError(f"Invalid day: {date}")

        actual = group["actual"].to_numpy(dtype=float)

        forecast = group[
            "ml_forecast"
        ].to_numpy(dtype=float)

        indices = rng.choice(
            len(errors),
            size=N_SCENARIOS,
            replace=False
        )

        scenarios = (
            forecast[None, :] + errors[indices, :]
        )

        baseline = float(item.ml_profit)

        row = {
            "seed": seed,
            "date": date,
            "deterministic_profit": baseline,
        }

        for name, risk_weight in RISK_LEVELS:

            charge, discharge, soc = optimize_stochastic(
                scenarios,
                risk_weight=risk_weight
            )

            charge = np.asarray(charge)
            discharge = np.asarray(discharge)
            soc = np.asarray(soc)

            if (
                abs(soc[0]) > 1e-4
                or abs(soc[-1]) > 1e-4
            ):
                raise ValueError("SOC boundary violation")

            profit = float(
                np.sum(
                    actual * (discharge - charge) * DT
                )
            )

            row[f"{name}_profit"] = profit

            row[f"{name}_advantage"] = (
                profit - baseline
            )

        results.append(row)

    print(f"Completed seed {seed}")


# ------------------------------------------
# 3. Summarize results
# ------------------------------------------

out = pd.DataFrame(results)
out.to_csv(OUTPUT, index=False)

print("\n--- RESULTS BY RANDOM SEED ---")

for seed, group in out.groupby("seed"):

    print(f"\nSeed: {seed}")
    print("Days:", len(group))

    for name, _ in RISK_LEVELS:

        advantage = group[
            f"{name}_advantage"
        ]

        print(
            f"{name}: "
            f"{advantage.mean():+.3f} EUR/day, "
            f"total {advantage.sum():+.2f} EUR"
        )


print("\n--- ACROSS-SEED SUMMARY ---")

for name, _ in RISK_LEVELS:

    seed_means = out.groupby("seed")[
        f"{name}_advantage"
    ].mean()

    print(f"\n{name}")
    print(
        f"Mean across seeds: "
        f"{seed_means.mean():+.3f} EUR/day"
    )

    print(
        f"Minimum seed mean: "
        f"{seed_means.min():+.3f} EUR/day"
    )

    print(
        f"Maximum seed mean: "
        f"{seed_means.max():+.3f} EUR/day"
    )

    print(
        "Seeds with positive advantage:",
        int((seed_means > 0).sum()),
        "of",
        len(SEEDS)
    )

print("\nSaved:", OUTPUT)
print("Scenario sensitivity audit completed.")
