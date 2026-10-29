
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Historical calibration errors
cal = pd.read_csv(
    OUT / "ml_error_scenarios_2015.csv",
    index_col=0,
    parse_dates=True
)

cal.index = cal.index.normalize()
cal.columns = cal.columns.astype(int)
cal = cal.reindex(columns=range(24))

# Test period predictions
pred = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

# Convert test errors to daily 24-hour trajectories
test_dates = []
test_values = []

for date, day in pred.groupby(pred.index.normalize()):

    day = day.sort_index()

    if len(day) != 24 or list(day.index.hour) != list(range(24)):
        raise ValueError(f"Incomplete day: {date}")

    error = (
        day["actual"] - day["ml_forecast"]
    ).to_numpy(dtype=float)

    test_dates.append(date)
    test_values.append(error)

test = pd.DataFrame(
    test_values,
    index=pd.DatetimeIndex(test_dates),
    columns=range(24)
)

# All residuals, but each decision may only
# access observations from its own past
history = pd.concat([cal, test]).sort_index()

assert not history.index.duplicated().any()
assert history.notna().all().all()

# Fixed 2015 calibration interval
fixed_lower, fixed_upper = np.quantile(
    cal.to_numpy().ravel(), [0.05, 0.95]
)

results = []

for date, row in test.iterrows():

    cutoff = date - pd.DateOffset(days=2)
    start = cutoff - pd.DateOffset(days=364)

    past = history.loc[start:cutoff]

    if len(past) < 300:
        raise ValueError(f"Insufficient history: {date}")

    rolling_lower, rolling_upper = np.quantile(
        past.to_numpy().ravel(), [0.05, 0.95]
    )

    actual_errors = row.to_numpy(dtype=float)

    fixed_covered = np.mean(
        (actual_errors >= fixed_lower) &
        (actual_errors <= fixed_upper)
    )

    rolling_covered = np.mean(
        (actual_errors >= rolling_lower) &
        (actual_errors <= rolling_upper)
    )

    results.append({
        "date": date,
        "fixed_coverage": fixed_covered,
        "rolling_coverage": rolling_covered,
        "rolling_width": rolling_upper - rolling_lower
    })

results = pd.DataFrame(results)
results["year"] = results["date"].dt.year

print("\n--- WALK-FORWARD CALIBRATION ---")
print("Window: 365 calendar days")
print("Information gap: 2 days")
print("Target coverage: 90%")

for year, group in results.groupby("year"):

    print(f"\n--- {year} ---")

    print(
        f"Fixed coverage: "
        f"{100 * group['fixed_coverage'].mean():.2f}%"
    )

    print(
        f"Rolling coverage: "
        f"{100 * group['rolling_coverage'].mean():.2f}%"
    )

    print(
        f"Average rolling interval width: "
        f"{group['rolling_width'].mean():.2f} EUR/MWh"
    )

# Monthly diagnostics
monthly = (
    results.set_index("date")
    .resample("MS")[["fixed_coverage", "rolling_coverage"]]
    .mean() * 100
)

print("\n--- WORST SIX ROLLING-COVERAGE MONTHS ---")
print(
    monthly.nsmallest(6, "rolling_coverage")
    .round(2)
    .to_string()
)

results.to_csv(
    OUT / "walkforward_coverage_2016_2017.csv",
    index=False
)

print("\nSaved: walkforward_coverage_2016_2017.csv")
