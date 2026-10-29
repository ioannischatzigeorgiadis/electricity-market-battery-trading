
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Historical calibration residuals (2015)
calibration = pd.read_csv(
    OUT / "ml_error_scenarios_2015.csv",
    index_col=0
)

calibration.columns = calibration.columns.astype(int)
calibration = calibration.reindex(columns=range(24))

# ML predictions for 2016-2017
predictions = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

test_errors = (
    predictions["actual"] -
    predictions["ml_forecast"]
)

cal_errors = calibration.to_numpy(dtype=float).ravel()

assert np.isfinite(cal_errors).all()
assert np.isfinite(test_errors.to_numpy()).all()

# Overall empirical 90% prediction-error interval
q05, q95 = np.quantile(cal_errors, [0.05, 0.95])

print("\n--- ERROR DISTRIBUTION COMPARISON ---")

for name, errors in [
    ("Calibration 2015", cal_errors),
    ("Test 2016-2017", test_errors.to_numpy())
]:
    print(f"\n{name}")
    print(f"MAE: {np.mean(np.abs(errors)):.3f}")
    print(f"Mean error: {np.mean(errors):.3f}")
    print(f"Standard deviation: {np.std(errors):.3f}")
    print(f"5th percentile: {np.quantile(errors, 0.05):.3f}")
    print(f"95th percentile: {np.quantile(errors, 0.95):.3f}")

print("\n--- GLOBAL 90% ERROR COVERAGE ---")
print(f"Calibration bounds: [{q05:.3f}, {q95:.3f}]")

for year, year_errors in test_errors.groupby(
    test_errors.index.year
):
    values = year_errors.to_numpy()

    below = np.mean(values < q05)
    above = np.mean(values > q95)
    coverage = 1 - below - above

    print(f"\nYear: {year}")
    print(f"Coverage: {100 * coverage:.2f}%")
    print(f"Below lower bound: {100 * below:.2f}%")
    print(f"Above upper bound: {100 * above:.2f}%")

# Hour-specific calibration
hourly_q05 = calibration.quantile(0.05)
hourly_q95 = calibration.quantile(0.95)

hours = test_errors.index.hour.to_numpy()

lower = np.array([hourly_q05[h] for h in hours])
upper = np.array([hourly_q95[h] for h in hours])

values = test_errors.to_numpy()

covered = (values >= lower) & (values <= upper)

print("\n--- HOUR-CONDITIONAL COVERAGE ---")
print(f"Overall hourly coverage: {100 * covered.mean():.2f}%")

hourly_coverage = pd.Series(
    [
        100 * covered[hours == h].mean()
        for h in range(24)
    ],
    index=range(24)
)

print("\nFive hours with lowest coverage:")
print(hourly_coverage.nsmallest(5).round(2).to_string())

print("\nTarget nominal coverage: 90%")
print("\nDiagnostics complete.")
