
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CAL_FILE = RESULTS / "ml_error_scenarios_2015.csv"
OUTPUT = RESULTS / "uncertainty_diagnostics_2019_2020.csv"

# -----------------------------------------
# 1. Historical calibration errors
# -----------------------------------------

cal = pd.read_csv(CAL_FILE)

HOURS = [str(h) for h in range(24)]

errors_2015 = cal[HOURS].to_numpy(dtype=float)

if errors_2015.shape != (365, 24):
    raise ValueError("Invalid calibration matrix")

if not np.isfinite(errors_2015).all():
    raise ValueError("Invalid calibration errors")

# Empirical quantiles for each hour
q05, q95 = np.quantile(
    errors_2015,
    [0.05, 0.95],
    axis=0
)

hourly_mean = errors_2015.mean(axis=0)

print("--- UNCERTAINTY CALIBRATION AUDIT ---")

print("\n--- 2015 CALIBRATION ERRORS ---")

print(
    "Mean error:",
    f"{errors_2015.mean():+.3f} EUR/MWh"
)

print(
    "MAE:",
    f"{np.abs(errors_2015).mean():.3f} EUR/MWh"
)

print(
    "Error standard deviation:",
    f"{errors_2015.std(ddof=1):.3f} EUR/MWh"
)

print(
    "Largest absolute hourly mean bias:",
    f"{np.abs(hourly_mean).max():.3f} EUR/MWh"
)

print("\nHourly residual means:")

print(
    pd.Series(
        hourly_mean,
        index=range(24)
    ).round(2).to_string()
)


# -----------------------------------------
# 2. Evaluate uncertainty on 2019 and 2020
# -----------------------------------------

summary = []

for year in [2019, 2020]:

    file = RESULTS / f"ml_predictions_{year}.csv"
    df = pd.read_csv(file)

    timestamps = pd.to_datetime(
        df["local_timestamp"],
        utc=True
    ).dt.tz_convert("Europe/Berlin")

    hours = timestamps.dt.hour.to_numpy()

    actual = df["actual"].to_numpy(dtype=float)
    forecast = df["ml_forecast"].to_numpy(dtype=float)

    residuals = actual - forecast

    if not np.isfinite(residuals).all():
        raise ValueError(f"Invalid residuals: {year}")

    # Match calibration quantiles by local hour
    lower = q05[hours]
    upper = q95[hours]

    covered = (
        (residuals >= lower)
        & (residuals <= upper)
    )

    below = residuals < lower
    above = residuals > upper

    coverage = 100 * covered.mean()

    print(f"\n--- {year} RESULTS ---")

    print("Observations:", len(residuals))

    print(
        "Mean error:",
        f"{residuals.mean():+.3f} EUR/MWh"
    )

    print(
        "MAE:",
        f"{np.abs(residuals).mean():.3f} EUR/MWh"
    )

    print(
        "Error standard deviation:",
        f"{residuals.std(ddof=1):.3f} EUR/MWh"
    )

    print(
        "Nominal interval coverage: 90.00%"
    )

    print(
        "Observed interval coverage:",
        f"{coverage:.2f}%"
    )

    print(
        "Below 5th-percentile bound:",
        f"{100 * below.mean():.2f}%"
    )

    print(
        "Above 95th-percentile bound:",
        f"{100 * above.mean():.2f}%"
    )

    monthly = pd.DataFrame({
        "month": timestamps.dt.month.to_numpy(),
        "covered": covered
    }).groupby("month")["covered"].mean() * 100

    print("\nMonthly coverage (%):")
    print(monthly.round(2).to_string())

    summary.append({
        "year": year,
        "observations": len(residuals),
        "mean_error": residuals.mean(),
        "mae": np.abs(residuals).mean(),
        "error_std": residuals.std(ddof=1),
        "coverage_90": coverage,
        "below_lower_pct": 100 * below.mean(),
        "above_upper_pct": 100 * above.mean(),
        "worst_month_coverage": monthly.min(),
    })


# -----------------------------------------
# 3. Save diagnostic results
# -----------------------------------------

out = pd.DataFrame(summary)
out.to_csv(OUTPUT, index=False)

print("\n--- FINAL COMPARISON ---")

print(out.round(3).to_string(index=False))

print("\nSaved:", OUTPUT)
print("Uncertainty calibration audit completed.")
