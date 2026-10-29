
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "DE.csv"
PRED = ROOT / "results" / "ml_predictions_2016_2017.csv"

# Load original electricity prices
df = pd.read_csv(
    RAW,
    index_col=0,
    parse_dates=True
).sort_index()

prices = df["Price"].astype(float)

# Load stored out-of-sample predictions
pred = pd.read_csv(
    PRED,
    index_col=0,
    parse_dates=True
).sort_index()

print("\n--- 1. DATA INTEGRITY ---")

print("Raw observations:", len(prices))
print("Start:", prices.index.min())
print("End:", prices.index.max())

print("Duplicate timestamps:", prices.index.duplicated().sum())
print("Missing prices:", prices.isna().sum())
print("Monotonic timestamps:", prices.index.is_monotonic_increasing)

# Check timestamps and hourly spacing
time_diff = prices.index.to_series().diff().dropna()

print(
    "Non-1-hour timestamp gaps:",
    int((time_diff != pd.Timedelta(hours=1)).sum())
)

daily_counts = prices.groupby(
    prices.index.normalize()
).size()

print(
    "Days with non-24 observations:",
    int((daily_counts != 24).sum())
)

print("\n--- 2. CHRONOLOGICAL SPLIT ---")

periods = {
    "Training": prices.index.year <= 2014,
    "Validation": prices.index.year == 2015,
    "Test": prices.index.year >= 2016
}

for name, mask in periods.items():
    part = prices.loc[mask]

    print(f"\n{name}")
    print("Observations:", len(part))
    print("First:", part.index.min())
    print("Last:", part.index.max())

print("\n--- 3. FEATURE LAG ALIGNMENT ---")

for lag in [24, 48, 168]:

    lagged_time = prices.index.to_series().shift(lag)

    elapsed = (
        prices.index.to_series() - lagged_time
    ).dropna()

    incorrect = (
        elapsed != pd.Timedelta(hours=lag)
    ).sum()

    print(
        f"Lag {lag}: {int(incorrect)} "
        "timestamp mismatches"
    )

print("\n--- 4. STORED PREDICTIONS ---")

print("Prediction observations:", len(pred))
print("Prediction start:", pred.index.min())
print("Prediction end:", pred.index.max())

print(
    "Prediction dates outside 2016-2017:",
    int((~pred.index.year.isin([2016, 2017])).sum())
)

print("Missing prediction values:")
print(pred.isna().sum().to_string())

missing_timestamps = pred.index.difference(prices.index)

print(
    "Predictions without raw timestamps:",
    len(missing_timestamps)
)

# Recalculate yesterday's price without using test forecasts
lag24_prices = prices.shift(24).reindex(pred.index)

if "naive_forecast" in pred.columns:

    difference = (
        pred["naive_forecast"] - lag24_prices
    ).abs()

    print("\n--- 5. NAIVE FORECAST ALIGNMENT ---")

    print(
        "Maximum difference from 24-hour lag:",
        difference.max()
    )

    print(
        "Matching observations:",
        int((difference < 1e-6).sum()),
        "/",
        len(difference)
    )

# Forecast performance on historical test period
print("\n--- 6. TEST FORECAST PERFORMANCE ---")

for name in ["naive_forecast", "ml_forecast"]:

    error = pred["actual"] - pred[name]

    mae = error.abs().mean()
    rmse = np.sqrt((error**2).mean())

    print(f"\n{name}")
    print(f"MAE: {mae:.3f} EUR/MWh")
    print(f"RMSE: {rmse:.3f} EUR/MWh")

# Important: this checks alignment, not source-code leakage
print("\n--- AUDIT NOTE ---")
print("Data alignment checked.")
print("Model training code must still be inspected.")
print("Market data publication times must be verified.")
