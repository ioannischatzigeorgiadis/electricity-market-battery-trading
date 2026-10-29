
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "DE.csv"
OUT = ROOT / "results"

OUT.mkdir(parents=True, exist_ok=True)

# Load historical electricity prices
df = pd.read_csv(DATA, index_col=0, parse_dates=True)
df = df.sort_index()

prices = df["Price"]

# Recreate the features of our ML model
X = pd.DataFrame(index=prices.index)

X["price_lag_24"] = prices.shift(24)
X["price_lag_48"] = prices.shift(48)
X["price_lag_168"] = prices.shift(168)

X["hour"] = X.index.hour
X["day_of_week"] = X.index.dayofweek
X["month"] = X.index.month

X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

data = X.copy()
data["actual"] = prices
data = data.dropna()

columns = list(X.columns)

# Chronological split
train = data[data.index.year <= 2014]
calibration = data[data.index.year == 2015].copy()

# Fit only on pre-2015 observations
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=150,
    max_leaf_nodes=15,
    early_stopping=False,
    random_state=42
)

model.fit(train[columns], train["actual"])

# Out-of-sample predictions for 2015
calibration["forecast"] = model.predict(
    calibration[columns]
)

# Forecast residuals
calibration["error"] = (
    calibration["actual"] - calibration["forecast"]
)

# Build complete daily error trajectories
calibration["date"] = calibration.index.normalize()
calibration["hour"] = calibration.index.hour

daily_errors = calibration.pivot(
    index="date",
    columns="hour",
    values="error"
).reindex(columns=range(24))

if daily_errors.shape != (365, 24):
    raise ValueError("Expected 365 daily error trajectories")

if daily_errors.isna().any().any():
    raise ValueError("Incomplete daily error trajectories")

# Save error scenarios for stochastic optimization
daily_errors.to_csv(
    OUT / "ml_error_scenarios_2015.csv",
    index_label="date"
)

# Statistical summary
errors = calibration["error"]

print("\n--- FORECAST UNCERTAINTY CALIBRATION ---")
print("Calibration year: 2015")
print("Observations:", len(errors))
print("Daily scenarios:", len(daily_errors))

print(f"\nMAE: {mean_absolute_error(calibration['actual'], calibration['forecast']):.3f}")
print(f"Mean error: {errors.mean():.3f}")
print(f"Error standard deviation: {errors.std():.3f}")
print(f"5th percentile: {errors.quantile(0.05):.3f}")
print(f"95th percentile: {errors.quantile(0.95):.3f}")

print("\nDaily error scenario matrix:")
print(daily_errors.shape)

print("\nSaved: ml_error_scenarios_2015.csv")
