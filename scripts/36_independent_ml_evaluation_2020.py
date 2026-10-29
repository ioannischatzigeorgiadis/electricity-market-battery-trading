
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]

OLD_DATA = ROOT / "data" / "raw" / "DE.csv"
NEW_DATA = ROOT / "data" / "raw" / "DE_LU_2020_clean.csv"
OUTPUT = ROOT / "results" / "ml_predictions_2020.csv"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
]


def make_features(prices):
    """Reproduce the original Script 06 features."""

    features = pd.DataFrame(index=prices.index)

    features["price_lag_24"] = prices.shift(24)
    features["price_lag_48"] = prices.shift(48)
    features["price_lag_168"] = prices.shift(168)

    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month

    features["hour_sin"] = np.sin(
        2 * np.pi * features["hour"] / 24
    )

    features["hour_cos"] = np.cos(
        2 * np.pi * features["hour"] / 24
    )

    features["target"] = prices

    return features.dropna()


# --------------------------------------------------
# 1. Load original historical dataset
# --------------------------------------------------

print("Loading original training data...")

old = pd.read_csv(OLD_DATA, index_col=0)
old.index = pd.to_datetime(old.index)
old = old.sort_index()

old_prices = pd.to_numeric(old["Price"])

old_features = make_features(old_prices)

train = old_features.loc[
    old_features.index.year <= 2015
].copy()

print("Training observations:", len(train))


# --------------------------------------------------
# 2. Rebuild the frozen original ML model
# --------------------------------------------------

params = {
    "learning_rate": 0.05,
    "max_iter": 150,
    "max_leaf_nodes": 15,
}

model = HistGradientBoostingRegressor(
    **params,
    early_stopping=False,
    random_state=42
)

model.fit(
    train[FEATURE_COLUMNS],
    train["target"]
)

print("Original ML model reconstructed.")
print("Hyperparameters:", params)


# --------------------------------------------------
# 3. Load 2020 data
# --------------------------------------------------

print("\nLoading 2020 SMARD prices...")

new = pd.read_csv(NEW_DATA)

utc = pd.to_datetime(
    new["utc_timestamp"],
    utc=True
)

local = utc.dt.tz_convert("Europe/Berlin")

new_prices = pd.Series(
    pd.to_numeric(
        new["price_eur_per_mwh"],
        errors="raise"
    ).to_numpy(),
    index=pd.DatetimeIndex(local),
    name="Price"
).sort_index()

if len(new_prices) != 8784:
    raise ValueError("Expected 8784 observations")

if new_prices.isna().any():
    raise ValueError("Missing 2020 prices")

if new_prices.index.duplicated().any():
    raise ValueError("Duplicate timestamps")

features_2020 = make_features(new_prices)

print("2020 observations:", len(new_prices))
print("Evaluable observations:", len(features_2020))
print("First evaluated hour:", features_2020.index.min())
print("Last evaluated hour:", features_2020.index.max())


# --------------------------------------------------
# 4. Generate forecasts
# --------------------------------------------------

ml_predictions = model.predict(
    features_2020[FEATURE_COLUMNS]
)

naive_predictions = features_2020[
    "price_lag_24"
].to_numpy()

actual = features_2020["target"].to_numpy()


# --------------------------------------------------
# 5. Evaluate forecasting performance
# --------------------------------------------------

def evaluate(name, predictions):
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f} EUR/MWh")
    print(f"RMSE: {rmse:.4f} EUR/MWh")

    return mae, rmse


print("\n--- INDEPENDENT 2020 TEST ---")

ml_mae, ml_rmse = evaluate(
    "Machine Learning", ml_predictions
)

naive_mae, naive_rmse = evaluate(
    "Naive Forecast", naive_predictions
)

improvement = (
    (naive_mae - ml_mae) / naive_mae
) * 100

print(
    f"\nML MAE improvement vs Naive: "
    f"{improvement:.2f}%"
)


# --------------------------------------------------
# 6. Save predictions
# --------------------------------------------------

results = pd.DataFrame(
    {
        "actual": actual,
        "ml_forecast": ml_predictions,
        "naive_forecast": naive_predictions,
    },
    index=features_2020.index
)

results.index.name = "local_timestamp"

results.to_csv(OUTPUT)

print("\nSaved predictions:", OUTPUT)


# --------------------------------------------------
# 7. Monthly analysis
# --------------------------------------------------

results["ml_absolute_error"] = (
    results["actual"] - results["ml_forecast"]
).abs()

results["naive_absolute_error"] = (
    results["actual"] - results["naive_forecast"]
).abs()

monthly = results.groupby(results.index.month).agg(
    observations=("actual", "size"),
    ml_mae=("ml_absolute_error", "mean"),
    naive_mae=("naive_absolute_error", "mean"),
)

print("\n--- MONTHLY 2020 MAE ---")
print(monthly.round(3).to_string())

print("\nIndependent ML evaluation completed.")
