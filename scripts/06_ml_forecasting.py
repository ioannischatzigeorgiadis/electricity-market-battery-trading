
from pathlib import Path
import numpy as np
import pandas as pd

# Define project paths
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "DE.csv"

# Load historical electricity prices
df = pd.read_csv(DATA, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

prices = df["Price"]

# Create a feature table
features = pd.DataFrame(index=prices.index)

# Historical market prices (EUR/MWh)
features["price_lag_24"] = prices.shift(24)
features["price_lag_48"] = prices.shift(48)
features["price_lag_168"] = prices.shift(168)

# Calendar features
features["hour"] = features.index.hour
features["day_of_week"] = features.index.dayofweek
features["month"] = features.index.month

# Encode hour as a circular quantity
features["hour_sin"] = np.sin(
    2 * np.pi * features["hour"] / 24
)
features["hour_cos"] = np.cos(
    2 * np.pi * features["hour"] / 24
)

# Add target electricity price
features["target"] = prices

# Remove rows with missing historical inputs
features = features.dropna()

print("Total observations:", len(features))
print("\nFirst five rows:")
print(features.head().round(2))

print("\nTraining observations:",
      len(features[features.index.year <= 2014]))

print("Validation observations:",
      len(features[features.index.year == 2015]))

print("Test observations:",
      len(features[features.index.year >= 2016]))


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define input features
X_cols = [
    "price_lag_24",
    "price_lag_48",
    "price_lag_168",
    "hour",
    "day_of_week",
    "month",
    "hour_sin",
    "hour_cos",
]

# Split the data chronologically
train = features[features.index.year <= 2014]
validation = features[features.index.year == 2015]
test = features[features.index.year >= 2016]

# Candidate model configurations
candidates = [
    {"learning_rate": 0.05, "max_iter": 150, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_iter": 250, "max_leaf_nodes": 31},
    {"learning_rate": 0.10, "max_iter": 150, "max_leaf_nodes": 15},
]

best_mae = float("inf")
best_params = None

print("\n--- VALIDATION RESULTS (2015) ---")

for params in candidates:

    model = HistGradientBoostingRegressor(
        **params,
        early_stopping=False,
        random_state=42
    )

    model.fit(train[X_cols], train["target"])

    predictions = model.predict(validation[X_cols])

    mae = mean_absolute_error(
        validation["target"], predictions
    )

    print(f"Parameters: {params}, MAE: {mae:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_params = params.copy()

print("\nBest parameters:", best_params)
print("Best validation MAE:", round(best_mae, 4))

# Retrain on 2012-2015 after model selection
train_final = features[features.index.year <= 2015]

final_model = HistGradientBoostingRegressor(
    **best_params,
    early_stopping=False,
    random_state=42
)

final_model.fit(
    train_final[X_cols],
    train_final["target"]
)

# Final evaluation on unseen test data
test_predictions = final_model.predict(test[X_cols])

ml_mae = mean_absolute_error(
    test["target"], test_predictions
)

ml_rmse = np.sqrt(
    mean_squared_error(test["target"], test_predictions)
)

# Compare with previous-day naive forecasting
naive_predictions = test["price_lag_24"]

naive_mae = mean_absolute_error(
    test["target"], naive_predictions
)

naive_rmse = np.sqrt(
    mean_squared_error(test["target"], naive_predictions)
)

print("\n--- FINAL TEST RESULTS (2016-2017) ---")
print(f"ML MAE:       {ml_mae:.4f} EUR/MWh")
print(f"ML RMSE:      {ml_rmse:.4f} EUR/MWh")
print(f"Naive MAE:    {naive_mae:.4f} EUR/MWh")
print(f"Naive RMSE:   {naive_rmse:.4f} EUR/MWh")


# --- GENERALIZATION AUDIT ---

print("\n--- GENERALIZATION AUDIT ---")

# Recreate the selected model using only 2012-2014.
# Do not use the variable 'model' from the candidate loop,
# because it refers to the last candidate, not necessarily
# the best candidate.

selected_model = HistGradientBoostingRegressor(
    **best_params,
    early_stopping=False,
    random_state=42
)

selected_model.fit(
    train[X_cols],
    train["target"]
)


def evaluate_model(label, model, dataset):
    y_true = dataset["target"]
    y_pred = model.predict(dataset[X_cols])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{label}")
    print(f"Observations: {len(dataset)}")
    print(f"MAE: {mae:.4f} EUR/MWh")
    print(f"RMSE: {rmse:.4f} EUR/MWh")

    return mae


# Same model: in-sample vs out-of-sample
train_mae = evaluate_model(
    "Training 2012-2014",
    selected_model,
    train
)

validation_mae = evaluate_model(
    "Validation 2015",
    selected_model,
    validation
)

print("\nValidation - Training MAE gap:")
print(f"{validation_mae - train_mae:.4f} EUR/MWh")


# Final model: trained on 2012-2015
final_train_mae = evaluate_model(
    "Final training 2012-2015",
    final_model,
    train_final
)

# Test performance by year
for year in [2016, 2017]:

    year_data = test[test.index.year == year]

    evaluate_model(
        f"Test {year}",
        final_model,
        year_data
    )

print("\nGeneralization audit complete.")


# Save predictions for subsequent experiments
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame({
    "actual": test["target"],
    "ml_forecast": test_predictions,
    "naive_forecast": naive_predictions
})

results.to_csv(OUT / "ml_predictions_2016_2017.csv")

print("\nPredictions saved successfully.")
