import numpy as np
import pandas as pd

# Load historical electricity prices
df = pd.read_csv("data/raw/DE.csv", index_col=0)
df.index = pd.to_datetime(df.index)

df["date"] = df.index.date
df["hour"] = df.index.hour

# One row per day, one column per hour
daily = df.pivot(
    index="date", columns="hour", values="Price"
).sort_index()

if daily.shape[1] != 24 or daily.isna().any().any():
    raise ValueError("Expected 24 valid hourly prices per day")

# Chronological data split
train = daily.loc[[d.year <= 2015 for d in daily.index]]
test = daily.loc[[d.year >= 2016 for d in daily.index]]

eta = 0.90

# Find the best fixed trading hours using TRAINING data only
strategies = []

for buy in range(23):
    for sell in range(buy + 1, 24):
        profit = eta * train[sell] - train[buy]
        strategies.append((profit.mean(), buy, sell))

_, best_buy, best_sell = max(strategies)

# Apply these hours to every day in the TEST period
test_profit = eta * test[best_sell] - test[best_buy]

# Perfect foresight benchmark over the SAME test days
oracle = np.zeros(len(test))

for buy in range(23):
    for sell in range(buy + 1, 24):
        profit = (
            eta * test[sell].to_numpy()
            - test[buy].to_numpy()
        )
        oracle = np.maximum(oracle, profit)

# Display results
print("Best charging hour:", best_buy)
print("Best discharging hour:", best_sell)
print("Test days:", len(test))

print("\nFixed strategy:")
print("Average daily profit:", test_profit.mean())
print("Total profit:", test_profit.sum())
print("Loss-making days:", (test_profit < 0).sum())

print("\nPerfect foresight benchmark:")
print("Total profit:", oracle.sum())
print("Profit capture (%):",
      100 * test_profit.sum() / oracle.sum())

      # Compare performance separately for 2016 and 2017

comparison = pd.DataFrame({
    "fixed_profit": test_profit.to_numpy(),
    "oracle_profit": oracle
}, index=pd.to_datetime(test.index))

yearly = comparison.groupby(comparison.index.year).sum()

yearly["profit_capture_pct"] = (
    100 * yearly["fixed_profit"] / yearly["oracle_profit"]
)

print("\nYearly performance:")
print(yearly.round(2))