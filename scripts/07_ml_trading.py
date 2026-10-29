
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

# Load the ML and naive forecasts
df = pd.read_csv(
    OUT / "ml_predictions_2016_2017.csv",
    index_col=0,
    parse_dates=True
).sort_index()

eta = 0.90

# Find the best predicted charging/discharging hours
def find_best_trade(prices):
    best_profit = 0.0
    best_pair = None

    for buy in range(23):
        for sell in range(buy + 1, 24):

            profit = eta * prices[sell] - prices[buy]

            if profit > best_profit:
                best_profit = profit
                best_pair = (buy, sell)

    return best_pair


def realized_profit(actual_prices, trade):
    if trade is None:
        return 0.0

    buy, sell = trade

    return (
        eta * actual_prices[sell]
        - actual_prices[buy]
    )


results = []

# Evaluate each day using forecasts available in advance
for date, day in df.groupby(df.index.date):

    day = day.sort_index()

    if len(day) != 24 or list(day.index.hour) != list(range(24)):
        raise ValueError(f"Unexpected hourly structure on {date}")

    if day[["actual", "ml_forecast", "naive_forecast"]].isna().any().any():
        raise ValueError(f"Missing prices on {date}")

    actual = day["actual"].to_numpy()
    ml_forecast = day["ml_forecast"].to_numpy()
    naive_forecast = day["naive_forecast"].to_numpy()

    # Choose trading decisions from predictions
    ml_trade = find_best_trade(ml_forecast)
    naive_trade = find_best_trade(naive_forecast)

    # Evaluate the decisions with actual prices
    ml_profit = realized_profit(actual, ml_trade)
    naive_profit = realized_profit(actual, naive_trade)

    # Fixed-hours benchmark
    fixed_profit = eta * actual[19] - actual[3]

    # Perfect foresight benchmark (upper bound)
    oracle_trade = find_best_trade(actual)
    oracle_profit = realized_profit(actual, oracle_trade)

    results.append({
        "date": date,
        "ml_profit": ml_profit,
        "naive_profit": naive_profit,
        "fixed_profit": fixed_profit,
        "oracle_profit": oracle_profit,
        "ml_no_trade": ml_trade is None,
        "naive_no_trade": naive_trade is None
    })

results = pd.DataFrame(results)

# Save the daily trading results
OUT.mkdir(parents=True, exist_ok=True)
results.to_csv(
    OUT / "trading_comparison_2016_2017.csv",
    index=False
)

print("\n--- TRADING RESULTS: 2016-2017 ---")

oracle_total = results["oracle_profit"].sum()

for strategy in ["fixed", "naive", "ml", "oracle"]:

    profits = results[f"{strategy}_profit"]

    print(f"\nStrategy: {strategy}")
    print(f"Total profit: {profits.sum():.2f} EUR")
    print(f"Average daily profit: {profits.mean():.2f} EUR")
    print(f"Loss-making days: {(profits < 0).sum()}")

    if oracle_total > 0:
        capture = 100 * profits.sum() / oracle_total
        print(f"Profit capture: {capture:.2f}%")

print("\nML no-trade days:",
      results["ml_no_trade"].sum())

print("Naive no-trade days:",
      results["naive_no_trade"].sum())
