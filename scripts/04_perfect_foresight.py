import pandas as pd

# Load electricity market data
df = pd.read_csv("data/raw/DE.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# Battery parameters
eta = 0.90

daily_profits = []

# Analyze each day separately
for date, day in df["Price"].groupby(df.index.date):

    prices = day.to_numpy()
    best_profit = 0.0

    # Select a charging hour
    for buy in range(len(prices) - 1):

        # Select a later discharging hour
        for sell in range(buy + 1, len(prices)):

            profit = eta * prices[sell] - prices[buy]

            if profit > best_profit:
                best_profit = profit

    daily_profits.append(best_profit)

# Results
profits = pd.Series(daily_profits)

print("Average daily profit:", profits.mean(), "EUR")
print("Median daily profit:", profits.median(), "EUR")
print("Maximum daily profit:", profits.max(), "EUR")
print("Total theoretical profit:", profits.sum(), "EUR")
print("Profitable days:", (profits > 0).sum())