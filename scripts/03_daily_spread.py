import pandas as pd

# Load the historical electricity prices
df = pd.read_csv("data/raw/DE.csv", index_col=0)

# Convert the index into timestamps
df.index = pd.to_datetime(df.index)

# Find minimum and maximum prices for each day
daily = df["Price"].groupby(df.index.date).agg(["min", "max"])

# Calculate daily price spreads
daily["spread"] = daily["max"] - daily["min"]

# Display statistical results
print("Number of days:", len(daily))
print("Average daily spread:", daily["spread"].mean())
print("Median daily spread:", daily["spread"].median())
print("Maximum daily spread:", daily["spread"].max())

# Display the five days with the largest spreads
print("\nTop 5 days:")
print(daily.sort_values("spread", ascending=False).head())