
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

FILE = ROOT / "data" / "raw" / "SMARD_2020.csv"

df = pd.read_csv(
    FILE,
    sep=";",
    encoding="utf-8-sig",
    dtype=str
)

df.columns = df.columns.str.strip()

price_col = next(
    c for c in df.columns
    if c.startswith("Germany/Luxembourg [€/MWh]")
)

start = pd.to_datetime(
    df["Start date"].str.strip(),
    format="%b %d, %Y %I:%M %p"
)

end = pd.to_datetime(
    df["End date"].str.strip(),
    format="%b %d, %Y %I:%M %p"
)

price = pd.to_numeric(
    df[price_col].str.strip().replace("-", np.nan),
    errors="coerce"
)

print("--- SMARD 2020 TIMESTAMP AUDIT ---")

print("Total CSV rows:", len(df))
print("Price column:", price_col)
print("Missing prices:", price.isna().sum())

print("\nFirst timestamp:", start.min())
print("Last timestamp:", start.max())

print(
    "Duplicated timestamps:",
    start.duplicated().sum()
)

duration = (
    (end - start).dt.total_seconds() / 3600
)

print("\nInterval durations (hours):")
print(duration.value_counts().sort_index().to_string())

mask_2020 = (
    (start >= pd.Timestamp("2020-01-01"))
    & (start < pd.Timestamp("2021-01-01"))
)

print("\nRows with start date in 2020:", mask_2020.sum())
print("Expected full-year hourly rows:", 8784)

print("\n--- DST TRANSITION AUDIT ---")

for date in [
    "2020-03-28",
    "2020-03-29",
    "2020-03-30",
    "2020-10-24",
    "2020-10-25",
    "2020-10-26"
]:
    selected = start[
        start.dt.strftime("%Y-%m-%d") == date
    ]

    hours = selected.dt.strftime("%H:%M").tolist()

    print(f"\n{date}")
    print("Observations:", len(selected))
    print("02:00 occurrences:", hours.count("02:00"))
    print(
        "Hours around transition:",
        [h for h in hours if h in (
            "01:00", "02:00", "03:00", "04:00"
        )]
    )

print("\nTimestamp audit completed.")
