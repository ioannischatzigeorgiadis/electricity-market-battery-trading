
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "raw" / "SMARD_2020.csv"
OUTPUT = ROOT / "data" / "raw" / "DE_LU_2020_clean.csv"

print("--- CLEANING SMARD 2020 DATA ---")

# 1. Read original CSV
df = pd.read_csv(
    INPUT,
    sep=";",
    encoding="utf-8-sig",
    dtype=str
)

df.columns = df.columns.str.strip()

price_col = next(
    col for col in df.columns
    if col.startswith("Germany/Luxembourg [€/MWh]")
)

# 2. Parse local German timestamps
local_naive = pd.to_datetime(
    df["Start date"].str.strip(),
    format="%b %d, %Y %I:%M %p",
    errors="raise"
)

# Crucial: timestamps are already local German time.
# Infer which occurrence of the repeated 02:00 is first.
local_time = local_naive.dt.tz_localize(
    "Europe/Berlin",
    ambiguous="infer",
    nonexistent="raise"
)

utc_time = local_time.dt.tz_convert("UTC")

# 3. Parse prices
prices = pd.to_numeric(
    df[price_col].str.strip().replace("-", np.nan),
    errors="coerce"
)

# 4. Keep only local calendar year 2020
mask = local_time.dt.year == 2020

clean = pd.DataFrame({
    "utc_timestamp": utc_time.astype(str),
    "local_timestamp": local_time.astype(str),
    "local_date": local_time.dt.strftime("%Y-%m-%d"),
    "price_eur_per_mwh": prices
})

clean = clean.loc[mask].copy()

# Sort chronologically by UTC
clean["_utc"] = pd.to_datetime(
    clean["utc_timestamp"],
    utc=True
)

clean = (
    clean.sort_values("_utc")
    .reset_index(drop=True)
)

# 5. Validation
if len(clean) != 8784:
    raise ValueError(
        f"Expected 8784 rows, found {len(clean)}"
    )

if clean["price_eur_per_mwh"].isna().any():
    raise ValueError("Missing price values")

if clean["_utc"].duplicated().any():
    raise ValueError("Duplicate UTC timestamps")

time_diff = clean["_utc"].diff().iloc[1:]

if not (time_diff == pd.Timedelta(hours=1)).all():
    raise ValueError("Non-hourly UTC gaps detected")

daily_counts = (
    clean.groupby("local_date")
    .size()
    .value_counts()
    .sort_index()
)

expected_daily_counts = {
    23: 1,
    24: 364,
    25: 1
}

if daily_counts.to_dict() != expected_daily_counts:
    raise ValueError(
        f"Unexpected daily counts: {daily_counts.to_dict()}"
    )

# 6. Save in compatible format
clean = clean.drop(columns=["_utc"])

clean.to_csv(OUTPUT, index=False)

print("Rows:", len(clean))
print("Missing prices:", clean["price_eur_per_mwh"].isna().sum())
print("Duplicate UTC timestamps:", 0)
print("Non-hourly UTC gaps:", 0)

print("\nDaily observation counts:")
print(daily_counts.to_string())

print("\n--- PRICE STATISTICS ---")
print(
    clean["price_eur_per_mwh"]
    .describe()
    .round(2)
    .to_string()
)

print("\nFirst local timestamp:")
print(clean["local_timestamp"].iloc[0])

print("\nLast local timestamp:")
print(clean["local_timestamp"].iloc[-1])

print("\nSaved:", OUTPUT)
print("2020 cleaning completed.")
