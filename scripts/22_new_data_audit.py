
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "SMARD_2019.csv"
OUTPUT = ROOT / "data" / "raw" / "DE_LU_2019_clean.csv"

print("Reading SMARD data...")

df = pd.read_csv(
    SOURCE,
    sep=";",
    encoding="utf-8-sig",
    na_values=["-"]
)

df.columns = df.columns.str.strip()

price_col = next(
    c for c in df.columns
    if c.startswith("Germany/Luxembourg [€/MWh]")
)

fmt = "%b %d, %Y %I:%M %p"

raw_time = pd.to_datetime(
    df["Start date"],
    format=fmt
)

# SMARD timestamps use German local time.
# The first repeated 02:00 in October is CEST,
# the second repeated 02:00 is CET.
local = raw_time.dt.tz_localize(
    "Europe/Berlin",
    ambiguous="infer",
    nonexistent="raise"
)

utc = local.dt.tz_convert("UTC")

data = pd.DataFrame({
    "utc_timestamp": utc,
    "local_timestamp": local,
    "price_eur_per_mwh": pd.to_numeric(
        df[price_col],
        errors="coerce"
    )
})

data = data.sort_values(
    "utc_timestamp"
).reset_index(drop=True)

# Select exactly the German local calendar year 2019.
data = data.loc[
    data["local_timestamp"].dt.year == 2019
].copy()

data["local_date"] = (
    data["local_timestamp"].dt.strftime("%Y-%m-%d")
)

print("\n--- SMARD 2019 DATA AUDIT ---")
print("Observations:", len(data))

print("First local:", data["local_timestamp"].min())
print("Last local:", data["local_timestamp"].max())

missing = data["price_eur_per_mwh"].isna().sum()

duplicates = (
    data["utc_timestamp"].duplicated().sum()
)

time_steps = (
    data["utc_timestamp"]
    .diff()
    .dropna()
    .dt.total_seconds()
)

gaps = (time_steps != 3600).sum()

print("Missing prices:", missing)
print("Duplicate UTC timestamps:", duplicates)
print("Non-hourly UTC gaps:", gaps)

daily_counts = data.groupby("local_date").size()

print("\nLocal day lengths:")
print(daily_counts.value_counts().sort_index())

print("\nPrice statistics:")
print(
    data["price_eur_per_mwh"]
    .describe()
    .round(2)
    .to_string()
)

expected_counts = {23: 1, 24: 363, 25: 1}
actual_counts = daily_counts.value_counts().to_dict()

if (
    len(data) != 8760
    or missing != 0
    or duplicates != 0
    or gaps != 0
    or actual_counts != expected_counts
):
    raise ValueError(
        "Data audit failed. Check date range and timestamps."
    )

data.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Audit completed successfully.")
