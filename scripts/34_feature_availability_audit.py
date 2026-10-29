
from pathlib import Path
from datetime import timedelta
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "raw"
    / "DE_LU_2020_clean.csv"
)

df = pd.read_csv(INPUT)

timestamps = pd.DatetimeIndex(
    pd.to_datetime(df["utc_timestamp"], utc=True)
).tz_convert("Europe/Berlin")

LAGS = [24, 48, 168]
GATE_HOUR = 12

# Skip first 168 observations, matching
# the original forecasting warm-up.
delivery_times = timestamps[168:]

print("--- DAY-AHEAD FEATURE AVAILABILITY AUDIT ---")
print("Evaluated hours:", len(delivery_times))

for lag in LAGS:

    unavailable = []
    available_count = 0

    for delivery_time in delivery_times:

        # Auction closes at 12:00 local time
        # on the preceding calendar day.
        gate_date = (
            delivery_time.date()
            - timedelta(days=1)
        )

        gate = pd.Timestamp(
            f"{gate_date} {GATE_HOUR:02d}:00:00"
        ).tz_localize("Europe/Berlin")

        # Beginning of the historical price hour
        historical_start = (
            delivery_time
            - pd.Timedelta(hours=lag)
        )

        # The full hour must have finished
        # strictly before the gate closure.
        historical_end = (
            historical_start
            + pd.Timedelta(hours=1)
        )

        if historical_end < gate:
            available_count += 1
        else:
            unavailable.append(delivery_time)

    total = len(delivery_times)
    unavailable_count = len(unavailable)

    print(f"\nLag {lag} hours")
    print("Available:", available_count)
    print("Unavailable:", unavailable_count)
    print(
        "Unavailable percentage:",
        f"{100 * unavailable_count / total:.2f}%"
    )

    if unavailable:
        print("First unavailable examples:")
        for t in unavailable[:3]:
            print(" ", t)

print("\nFeature availability audit completed.")
