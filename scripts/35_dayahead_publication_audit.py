
from pathlib import Path
from collections import Counter
from datetime import timedelta
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "raw"
    / "DE_LU_2020_clean.csv"
)

df = pd.read_csv(INPUT)

timestamps = pd.DatetimeIndex(
    pd.to_datetime(
        df["utc_timestamp"],
        utc=True
    )
).tz_convert("Europe/Berlin")

LAGS = [24, 48, 168]
WARMUP = 168

daily_counts = Counter(
    timestamp.date()
    for timestamp in timestamps
)

print("--- DAY-AHEAD PUBLICATION AUDIT ---")
print("Evaluated hours:", len(timestamps) - WARMUP)

# Historical Day-Ahead prices for delivery day H
# are normally published on H-1.
#
# When bidding for delivery day D at noon D-1,
# Day-Ahead prices for delivery days up to D-1
# are already known.
#
# Therefore historical delivery dates must be
# strictly earlier than the target delivery date.

for lag in LAGS:

    unavailable = []
    unavailable_regular = []
    regular_count = 0

    for i in range(WARMUP, len(timestamps)):

        target = timestamps[i]
        historical = timestamps[i - lag]

        target_date = target.date()
        historical_date = historical.date()

        is_regular_day = (
            daily_counts[target_date] == 24
        )

        if is_regular_day:
            regular_count += 1

        # Prices for earlier delivery dates
        # are published before the target auction.
        available = historical_date < target_date

        if not available:
            unavailable.append(
                (str(target), str(historical))
            )

            if is_regular_day:
                unavailable_regular.append(
                    (str(target), str(historical))
                )

    print(f"\nLag {lag} hours")
    print("Unavailable, all hours:", len(unavailable))
    print(
        "Unavailable, regular 24h days:",
        len(unavailable_regular)
    )
    print("Regular-day observations:", regular_count)

    if unavailable:
        print("First unavailable examples:")

        for target, historical in unavailable[:5]:
            print(
                f"  Target: {target}"
                f" | Historical: {historical}"
            )

    if unavailable_regular:
        raise ValueError(
            f"Lag {lag} unavailable on regular days"
        )

print("\nDay-Ahead publication audit completed.")
