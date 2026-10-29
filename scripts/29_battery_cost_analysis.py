
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results"
    / "stochastic_backtest_2019_with_throughput.csv"
)

SUMMARY_OUTPUT = (
    ROOT / "results"
    / "battery_cost_summary_2019.csv"
)

DAILY_OUTPUT = (
    ROOT / "results"
    / "battery_cost_daily_2019.csv"
)

STRATEGIES = [
    "deterministic",
    "risk_neutral",
    "cvar_025",
    "cvar_100",
]

# Hypothetical costs: EUR/MWh
COST_SCENARIOS = [
    ("no_costs", 0.0, 0.0),
    ("low", 2.0, 0.25),
    ("medium", 5.0, 0.50),
    ("high", 10.0, 1.00),
]

ROUND_TRIP_EFFICIENCY = 0.90
BATTERY_CAPACITY_MWH = 2.0

print("--- BATTERY COST ANALYSIS 2019 ---")

df = pd.read_csv(INPUT, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

if len(df) != 356:
    raise ValueError("Expected 356 evaluated days")

if df["date"].duplicated().any():
    raise ValueError("Duplicate dates found")

required_columns = []

for name in STRATEGIES:
    required_columns.extend([
        f"{name}_profit",
        f"{name}_charged_mwh",
        f"{name}_discharged_mwh",
    ])

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

if not np.isfinite(
    df[required_columns].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid numerical data")

print("Evaluated days:", len(df))


# -------------------------------------------
# 1. Check physical energy balance
# -------------------------------------------

print("\n--- ENERGY THROUGHPUT AUDIT ---")

for name in STRATEGIES:

    charged = df[f"{name}_charged_mwh"]
    discharged = df[f"{name}_discharged_mwh"]

    if (charged < -1e-5).any():
        raise ValueError("Negative charging energy")

    if (discharged < -1e-5).any():
        raise ValueError("Negative discharging energy")

    discrepancy = (
        discharged
        - ROUND_TRIP_EFFICIENCY * charged
    )

    max_error = discrepancy.abs().max()

    if max_error > 1e-3:
        raise ValueError(
            f"Energy balance mismatch: {name}"
        )

    total_charged = charged.sum()
    total_discharged = discharged.sum()

    equivalent_cycles = (
        total_discharged / BATTERY_CAPACITY_MWH
    )

    print(f"\n{name}")
    print(f"Charged: {total_charged:.2f} MWh")
    print(f"Discharged: {total_discharged:.2f} MWh")
    print(
        f"Equivalent discharge cycles: "
        f"{equivalent_cycles:.2f}"
    )


# -------------------------------------------
# 2. Evaluate cost scenarios
# -------------------------------------------

summary_rows = []
daily_frames = []

print("\n--- COST SENSITIVITY RESULTS ---")

for scenario, degradation_cost, trading_cost in COST_SCENARIOS:

    print(f"\nScenario: {scenario}")
    print(
        f"Degradation: {degradation_cost:.2f} EUR/MWh"
    )
    print(
        f"Trading: {trading_cost:.2f} EUR/MWh"
    )

    scenario_results = {}

    for name in STRATEGIES:

        gross = df[f"{name}_profit"]

        charged = df[f"{name}_charged_mwh"]
        discharged = df[f"{name}_discharged_mwh"]

        degradation_expense = (
            degradation_cost * discharged
        )

        trading_expense = (
            trading_cost * (charged + discharged)
        )

        adjusted = (
            gross
            - degradation_expense
            - trading_expense
        )

        scenario_results[name] = {
            "scenario": scenario,
            "strategy": name,
            "gross_profit_eur": gross.sum(),
            "charged_mwh": charged.sum(),
            "discharged_mwh": discharged.sum(),
            "degradation_cost_eur": (
                degradation_expense.sum()
            ),
            "trading_cost_eur": trading_expense.sum(),
            "adjusted_profit_eur": adjusted.sum(),
            "mean_daily_eur": adjusted.mean(),
            "loss_days": int((adjusted < 0).sum()),
        }

        daily_frames.append(
            pd.DataFrame({
                "date": df["date"],
                "scenario": scenario,
                "strategy": name,
                "gross_profit_eur": gross,
                "adjusted_profit_eur": adjusted,
            })
        )

    baseline = scenario_results[
        "deterministic"
    ]["adjusted_profit_eur"]

    for name in STRATEGIES:

        row = scenario_results[name]

        row["advantage_vs_deterministic_eur"] = (
            row["adjusted_profit_eur"] - baseline
        )

        summary_rows.append(row)

        print(
            f"{name}: "
            f"{row['adjusted_profit_eur']:.2f} EUR "
            f"(vs deterministic: "
            f"{row['advantage_vs_deterministic_eur']:+.2f} EUR)"
        )


# -------------------------------------------
# 3. Break-even cost for Risk-Neutral
# -------------------------------------------

print("\n--- RISK-NEUTRAL BREAK-EVEN ---")

rn = "risk_neutral"
det = "deterministic"

extra_gross = (
    df[f"{rn}_profit"].sum()
    - df[f"{det}_profit"].sum()
)

extra_discharge = (
    df[f"{rn}_discharged_mwh"].sum()
    - df[f"{det}_discharged_mwh"].sum()
)

extra_traded = (
    df[f"{rn}_charged_mwh"].sum()
    + df[f"{rn}_discharged_mwh"].sum()
    - df[f"{det}_charged_mwh"].sum()
    - df[f"{det}_discharged_mwh"].sum()
)

print(f"Extra gross profit: {extra_gross:.2f} EUR")
print(
    f"Extra discharged energy: "
    f"{extra_discharge:.2f} MWh"
)
print(
    f"Extra traded energy: "
    f"{extra_traded:.2f} MWh"
)

if extra_discharge > 1e-9:

    break_even = extra_gross / extra_discharge

    print(
        "Break-even degradation cost "
        "with zero trading fees:"
    )

    print(f"{break_even:.3f} EUR/MWh")

else:
    print(
        "No positive incremental discharge; "
        "simple break-even not applicable."
    )


# -------------------------------------------
# 4. Save results
# -------------------------------------------

summary = pd.DataFrame(summary_rows)
daily = pd.concat(daily_frames, ignore_index=True)

summary.to_csv(SUMMARY_OUTPUT, index=False)
daily.to_csv(DAILY_OUTPUT, index=False)

print("\nSaved:", SUMMARY_OUTPUT)
print("Saved:", DAILY_OUTPUT)
print("Battery cost analysis completed.")
