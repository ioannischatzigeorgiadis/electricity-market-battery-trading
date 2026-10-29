
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "ml_predictions_2016_2017.csv"

# Battery parameters
P_MAX = 1.0       # MW
E_MAX = 2.0       # MWh
ETA = np.sqrt(0.90)
DT = 1.0          # hours


def optimize_battery(prices):
    prices = np.asarray(prices, dtype=float)
    T = len(prices)

    # Decision variables: charge, discharge, SOC, binary mode
    n = 4 * T + 1

    ch = slice(0, T)
    dis = slice(T, 2*T)
    soc = slice(2*T, 3*T+1)
    mode = slice(3*T+1, 4*T+1)

    # Minimize negative trading profit
    c = np.zeros(n)
    c[ch] = prices * DT
    c[dis] = -prices * DT

    lower = np.zeros(n)
    upper = np.ones(n)

    upper[ch] = P_MAX
    upper[dis] = P_MAX
    upper[soc] = E_MAX

    # Empty at the beginning and end of the day
    lower[soc.start] = upper[soc.start] = 0
    lower[soc.stop-1] = upper[soc.stop-1] = 0

    # Binary variables enforce exclusive operating mode
    integrality = np.zeros(n, dtype=int)
    integrality[mode] = 1

    A_eq = np.zeros((T, n))
    A_ub = np.zeros((2*T, n))
    b_ub = np.zeros(2*T)

    for t in range(T):

        # Battery energy balance
        A_eq[t, soc.start+t+1] = 1
        A_eq[t, soc.start+t] = -1
        A_eq[t, ch.start+t] = -ETA * DT
        A_eq[t, dis.start+t] = DT / ETA

        # Charging allowed only in charging mode
        A_ub[t, ch.start+t] = 1
        A_ub[t, mode.start+t] = P_MAX
        b_ub[t] = P_MAX

        # Discharging allowed only in discharging mode
        A_ub[T+t, dis.start+t] = 1
        A_ub[T+t, mode.start+t] = -P_MAX

    constraints = [
        LinearConstraint(A_eq, 0, 0),
        LinearConstraint(A_ub, -np.inf, b_ub)
    ]

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"mip_rel_gap": 1e-8}
    )

    if not result.success:
        raise RuntimeError(result.message)

    return (
        result.x[ch],
        result.x[dis],
        result.x[soc]
    )

if __name__ == "__main__":
    # First experiment: one day only
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    day = df.loc["2016-01-01"].sort_index()

    if len(day) != 24:
        raise ValueError("Expected 24 hourly observations")

    actual = day["actual"].to_numpy()
    forecast = day["ml_forecast"].to_numpy()

    # Optimize using ML predictions
    charge, discharge, soc = optimize_battery(forecast)

    # Evaluate using actual market prices
    realized_profit = np.sum(
        actual * (discharge - charge) * DT
    )

    # Optimize using perfect information for comparison
    oracle_ch, oracle_dis, _ = optimize_battery(actual)

    oracle_profit = np.sum(
        actual * (oracle_dis - oracle_ch) * DT
    )

    print("\n--- BATTERY OPTIMIZATION ---")
    print("Date: 2016-01-01")
    print(f"Realized ML profit: {realized_profit:.2f} EUR")
    print(f"Perfect foresight profit: {oracle_profit:.2f} EUR")

    schedule = pd.DataFrame({
        "Hour": day.index.hour,
        "Actual_Price": actual,
        "ML_Forecast": forecast,
        "Charging_MW": charge,
        "Discharging_MW": discharge,
        "SOC_MWh": soc[1:]
    })

    print("\nOptimal ML-based schedule:")
    print(schedule.round(3).to_string(index=False))

    print("\nFinal SOC:", round(soc[-1], 6), "MWh")


    # --- BATTERY MODEL VALIDATION ---

    # Check energy balance at every time step
    energy_residual = (
        soc[1:] - soc[:-1]
        - ETA * charge * DT
        + discharge * DT / ETA
    )

    # Check simultaneous charging and discharging
    overlap = np.minimum(charge, discharge)

    print("\n--- MODEL VALIDATION ---")

    print("Maximum energy balance error:",
        np.max(np.abs(energy_residual)))

    print("Maximum simultaneous operation:",
        np.max(overlap))

    print("Minimum SOC:", soc.min())
    print("Maximum SOC:", soc.max())

    # Numerical tolerance
    tol = 1e-6

    assert np.max(np.abs(energy_residual)) < tol
    assert np.max(overlap) < tol
    assert soc.min() >= -tol
    assert soc.max() <= E_MAX + tol
    assert np.max(charge) <= P_MAX + tol
    assert np.max(discharge) <= P_MAX + tol
    assert abs(soc[0]) < tol
    assert abs(soc[-1]) < tol

    print("All battery constraints verified successfully!")
