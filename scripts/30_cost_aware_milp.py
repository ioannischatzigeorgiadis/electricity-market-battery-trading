
from pathlib import Path
from importlib import import_module
import sys

import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

optimize_battery = import_module(
    "10_battery_optimization"
).optimize_battery

PRED_FILE = ROOT / "results" / "ml_predictions_2019.csv"
REF_FILE = ROOT / "results" / "milp_backtest_2019.csv"

T = 24
P_MAX = 1.0
E_MAX = 2.0
ETA = np.sqrt(0.90)
DT = 1.0

N_DAYS = None

COST_SCENARIOS = [
    ("none", 0.0, 0.0),
    ("medium", 5.0, 0.50),
    ("high", 10.0, 1.00),
]


def optimize_cost_aware(prices, c_deg, c_trade):
    """Cost-aware MILP with the original battery limits."""

    prices = np.asarray(prices, dtype=float)

    if prices.shape != (T,):
        raise ValueError("Expected 24 forecast prices")

    # Variables:
    # charge[0:24]
    # discharge[24:48]
    # soc[48:73]
    # binary mode[73:97]

    n = 4 * T + 1

    c = np.zeros(n)

    # SciPy minimizes, so negate the net profit.
    c[:T] = (prices + c_trade) * DT
    c[T:2*T] = (
        -prices + c_deg + c_trade
    ) * DT

    lower = np.zeros(n)
    upper = np.ones(n)

    upper[:2*T] = P_MAX
    upper[2*T:3*T+1] = E_MAX

    # Initial and final SOC = 0
    upper[2*T] = 0.0
    upper[3*T] = 0.0

    integrality = np.zeros(n, dtype=int)
    integrality[3*T+1:] = 1

    # 24 energy balances + 48 binary-mode constraints
    A = np.zeros((3*T, n))

    lb = np.full(3*T, -np.inf)
    ub = np.full(3*T, np.inf)

    for t in range(T):

        # SOC[t+1] - SOC[t]
        # - eta*charge + discharge/eta = 0

        A[t, 2*T+t+1] = 1.0
        A[t, 2*T+t] = -1.0

        A[t, t] = -ETA * DT
        A[t, T+t] = DT / ETA

        lb[t] = 0.0
        ub[t] = 0.0

        # Charge <= Pmax * (1 - z)
        A[T+t, t] = 1.0
        A[T+t, 3*T+1+t] = P_MAX

        ub[T+t] = P_MAX

        # Discharge <= Pmax * z
        A[2*T+t, T+t] = 1.0
        A[2*T+t, 3*T+1+t] = -P_MAX

        ub[2*T+t] = 0.0

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A, lb, ub)
    )

    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"MILP failed: {result.message}"
        )

    x = result.x

    ch = x[:T]
    dis = x[T:2*T]
    soc = x[2*T:3*T+1]

    # Physical feasibility checks
    expected_soc = (
        soc[:-1] + ETA*ch*DT - dis*DT/ETA
    )

    if not np.allclose(
        soc[1:], expected_soc, atol=1e-5, rtol=0
    ):
        raise ValueError("Energy balance mismatch")

    if (
        np.any(ch > 1e-5)
        and np.any(dis > 1e-5)
        and np.any((ch > 1e-5) & (dis > 1e-5))
    ):
        raise ValueError("Simultaneous operation")

    return ch, dis, soc


def calculate_net(prices, ch, dis, c_deg, c_trade):

    charged = float(np.sum(ch) * DT)
    discharged = float(np.sum(dis) * DT)

    gross = float(
        np.sum(prices * (dis - ch) * DT)
    )

    net = (
        gross
        - c_deg * discharged
        - c_trade * (charged + discharged)
    )

    return gross, net, charged, discharged


# ----------------------------------------
# Load predictions
# ----------------------------------------

pred = pd.read_csv(PRED_FILE)

utc = pd.to_datetime(
    pred["local_timestamp"],
    utc=True
)

pred.index = pd.DatetimeIndex(
    utc
).tz_convert("Europe/Berlin")

groups = {
    date: group.sort_index()
    for date, group in pred.groupby(
        pred.index.strftime("%Y-%m-%d")
    )
}

reference = pd.read_csv(REF_FILE)
reference = reference.sort_values("date")

selected = (
    reference.head(N_DAYS)
    if N_DAYS is not None
    else reference
)

rows = []

print("--- COST-AWARE MILP SMOKE TEST ---")

for item in selected.itertuples(index=False):

    date = str(item.date)
    group = groups[date]

    if len(group) != 24:
        raise ValueError(f"Invalid day: {date}")

    actual = group["actual"].to_numpy(dtype=float)
    forecast = group["ml_forecast"].to_numpy(dtype=float)

    # Original gross-profit optimization
    old_ch, old_dis, old_soc = optimize_battery(
        forecast
    )

    for label, c_deg, c_trade in COST_SCENARIOS:

        # New cost-aware optimization
        ch, dis, soc = optimize_cost_aware(
            forecast, c_deg, c_trade
        )

        old_gross, old_net, _, _ = calculate_net(
            actual, old_ch, old_dis,
            c_deg, c_trade
        )

        new_gross, new_net, charged, discharged = (
            calculate_net(
                actual, ch, dis, c_deg, c_trade
            )
        )

        # The optimized FORECAST net profit
        # must not be worse than the original
        # feasible schedule under the same costs.

        _, old_pred_net, _, _ = calculate_net(
            forecast, old_ch, old_dis,
            c_deg, c_trade
        )

        _, new_pred_net, _, _ = calculate_net(
            forecast, ch, dis,
            c_deg, c_trade
        )

        if new_pred_net < old_pred_net - 0.05:
            raise ValueError(
                f"Objective consistency failed: {date}"
            )

        rows.append({
            "date": date,
            "scenario": label,
            "old_net": old_net,
            "cost_aware_net": new_net,
            "old_gross": old_gross,
            "cost_aware_gross": new_gross,
            "charged_mwh": charged,
            "discharged_mwh": discharged,
        })

out = pd.DataFrame(rows)

print("Evaluated days:", len(selected))

print("\n--- RESULTS ---")

summary = out.groupby("scenario").agg(
    old_net=("old_net", "sum"),
    cost_aware_net=("cost_aware_net", "sum"),
    charged_mwh=("charged_mwh", "sum"),
    discharged_mwh=("discharged_mwh", "sum"),
)

summary["difference"] = (
    summary["cost_aware_net"] - summary["old_net"]
)

print(summary.round(2).to_string())

OUTPUT = (
    ROOT / "results" / "cost_aware_backtest_2019.csv"
)

out.to_csv(OUTPUT, index=False)

print("\nSaved:", OUTPUT)
print("Cost-aware MILP smoke test completed.")
