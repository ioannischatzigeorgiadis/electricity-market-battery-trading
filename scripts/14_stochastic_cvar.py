
from pathlib import Path
from importlib import import_module
import sys
import numpy as np
import pandas as pd

from scipy.optimize import milp, Bounds, LinearConstraint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

sys.path.insert(0, str(ROOT / "scripts"))
battery = import_module("10_battery_optimization")

ALPHA = 0.95
N_SCENARIOS = 100


def optimize_stochastic(scenarios, risk_weight):

    scenarios = np.asarray(scenarios, dtype=float)
    S, T = scenarios.shape

    if risk_weight == 0:
        return battery.optimize_battery(
            scenarios.mean(axis=0)
        )

    # Decision variable indices
    ch = slice(0, T)
    dis = slice(T, 2*T)
    soc = slice(2*T, 3*T+1)
    mode = slice(3*T+1, 4*T+1)

    # CVaR auxiliary variables
    z = 4*T + 1
    u = slice(z+1, z+1+S)
    n = u.stop

    # Minimize negative expected profit + CVaR penalty
    c = np.zeros(n)
    mean_prices = scenarios.mean(axis=0)

    c[ch] = mean_prices * battery.DT
    c[dis] = -mean_prices * battery.DT

    c[z] = risk_weight
    c[u] = risk_weight / ((1-ALPHA)*S)

    lower = np.zeros(n)
    upper = np.full(n, np.inf)

    upper[ch] = battery.P_MAX
    upper[dis] = battery.P_MAX
    upper[soc] = battery.E_MAX
    upper[mode] = 1

    lower[z] = -np.inf

    # Initial and final SOC = 0
    lower[soc.start] = upper[soc.start] = 0
    lower[soc.stop-1] = upper[soc.stop-1] = 0

    integrality = np.zeros(n, dtype=int)
    integrality[mode] = 1

    # Energy balance constraints
    A_eq = np.zeros((T, n))

    # Operating mode constraints
    A_mode = np.zeros((2*T, n))

    for t in range(T):

        A_eq[t, soc.start+t+1] = 1
        A_eq[t, soc.start+t] = -1
        A_eq[t, ch.start+t] = -battery.ETA*battery.DT
        A_eq[t, dis.start+t] = battery.DT/battery.ETA

        A_mode[t, ch.start+t] = 1
        A_mode[t, mode.start+t] = battery.P_MAX

        A_mode[T+t, dis.start+t] = 1
        A_mode[T+t, mode.start+t] = -battery.P_MAX

    mode_upper = np.concatenate([
        np.full(T, battery.P_MAX),
        np.zeros(T)
    ])

    # CVaR constraints:
    # u_s >= loss_s - z
    A_risk = np.zeros((S, n))

    A_risk[:, ch] = scenarios * battery.DT
    A_risk[:, dis] = -scenarios * battery.DT
    A_risk[:, z] = -1

    A_risk[
        np.arange(S),
        u.start + np.arange(S)
    ] = -1

    constraints = [
        LinearConstraint(A_eq, 0, 0),
        LinearConstraint(
            A_mode, -np.inf, mode_upper
        ),
        LinearConstraint(
            A_risk, -np.inf, 0
        )
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

    return result.x[ch], result.x[dis], result.x[soc]

if __name__ == "__main__":
    # Load ML predictions
    df = pd.read_csv(
        OUT / "ml_predictions_2016_2017.csv",
        index_col=0,
        parse_dates=True
    )

    day = df.loc["2016-01-01"].sort_index()

    actual = day["actual"].to_numpy()
    forecast = day["ml_forecast"].to_numpy()

    assert len(actual) == 24

    # Load historical daily residual trajectories
    errors = pd.read_csv(
        OUT / "ml_error_scenarios_2015.csv",
        index_col=0
    )

    assert errors.shape == (365, 24)

    # Sample whole daily error trajectories
    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(errors),
        size=N_SCENARIOS,
        replace=False
    )

    sampled_errors = errors.iloc[indices].to_numpy()

    # Create possible price scenarios
    scenarios = forecast[None, :] + sampled_errors

    assert scenarios.shape == (N_SCENARIOS, 24)
    assert np.isfinite(scenarios).all()


    def evaluate(name, charge, discharge, soc):

        residual = (
            soc[1:] - soc[:-1]
            - battery.ETA * charge * battery.DT
            + discharge * battery.DT / battery.ETA
        )

        assert np.max(np.abs(residual)) < 1e-5
        assert np.max(np.minimum(charge, discharge)) < 1e-5
        assert soc.min() >= -1e-5
        assert soc.max() <= battery.E_MAX + 1e-5
        assert abs(soc[0]) < 1e-5
        assert abs(soc[-1]) < 1e-5

        net_power = discharge - charge

        realized = float(
            actual @ net_power * battery.DT
        )

        scenario_profits = (
            scenarios @ net_power * battery.DT
        )

        tail_count = int(
            round((1-ALPHA) * N_SCENARIOS)
        )

        worst_tail = np.sort(scenario_profits)[:tail_count]

        print(f"\n--- {name} ---")
        print(f"Realized profit: {realized:.2f} EUR")
        print(
            f"Expected scenario profit: "
            f"{scenario_profits.mean():.2f} EUR"
        )
        print(
            f"Average worst 5% scenario profit: "
            f"{worst_tail.mean():.2f} EUR"
        )

        print("Physical constraints: OK")


    print("\nSTOCHASTIC MILP — 2016-01-01")
    print("Price scenarios:", N_SCENARIOS)

    # Original point forecast
    ch, dis, soc = battery.optimize_battery(forecast)
    evaluate("Deterministic ML", ch, dis, soc)

    # Risk-neutral and risk-aware optimization
    for weight in [0.0, 0.25, 1.0]:

        ch, dis, soc = optimize_stochastic(
            scenarios, risk_weight=weight
        )

        evaluate(
            f"Stochastic lambda={weight}",
            ch, dis, soc
        )
