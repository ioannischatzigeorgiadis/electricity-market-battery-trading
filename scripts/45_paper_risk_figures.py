
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_figure(fig, filename):
    for ext in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"{filename}.{ext}",
            dpi=300,
            bbox_inches="tight"
        )
    plt.close(fig)


# ==========================================
# FIGURE 3: STOCHASTIC ECONOMIC PERFORMANCE
# ==========================================

strategies = {
    "Risk-neutral": "risk_neutral_profit",
    "CVaR 0.25": "cvar_025_profit",
    "CVaR 1.00": "cvar_100_profit",
}

advantages = {}

for year, expected_days in [(2019, 356), (2020, 357)]:

    file = (
        RESULTS
        / f"stochastic_backtest_{year}_with_throughput.csv"
    )

    df = pd.read_csv(file)

    required = [
        "date",
        "deterministic_profit",
        *strategies.values()
    ]

    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing columns in {file}")

    if len(df) != expected_days:
        raise ValueError(
            f"Unexpected number of days in {year}"
        )

    if df["date"].duplicated().any():
        raise ValueError(f"Duplicate dates: {year}")

    if not np.isfinite(
        df[required[1:]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Invalid profits: {year}")

    advantages[year] = {}

    for name, column in strategies.items():
        delta = (
            df[column] - df["deterministic_profit"]
        )

        advantages[year][name] = float(delta.sum())

    print(f"\n--- {year} STOCHASTIC ADVANTAGES ---")
    for name, value in advantages[year].items():
        print(f"{name}: {value:+.2f} EUR")


fig, ax = plt.subplots(figsize=(3.45, 2.55))

x = np.arange(len(strategies))
width = 0.35

for year, offset, hatch in [
    (2019, -width / 2, ""),
    (2020, width / 2, "//"),
]:
    values = [
        advantages[year][name]
        for name in strategies
    ]

    bars = ax.bar(
        x + offset,
        values,
        width,
        label=str(year),
        hatch=hatch
    )

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:+.0f}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                value
            ),
            xytext=(0, 3 if value >= 0 else -3),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=6.5
        )

ax.axhline(0, linestyle=":", linewidth=0.9)

ax.set_xticks(x)
ax.set_xticklabels([
    "Risk-neutral",
    r"CVaR $\lambda=0.25$",
    r"CVaR $\lambda=1.0$",
])

ax.set_ylabel("Gross-profit difference (EUR)")
ax.legend(frameon=False, loc="lower left")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

all_values = [
    v
    for yearly in advantages.values()
    for v in yearly.values()
]

ax.set_ylim(
    min(all_values) * 1.17,
    max(250, max(all_values) * 1.6)
)

fig.tight_layout(pad=0.5)

save_figure(
    fig,
    "fig3_stochastic_profit_difference"
)


# ==========================================
# FIGURE 4: UNCERTAINTY CALIBRATION
# ==========================================

cal = pd.read_csv(
    RESULTS / "ml_error_scenarios_2015.csv"
)

hour_columns = [str(h) for h in range(24)]

cal_errors = cal[hour_columns].to_numpy(
    dtype=float
)

if cal_errors.shape != (365, 24):
    raise ValueError("Invalid 2015 calibration data")

cal_std = cal_errors.std(ddof=1)

diagnostics = pd.read_csv(
    RESULTS / "uncertainty_diagnostics_2019_2020.csv"
)

if set(diagnostics["year"].astype(int)) != {2019, 2020}:
    raise ValueError("Missing uncertainty diagnostics")

if diagnostics["year"].duplicated().any():
    raise ValueError("Duplicate diagnostic years")

diagnostics = diagnostics.set_index("year")

std_values = [
    cal_std,
    diagnostics.loc[2019, "error_std"],
    diagnostics.loc[2020, "error_std"],
]

coverage_values = [
    diagnostics.loc[2019, "coverage_90"],
    diagnostics.loc[2020, "coverage_90"],
]

if not np.isfinite(
    std_values + coverage_values
).all():
    raise ValueError("Invalid uncertainty diagnostics")


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(3.45, 4.0),
    constrained_layout=True
)

# Panel A: Error standard deviation
years = ["2015", "2019", "2020"]

bars = ax1.bar(
    years,
    std_values,
    width=0.55
)

for bar, value in zip(bars, std_values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value:.2f}",
        ha="center",
        fontsize=7
    )

ax1.set_ylabel("Error std. dev. (EUR/MWh)")
ax1.set_ylim(0, max(std_values) * 1.25)
ax1.set_title(
    "(a) Forecast-error variability",
    fontsize=8
)
ax1.grid(axis="y", alpha=0.2)
ax1.set_axisbelow(True)

# Panel B: Out-of-sample interval coverage
x2 = np.arange(2)

ax2.plot(
    x2,
    coverage_values,
    marker="o",
    linewidth=1.5,
    label="Observed coverage"
)

ax2.axhline(
    90,
    linestyle="--",
    linewidth=1.1,
    label="Nominal 90%"
)

for pos, value in zip(x2, coverage_values):
    ax2.annotate(
        f"{value:.2f}%",
        (pos, value),
        xytext=(6, -12),
        textcoords="offset points",
        fontsize=7
    )

ax2.set_xticks(x2)
ax2.set_xticklabels(["2019", "2020"])
ax2.set_xlim(-0.35, 1.35)
ax2.set_ylim(75, 95)

ax2.set_ylabel("Interval coverage (%)")
ax2.set_title(
    "(b) Empirical 90% interval coverage",
    fontsize=8
)
ax2.legend(frameon=False, fontsize=6.5)
ax2.grid(axis="y", alpha=0.2)

save_figure(
    fig,
    "fig4_uncertainty_calibration"
)

print("\n--- UNCERTAINTY CALIBRATION ---")

for year, std in zip(years, std_values):
    print(f"{year} error std: {std:.3f} EUR/MWh")

for year, value in zip([2019, 2020], coverage_values):
    print(f"{year} coverage: {value:.2f}%")

print("\nAll figures saved to:")
print(FIGURES)
