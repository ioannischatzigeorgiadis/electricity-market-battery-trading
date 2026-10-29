
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper_figures"

FIGURES.mkdir(parents=True, exist_ok=True)

# IEEE single-column figure formatting
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


def load_csv(path, required):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing columns {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    )

    return df.sort_values("date")


def plot_cumulative(series_by_year, filename):
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    line_styles = {
        2019: "-",
        2020: "--",
    }

    for year, (dates, gains) in series_by_year.items():

        # Align both years by calendar month/day.
        # Year 2000 is used only for the plotting axis.
        plot_dates = pd.to_datetime(
            "2000-" + dates.dt.strftime("%m-%d")
        )

        cumulative = np.cumsum(gains)

        ax.plot(
            plot_dates,
            cumulative,
            linestyle=line_styles[year],
            linewidth=1.6,
            label=str(year),
        )

        print(
            f"{filename} | {year}: "
            f"{cumulative[-1]:+.2f} EUR"
        )

    ax.axhline(
        0,
        linestyle=":",
        linewidth=0.8,
        color="gray",
    )

    ax.set_xlim(
        pd.Timestamp("2000-01-01"),
        pd.Timestamp("2000-12-31"),
    )

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(
            bymonth=[1, 3, 5, 7, 9, 11]
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b")
    )

    ax.set_ylabel("Cumulative gain (EUR)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout(pad=0.4)

    for extension in ["pdf", "png"]:
        path = FIGURES / f"{filename}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


# ==========================================
# FIGURE 1: ML VS FIXED SCHEDULE
# ==========================================

ml_series = {}

for year in [2019, 2020]:

    file = RESULTS / f"milp_backtest_{year}.csv"

    df = load_csv(
        file,
        ["date", "ml_profit", "fixed_profit"],
    )

    gain = (
        df["ml_profit"] - df["fixed_profit"]
    ).to_numpy(dtype=float)

    if not np.isfinite(gain).all():
        raise ValueError(f"Invalid ML gains: {year}")

    ml_series[year] = (df["date"], gain)

plot_cumulative(
    ml_series,
    "fig1_cumulative_ml_advantage",
)


# ==========================================
# FIGURE 2: COST-AWARE VS COST-BLIND
# HIGH-COST SCENARIO
# ==========================================

cost_series = {}

for year in [2019, 2020]:

    file = RESULTS / f"cost_aware_backtest_{year}.csv"

    df = load_csv(
        file,
        [
            "date",
            "scenario",
            "old_net",
            "cost_aware_net",
        ],
    )

    # Compare schedules under identical costs.
    df = df[df["scenario"] == "high"].copy()

    if df.empty:
        raise ValueError(
            f"No high-cost results for {year}"
        )

    if df["date"].duplicated().any():
        raise ValueError(
            f"Duplicate high-cost dates: {year}"
        )

    gain = (
        df["cost_aware_net"] - df["old_net"]
    ).to_numpy(dtype=float)

    if not np.isfinite(gain).all():
        raise ValueError(
            f"Invalid cost-aware gains: {year}"
        )

    cost_series[year] = (df["date"], gain)

plot_cumulative(
    cost_series,
    "fig2_cumulative_cost_advantage",
)

print("\nFigures generated successfully.")
print("Saved in:", FIGURES)
