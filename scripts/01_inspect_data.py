"""Day-1 data audit and plots. Reads actual market prices, never synthesizes data."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "DE.csv"
OUT = ROOT / "results"


def load_prices(path: Path = DATA) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python scripts/00_download_data.py first.")
    df = pd.read_csv(path, index_col=0)
    if "Price" not in df.columns:
        raise ValueError(f"Price column not found. Available columns: {list(df.columns)}")
    dates = pd.to_datetime(df.index, errors="raise")
    if dates.hasnans:
        raise ValueError("Unparseable missing timestamps")
    prices = pd.to_numeric(df["Price"], errors="coerce")
    prices.index = pd.DatetimeIndex(dates, name="delivery_time")
    return prices.rename("price_EUR_per_MWh")


def build_report(prices: pd.Series) -> str:
    idx = prices.index
    duplicate_count = int(idx.duplicated().sum())
    differences = idx.to_series().diff().dropna()
    unusual_steps = int((differences != pd.Timedelta(hours=1)).sum())
    start, end = idx.min(), idx.max()
    expected_hourly = int((end - start) / pd.Timedelta(hours=1)) + 1
    return "\n".join([
        "EPF Toolbox DE.csv — data quality overview",
        f"First delivery timestamp: {start}",
        f"Last delivery timestamp:  {end}",
        f"Rows:                     {len(prices):,}",
        f"Expected hourly rows:     {expected_hourly:,}",
        f"Missing price values:     {int(prices.isna().sum()):,}",
        f"Duplicate timestamps:     {duplicate_count:,}",
        f"Non-1-hour index steps:   {unusual_steps:,}",
        f"Negative price values:    {int((prices < 0).sum()):,}",
        f"Zero price values:        {int((prices == 0).sum()):,}",
        f"Min EUR/MWh:              {prices.min():.3f}",
        f"Mean EUR/MWh:             {prices.mean():.3f}",
        f"Median EUR/MWh:           {prices.median():.3f}",
        f"Max EUR/MWh:              {prices.max():.3f}",
        "",
        "Interpret timestamps and potential DST/24-hour-day normalization before modeling.",
        "Price units EUR/MWh presumed for EPEX-DE; confirm in source metadata.",
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_prices()
    report = build_report(series)
    print(report)
    (OUT / "day1_data_audit.txt").write_text(report + "\n", encoding="utf-8")
    if series.isna().all():
        raise ValueError("All price values missing")
    fig, ax = plt.subplots(figsize=(11, 4))
    series.resample("7D").mean().plot(ax=ax)
    ax.set(title="Weekly mean day-ahead electricity price — historical Germany", xlabel="Delivery date", ylabel="EUR/MWh")
    fig.tight_layout()
    fig.savefig(OUT / "weekly_prices.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    series.groupby(series.index.hour).mean().plot(ax=ax, marker="o")
    ax.set(title="Average price by hour of day", xlabel="Hour", ylabel="EUR/MWh")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(OUT / "hourly_profile.png", dpi=160)
    plt.close(fig)
    print(f"\nSaved report and figures to: {OUT}")


if __name__ == "__main__":
    main()
