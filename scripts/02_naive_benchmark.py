"""First out-of-sample *forecasting* benchmark (no battery optimization yet)."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
load_prices = import_module("01_inspect_data").load_prices


def naive_benchmark(prices: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate delivery timestamps: resolve before building forecasts")
    # Reindex to continuous hourly timestamps so shift(24) *means* the previous day.
    # Fail instead of silently filling gaps in the dataset.
    if len(prices) > 1 and (prices.index.to_series().diff().iloc[1:] != pd.Timedelta(hours=1)).any():
        raise ValueError("Non-hourly index steps: resolve time indexing/DST before backtesting")
    if prices.isna().any():
        raise ValueError("Missing price values: resolve before benchmarking")
    df = pd.DataFrame({"actual": prices})
    df["naive_day_lag"] = prices.shift(24)
    df["naive_week_lag"] = prices.shift(24 * 7)
    test = df[
    (df.index.year >= 2016) &
    (df.index.year <= 2017)
].dropna().copy()
    if test.empty:
        raise ValueError(f"No complete observations for requested test year {test_year}")
    metrics = []
    for name in ("naive_day_lag", "naive_week_lag"):
        err = test[name] - test["actual"]
        metrics.append({"model": name, "n_hours": len(test),
                        "MAE_EUR_MWh": float(np.mean(np.abs(err))),
                        "RMSE_EUR_MWh": float(np.sqrt(np.mean(err ** 2)))})
    return test, pd.DataFrame(metrics)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    test, metrics = naive_benchmark(prices)
    print("Historical forecast-only benchmark; NO battery profit has been estimated.")
    print(metrics.to_string(index=False))
    test.to_csv(
    OUT / "naive_test_predictions_2016_2017.csv",
    index_label="delivery_time"
)
    metrics.to_csv(
    OUT / "naive_metrics_2016_2017.csv",
    index=False
)
    print(f"Wrote out-of-sample forecasts and metrics to {OUT}")


if __name__ == "__main__":
    main()
