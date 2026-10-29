# Machine Learning and Optimization for Electricity Market Trading

### Uncertainty-Aware Battery Trading Strategies

A Python-based research project investigating the integration of **electricity price forecasting, battery arbitrage optimization, operating-cost modeling, and risk-aware decision-making** in the German and day-ahead electricity market.

The project evaluates how machine learning forecasts translate into realized trading performance and examines the economic implications of battery operating costs and electricity price uncertainty.

**[Read the full research paper](paper/Paper.pdf)**

## Research Overview

Electricity price forecasting is an important component of algorithmic trading in wholesale electricity markets. However, improved forecast accuracy does not necessarily translate into improved economic performance.

This project investigates three interconnected questions:

1. Can machine learning-based electricity price forecasts improve battery arbitrage profitability compared with simple trading benchmarks?
2. How does explicitly accounting for battery degradation and transaction costs affect optimal trading decisions?
3. Can stochastic optimization and Conditional Value-at-Risk (CVaR) improve trading outcomes under electricity price uncertainty?

The analysis combines historical market data, machine learning, mixed-integer linear programming (MILP), and out-of-sample economic backtesting.

## Methodology

### 1. Electricity Price Forecasting

A `HistGradientBoostingRegressor` model is trained using historical German electricity market data through 2015.

The model uses historical electricity prices and calendar-based features to predict hourly day-ahead electricity prices.

Forecasting performance is evaluated against a naive persistence benchmark using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The model is evaluated on 2019 and 2020 data without retraining.

### 2. Battery Arbitrage Optimization

Forecasted electricity prices are used as inputs to a mixed-integer linear programming model that determines hourly battery charging and discharging schedules.

The battery model assumes:

| Parameter | Value |
|---|---|
| Power capacity | 1 MW |
| Energy capacity | 2 MWh |
| Round-trip efficiency | 90% |
| Optimization horizon | 24 hours |
| Market | German day-ahead electricity market |

The optimization incorporates battery power limits, state-of-charge constraints, and charging/discharging decisions.

### 3. Cost-Aware Optimization

An extended MILP formulation incorporates illustrative battery degradation and transaction costs into the scheduling objective.

The analysis compares cost-aware schedules with schedules optimized solely for gross trading profit, while evaluating both under identical operating-cost assumptions.

### 4. Stochastic Optimization and Risk Management

Electricity price uncertainty is represented through scenarios constructed using historical forecasting errors.

Two approaches are evaluated:

- Risk-neutral stochastic optimization, maximizing expected trading profit.
- CVaR-based optimization, incorporating downside-risk penalization.

The analysis investigates the trade-off between realized trading profitability, risk aversion, and the calibration of forecast uncertainty.

## Key Results

### Forecasting Performance

| Year | ML MAE (€/MWh) | Naive MAE (€/MWh) | MAE improvement |
|---|---:|---:|---:|
| 2019 | 7.205 | 9.128 | 21.07% |
| 2020 | 7.756 | 10.006 | 22.49% |

The machine learning model achieves lower forecasting errors than the naive benchmark in both evaluation periods.

### Realized Gross Trading Profit

| Strategy | 2019 (€) | 2020 (€) |
|---|---:|---:|
| Fixed schedule | 15,001.46 | 17,038.66 |
| Naive forecast | 14,910.37 | 16,645.69 |
| Deterministic ML | **16,926.46** | **18,859.17** |
| Perfect foresight | 20,223.64 | 22,596.10 |

Relative to the fixed-schedule benchmark, ML-based deterministic optimization improves realized gross trading profit by **12.83% in 2019** and **10.69% in 2020**.

The perfect-foresight strategy represents an ex-post benchmark rather than an implementable trading strategy.

### Impact of Operating Costs

Under the illustrative high-cost configuration, cost-aware optimization improves realized net operating profit by:

- **€2,674.50 in 2019**
- **€2,747.49 in 2020**

These results demonstrate the importance of incorporating operating costs directly into battery scheduling decisions. They do not represent a complete battery lifecycle profitability assessment.

### Stochastic Optimization and CVaR

Risk-neutral stochastic optimization achieves modest increases in realized gross profit relative to deterministic ML scheduling.

However, stronger CVaR penalization reduces realized gross profit without improving the empirical worst-5% mean daily profit in the evaluated periods.

Furthermore, nominal 90% forecast uncertainty intervals achieve realized coverage of 84.80% in 2019 and 82.85% in 2020, highlighting limitations in transferring historical forecast-error distributions to later market conditions.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
├── scripts/       # Python research and experimental scripts
├── data/raw/      # Directory for original market datasets
├── results/       # Forecasting and optimization results
├── figures/       # Figures used in the research paper
└── paper/
    └── paper.pdf  # Full IEEE-format research paper
```

## Technologies

- **Python** — data processing, forecasting, optimization, and evaluation
- **pandas / NumPy** — time-series processing and numerical analysis
- **scikit-learn** — histogram-based gradient boosting
- **SciPy / HiGHS** — mixed-integer linear programming
- **Matplotlib** — scientific visualization
- **Bootstrap resampling** — statistical evaluation of trading performance

## Data and Reproducibility

The study uses historical German electricity prices, including data from the 2012–2017 period and Germany/Luxembourg day-ahead market observations for 2019 and 2020.

- [SMARD — German Electricity Market Data](https://www.smard.de/en)
- [Zenodo — Historical Electricity Price Dataset](https://doi.org/10.5281/zenodo.4624805)

The original datasets are not redistributed in this repository.
Users should obtain the data from the respective sources
and follow their applicable licensing conditions.

Python dependencies are documented in `requirements.txt`, while `requirements-lock.txt` records a specific dependency environment.

The repository contains the research scripts and selected experimental outputs. A fully automated end-to-end reproduction workflow has not yet been established. Independent reruns produced small differences in ML forecasts and resulting optimization profits, potentially associated with software-environment differences; the precise cause has not been confirmed.

Reported numerical results correspond to the research paper.

## Research Paper

**Machine Learning and Optimization for Electricity Market Trading: Uncertainty-Aware Battery Trading Strategies**

Ioannis O. Chatzigeorgiadis  
Diploma in Electrical and Computer Engineering  
National Technical University of Athens, Greece

[View full paper (PDF)](paper/Paper.pdf)

## Limitations and Future Work

The analysis is based on historical backtesting under explicit battery and market assumptions. It does not incorporate all real-world trading costs, market participation requirements, or battery lifecycle effects.

Potential extensions include:

- Improved probabilistic electricity price forecasting and scenario calibration
- Rolling-horizon optimization and more realistic battery operating constraints
- Market-regime adaptation and periodic model retraining
- Alternative risk measures and distributionally robust optimization
- Joint participation in multiple electricity markets

## Author

**Ioannis O. Chatzigeorgiadis**  
Diploma in Electrical and Computer Engineering  
National Technical University of Athens, Greece

*Independent research project in electricity markets, machine learning, and optimization.*
