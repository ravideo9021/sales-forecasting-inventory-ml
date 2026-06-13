# Sales Forecasting & Inventory Optimization ML System

> A production-grade machine learning pipeline that forecasts retail sales and translates those forecasts directly into actionable inventory decisions — all surfaced through a live interactive dashboard.

**🚀 Live demo → [sales-forecasting-inventory.streamlit.app](https://sales-forecasting-inventory.streamlit.app/)**

The hosted demo runs on synthetic data generated at startup (the 1.7 GB training dataset is too large for free hosting), so every page is fully interactive without any setup.

---

## Table of Contents

- [What This System Does](#what-this-system-does)
- [Who It Is For](#who-it-is-for)
- [Why It Matters](#why-it-matters)
- [Architecture & How It Works](#architecture--how-it-works)
- [Feature Engineering Deep Dive](#feature-engineering-deep-dive)
- [The Model: XGBoost for Time Series](#the-model-xgboost-for-time-series)
- [Inventory Optimization Engine](#inventory-optimization-engine)
- [Dashboard (Streamlit)](#dashboard-streamlit)
- [Performance Numbers](#performance-numbers)
- [Technology Stack](#technology-stack)
- [How to Run — New Machine Setup](#how-to-run--new-machine-setup)
- [Project Structure](#project-structure)
- [Comparable Tools — And How This Differs](#comparable-tools--and-how-this-differs)
- [Tests](#tests)

---

## What This System Does

This system solves two connected retail operations problems in one end-to-end pipeline:

**Problem 1 — Demand Forecasting**
> "How much of each product will each store sell over the next 30 days?"

**Problem 2 — Inventory Optimization**
> "Given those forecasts, how much stock should we carry, when should we reorder, and how much?"

Most tools solve only one of these. This system chains them: the ML forecast feeds directly into the inventory optimizer, so reorder decisions are grounded in predicted demand rather than historical averages or gut feel.

---

## Who It Is For

| Role | What They Get |
|------|--------------|
| **Supply Chain Manager** | Reorder recommendations, safety stock levels, ABC classification of items |
| **Category Manager** | Sales forecasts by product family and store with trend/seasonality breakdown |
| **Operations Analyst** | Model diagnostics, residual analysis, cross-validation results |
| **Executive / Decision Maker** | Executive summary dashboard — KPIs, inventory risk, and cost impact at a glance |
| **Data Scientist / ML Engineer** | Full reproducible pipeline, configurable features and hyperparameters, 25 unit tests |

The system works out-of-the-box with the [Kaggle Store Sales dataset](https://www.kaggle.com/c/store-sales-time-series-forecasting) or generates realistic synthetic data automatically if no real data is present — **no manual data download required to run it**.

---

## Why It Matters

**The core business problem:**
Retail inventory is a cash-flow killer at both extremes. Too much stock ties up working capital and increases storage costs. Too little causes stockouts, lost sales, and customer churn. The only way to thread this needle at scale is with accurate forward-looking demand signals.

**The gap this fills:**
Traditional approaches — spreadsheet-based moving averages, simple exponential smoothing, or rule-of-thumb reorder points — cannot capture the complex interaction of seasonality, promotions, oil prices, and holidays that drives real retail demand. This system encodes all of those signals as model features and learns their combined effect.

**Projected business impact (based on modeled outputs):**

```
Inventory reduction potential:    ~22%
Estimated annual cost savings:    ~$2.4M (for a 10-store operation)
Service level maintained:         96.8%
Working capital improvement:      ~$1.8M
Inventory turns increase:         ~15%
Storage cost reduction:           ~12%
```

---

## Architecture & How It Works

The pipeline has six sequential phases, each independently runnable:

```
┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
│  1. Load    │───▶│  2. Clean   │───▶│  3. Feature Eng  │
│   Data      │    │   Data      │    │                  │
└─────────────┘    └─────────────┘    └────────┬─────────┘
                                               │
┌─────────────┐    ┌─────────────┐    ┌────────▼─────────┐
│ 6. Optimize │◀───│ 5. Forecast │◀───│  4. Train Model  │
│  Inventory  │    │             │    │                  │
└─────────────┘    └─────────────┘    └──────────────────┘
```

### Phase 1 — Data Loading
Loads six datasets: training sales, test set, stores metadata, oil prices, holidays/events, and transaction counts. If no real data is present, the system automatically generates 3 years of synthetic retail data with:
- Stable per-store/family base demand (autocorrelated AR(1) process)
- Seasonal factors (sin wave over day-of-year)
- Weekend uplifts (+20%)
- Promotional spikes (+50%, ~10% of days)
- Gaussian noise (σ = 8% of base)

### Phase 2 — Data Cleaning
- Removes duplicate rows and enforces unique `(date, store, family)` keys
- Drops rows with null target values
- Caps outliers using IQR-based detection
- Validates and coerces data types across all six datasets

### Phase 3 — Feature Engineering
See [Feature Engineering Deep Dive](#feature-engineering-deep-dive) below.

### Phase 4 — Model Training
Chronological 80/20 split (never random — time series data must not leak future into past). Trains XGBoost with early stopping on the validation fold. See [The Model](#the-model-xgboost-for-time-series).

### Phase 5 — Forecast Generation
Prepends 455 days of training history before engineering features for test data. This is critical: lag features up to 365 days and rolling windows up to 90 days require real historical sales, not dummy zeros.

```python
history_needed = max_lag (365) + max_window (90) = 455 days
```

### Phase 6 — Inventory Optimization
Uses the 30-day forecast as input to EOQ, safety stock, and reorder point calculations. Runs ABC/XYZ analysis and generates prioritized reorder recommendations.

---

## Feature Engineering Deep Dive

The feature set is where most of the predictive power lives. 70+ features are created across five categories:

### Temporal Features (14 features)
| Feature | Description |
|---------|-------------|
| `year`, `month`, `day` | Calendar decomposition |
| `day_of_week` | 0–6, captures weekday/weekend patterns |
| `day_of_year`, `week_of_year`, `quarter` | Annual cycle positioning |
| `is_weekend`, `is_month_start/end` | Binary flags |
| `month_sin`, `month_cos` | Cyclical encoding — avoids treating Dec→Jan as a large jump |
| `day_sin`, `day_cos` | Cyclical encoding of day-of-week |

### Lag Features (6 lags × per store/family group)
Lag periods: **1, 7, 14, 30, 91, 365 days**

These are the model's single most important feature group. `sales_lag_7` asks: "what did this store sell of this product exactly one week ago?" — a very strong predictor for weekly-cycling retail demand.

> **Critical implementation detail:** Lag features for the test set are computed by prepending 455 days of real training history before feature engineering. Without this, all lag features for test dates would be zero — the model would be flying blind.

### Rolling Statistics (3 windows × 6 stats)
Windows: **7, 30, 90 days**. For each window: mean, std, min, max, and coefficient of variation.

- `sales_ma_7` → short-term trend
- `sales_ma_30` → medium-term trend
- `sales_std_90` → demand variability (directly feeds safety stock calculation)

### Holiday & External Features
| Feature | Source | Notes |
|---------|--------|-------|
| `is_holiday`, `is_transferred` | Holidays dataset | Binary flags per date |
| `days_to_holiday` | Computed vectorially | Proximity effect — demand rises before major holidays |
| `days_from_holiday` | Computed vectorially | Post-holiday demand decay |
| `dcoilwtico` (oil price) | Oil dataset | Ecuador economy proxy; affects consumer spending |
| `oil_ma_7`, `oil_ma_30` | Derived | Smoothed oil price signal |
| `oil_change`, `oil_volatility` | Derived | Rate of change and 30-day std |

### Promotional Features
- `onpromotion` (raw flag)
- `promo_lag_1`, `promo_lag_7` — was there a promo yesterday / last week?
- `promo_rolling_7` — promotion density over past week

---

## The Model: XGBoost for Time Series

### Why XGBoost (and not LSTM, Prophet, or ARIMA)?

| Model | Pros | Cons | Verdict |
|-------|------|------|---------|
| **XGBoost** | Handles tabular features natively, captures non-linear interactions, fast training, interpretable feature importance | Not inherently sequential | **Best fit** for this feature-rich tabular setup |
| LSTM / RNN | Native sequence learning | Needs large data, slow, hard to include external features | Overkill for daily retail with good lag features |
| Prophet | Great for simple seasonal decomposition | Struggles with multi-store multi-family; no external features | Too limited |
| ARIMA/SARIMA | Classic, interpretable | Univariate only; requires separate model per series; can't share signal across stores | Would require 50 separate models |

### Hyperparameters (config/config.yaml)
```yaml
n_estimators: 1000
max_depth: 6
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
```

`subsample: 0.8` and `colsample_bytree: 0.8` reduce overfitting by subsampling rows and columns at each tree split — similar to how Random Forest works.

### Training Protocol
- Chronological split: first 80% of dates → train, last 20% → validation
- No random shuffling (would leak future data into training)
- No hyperparameter tuning by default (set `tune_hyperparameters=True` to enable Optuna search)
- Model persisted via `joblib` to `models/xgboost_model.joblib`

### Objective & Probabilistic Forecasting
- **Tweedie objective** (`reg:tweedie`, `variance_power=1.2`) is the default instead of plain L2/RMSE. Retail sales are right-skewed and zero-inflated; Tweedie models that distribution directly (the same choice the M5-competition winners made). Override via `config['models']['xgboost']['objective']`.
- **Quantile regression** trains P10 / P50 / P90 models with XGBoost's native pinball loss (`reg:quantileerror`). This produces honest, *asymmetric* prediction intervals — surfaced as the P10–P90 band on the Forecasting page — replacing the old heuristic ±10% band. Access via `model.predict_quantiles(X)`; the pipeline writes `forecast_lo` / `forecast_hi` columns to `forecasts_xgboost.csv`.

### Ensemble, Backtesting & Foundation-Model Baseline
Upgrades that move the pipeline from "solid GBM" toward competition-grade:
- **LightGBM second learner + averaging ensemble** (`src/models/lightgbm_model.py`, `src/models/ensemble.py`). LightGBM (same Tweedie + quantile setup) is trained alongside XGBoost; the `EnsembleForecaster` averages their point forecasts and quantiles. Ensembling diverse-but-comparable GBMs is the cheapest reliable accuracy gain on tabular retail data. `generate_forecasts()` uses the ensemble automatically when available.
- **Rolling-origin backtest** (`src/evaluation/backtest.py`). Instead of a single 80/20 split, the approach is re-trained at several chronological cutoffs and scored on the next *N* days at each — an honest mean ± variance error estimate. Written to `reports/backtest_xgboost.csv` during training.
- **Foundation-model baseline (Amazon Chronos)** (`src/models/foundation_baseline.py`) — *optional*, import-guarded, zero-shot probabilistic forecasting as a "different method" benchmark/ensemble member. Heavy deps (`torch`, `chronos-forecasting`) are **not** installed by default; enable with `pip install chronos-forecasting torch`.

---

## Inventory Optimization Engine

### Economic Order Quantity (EOQ)

```
EOQ = √(2 × Annual Demand × Ordering Cost / Holding Cost per Unit)
```

Configured with: `ordering_cost` = $50, `holding_cost_rate` = 25% of unit cost/year.

### Safety Stock

```
Safety Stock = z(service_level) × σ_demand × √(lead_time + review_period)
```

At 95% service level: `z = 1.645` · At 99%: `z = 2.326`

σ_demand comes from the rolling 90-day sales std computed during feature engineering — tying optimization directly to the ML pipeline.

### Reorder Point

```
Reorder Point = (Daily Demand Forecast × Lead Time) + Safety Stock
```

Uses the 30-day ML forecast as demand input — not a historical average. Reorder points automatically anticipate upcoming promotions or seasonal peaks.

### ABC Analysis

Items are ranked by annual value (forecast volume × unit cost):
- **Class A**: Top 80% of value — highest attention, tightest safety stock
- **Class B**: Next 15% of value — standard review cycle
- **Class C**: Bottom 5% of value — simplified ordering rules

---

## Dashboard (RetailIQ Platform)

Six pages accessible from the sidebar. All pages share live sidebar filters (date range, store multi-select, product family multi-select). Filter options are derived from the loaded dataset at runtime — date bounds span every date-bearing table (so forecast dates beyond the last sales day are never clipped), and the store/family lists reflect whatever the data actually contains. This means the dashboard works unchanged whether it loads the real Favorita CSVs (2013–2017, integer store ids, 33 families) or the synthetic demo fallback.

Design: restrained single-accent (indigo `#6366f1`) dark theme (`#08090d` background) inspired by Linear / Vercel / LangChain. Inter + JetBrains Mono typography, stroke-based Lucide-style inline SVG icons, left-bordered alert rows, sparkline KPI cards, and an indigo-anchored chart palette (no rainbow). Interactive Plotly charts are themed to match.

Data loading is column-pruned: the sales feature table (`features_engineered.csv`, ~1.7 GB / 79 columns) is read with only the four columns the dashboard uses (`date`, `store_nbr`, `family`, `sales`) via the pyarrow CSV engine, and cached for the life of the session (the source CSVs are static, so there is no TTL — a TTL would force a fresh multi-second read on the first click after any idle and freeze the single Streamlit script thread). This keeps first paint to a few seconds instead of stalling ~40s+ on a full-table parse every rerun. Large exports (e.g. the full sales table) are generated only when explicitly requested via a button, never eagerly on every render.

### 1. ⚡ Executive Dashboard
- Five KPI metric cards: Total Sales, Forecast Accuracy, Model R², Critical Alerts, Avg Inventory Turns
- Monthly sales bar chart overlaid with 3-month rolling average line
- Inventory health donut chart (Critical / High / Normal with colors)
- Three model performance gauges (WMAPE, R², Service Level) using Plotly Indicator
- Critical alert feed — card-per-item with stock health progress bars
- Top 5 stores and sales-by-family horizontal bar charts

### 2. 📈 Sales Analytics
- **Anomaly detection**: rolling z-score (window=30, threshold=2.8σ) plotted as red dots on the daily sales time series
- **Calendar heatmap**: day-of-week × week-of-year intensity heatmap of daily sales
- **Sales treemap**: store × family hierarchical area chart colored by revenue
- Seasonal analysis tabs: monthly average bar, day-of-week bar (weekends highlighted), violin distribution by family, log-sales histogram
- CSV export download button

### 3. 🔮 Forecasting
- Live model metrics row (Val WMAPE, MAPE, R², Train WMAPE) loaded from `models/xgboost_model.joblib`
- 30-day forecast with **P10–P90 quantile bands** (real quantile-regression bounds — asymmetric, demand-aware) overlaid on the last 60 days of actual sales; falls back to a clearly-labeled heuristic band only if the forecast data lacks quantile columns
- Forecast by family (color-scaled bar)
- Forecast by store (color-scaled bar)
- Feature importance horizontal bar (red = high impact, blue = medium, gray = low)
- CSV export of forecast data

### 4. 📦 Inventory Command Center
- KPI row: Total SKUs, Critical count, Pending Reorders, Avg Turns
- **Priority action queue**: card per item with stock health progress bar, badge color-coded by urgency
- **ABC Pareto chart**: dual-axis (bar = item value, line = cumulative %) with class A/B boundary lines
- Full inventory table with sort and filter; CSV export

### 5. 🎛️ Scenario Planner *(new)*
Interactive what-if optimizer powered by `scipy.stats.norm`:
- **Sliders**: service level (85–99%), lead time, review period, ordering cost, holding rate, unit cost
- **Number inputs**: mean daily demand, demand standard deviation
- **Six result tiles**: Safety Stock, EOQ, Reorder Point, Annual Total Cost, Holding Cost, Ordering Cost — each showing % change vs. a computed baseline
- **Sensitivity chart**: total cost + safety stock vs. service level (dual axis), with vertical marker at current selection
- **Trade-off chart**: holding cost vs. ordering cost vs. total cost as ordering cost varies — shows the classic EOQ crossover

### 6. 🏪 Store Intelligence *(new)*
- **Normalized heatmap**: store × family relative performance (each store normalized to its own peak)
- **Absolute heatmap**: raw sales volume by store × family
- **Store rankings**: bar chart showing each store's % deviation from the network average; sortable table with CV, growth
- **Store vs. Store comparator**: pick any two stores, see overlaid daily sales + summary stats table (total, daily avg, peak, lowest, std dev, MoM growth)

---

## Performance Numbers

Measured on synthetic dataset (10 stores × 5 families × 3 years = 54,750 rows):

| Metric | Training Set | Validation Set |
|--------|-------------|----------------|
| **WMAPE** | 4.57% | 6.19% |
| **MAPE** | ~5.1% | ~6.8% |
| **R²** | 0.9832 | 0.9648 |
| **Directional Accuracy** | ~94% | ~91% |

> **What these numbers mean:**
> - WMAPE of 6.19% means the model's forecast volume is within 6.19% of actual volume when weighted by sales size.
> - R² of 0.9648 means the model explains 96.5% of variance in validation sales.
> - Industry benchmark for good retail forecasting: WMAPE < 15%. This system achieves less than half that threshold.

### Pipeline Runtime

| Phase | Time (approx) |
|-------|--------------|
| Data generation + loading | ~3s |
| Data cleaning | ~1s |
| Feature engineering (54,750 rows) | ~8s |
| Model training (1,000 trees) | ~4s |
| Forecast generation | ~2s |
| Inventory optimization | ~1s |
| **Total end-to-end** | **~19s** |

### Test Suite

```
25 tests — all pass
Coverage:
  DataCleaner           4 tests
  FeatureEngineer       5 tests
  XGBoostModel          5 tests
  ModelEvaluator        3 tests
  SafetyStockCalculator 3 tests
  ReorderPointOptimizer 2 tests
  InventoryOptimizer    3 tests
```

---

## Technology Stack

### Core ML & Data
| Library | Version | Role |
|---------|---------|------|
| **Python** | 3.9+ | Primary language |
| **pandas** | 2.x | Data manipulation, time series handling |
| **numpy** | 1.21+ | Vectorized numerical computation |
| **XGBoost** | 1.6+ | Primary forecasting model |
| **scikit-learn** | 1.1+ | Preprocessing, metrics, cross-validation |
| **statsmodels** | 0.13+ | Ljung-Box autocorrelation test, statistical diagnostics |
| **scipy** | 1.9+ | Statistical tests (Jarque-Bera, normality) |

### Optimization
| Library | Version | Role |
|---------|---------|------|
| **PuLP** | 2.6+ | Linear programming (inventory optimization) |
| **Optuna** | 3.0+ | Hyperparameter tuning (optional) |

### Infrastructure
| Library | Version | Role |
|---------|---------|------|
| **joblib** | 1.1+ | Model serialization / deserialization |
| **PyYAML** | 6.0+ | Configuration management |
| **loguru** | 0.6+ | Structured logging with rotation |

### Visualization & Dashboard
| Library | Version | Role |
|---------|---------|------|
| **Streamlit** | 1.12+ | Interactive web dashboard |
| **Plotly** | 5.10+ | Interactive charts (line, bar, scatter, histogram, pie) |
| **seaborn** | 0.11+ | Statistical plotting |
| **matplotlib** | 3.5+ | Static figure backend |

### Testing
| Library | Version | Role |
|---------|---------|------|
| **pytest** | 7.0+ | Unit and integration tests |
| **pytest-cov** | 4.0+ | Test coverage reporting |

### Why These Choices?

**XGBoost as the primary learner, LightGBM as the ensemble partner:** Both are best-in-class for tabular time-series regression and dominate retail-forecasting competitions (M5, Favorita). Their errors are decorrelated enough that averaging the two beats either alone, so the pipeline trains both and ensembles them rather than picking just one. (CatBoost would be a fine third member.)

**Streamlit over Dash:** Requires ~3× less code for equivalent dashboards and handles state management automatically. Dash offers more customization but at significant complexity cost.

**PuLP over scipy.optimize:** Clean algebraic modeling interface for EOQ/safety stock constraints. You write the mathematical model, not the solver implementation.

**loguru over Python's logging module:** Zero-configuration structured logging with automatic rotation. Writes to `logs/pipeline.log` with 30-day retention out of the box.

---

## How to Run — New Machine Setup

This section covers everything needed to run the project on a fresh machine. **No manual data download is required** — the pipeline auto-generates synthetic data if no Kaggle data is present.

### Step 1 — System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.9 or higher |
| RAM | 4 GB |
| Disk | 500 MB free |
| OS | macOS, Linux, or Windows |

Check your Python version:
```bash
python --version    # or: python3 --version
```

If Python is not installed: download from [python.org](https://www.python.org/downloads/)

### Step 2 — Get the Project

If you have the folder already (copy/transfer to new machine):
```bash
# No git required — just copy the project folder
```

### Step 3 — Create a Virtual Environment

```bash
# Navigate to the project directory
cd "Sales Forecasting & Inventory Optimization ML Project"

# Create a virtual environment (isolates dependencies from system Python)
python -m venv venv

# Activate it
# macOS / Linux:
source venv/bin/activate

# Windows (Command Prompt):
venv\Scripts\activate.bat

# Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should see `(venv)` at the start of your terminal prompt.

### Step 4 — Install Dependencies

Two dependency files are provided:

- **`requirements.txt`** — slim set for the **dashboard** only (Streamlit, Plotly, pandas, numpy, scipy, pyyaml, joblib, xgboost, pyarrow). This is what free hosting installs.
- **`requirements-full.txt`** — everything the **ML pipeline** needs (adds scikit-learn, statsmodels, PuLP, Optuna, matplotlib, seaborn, loguru, pytest). Every package listed is actually imported by the code — no unused heavy libraries.

```bash
pip install --upgrade pip
pip install -r requirements-full.txt   # full pipeline + dashboard
# or, dashboard only:
pip install -r requirements.txt
```

Takes 1–3 minutes on first run.

### Step 5 — Run the Pipeline

```bash
python main.py
```

What happens automatically:
1. Checks for data files in `data/raw/`
2. If missing → **generates 3 years of synthetic retail data** (10 stores, 5 product families)
3. Cleans and engineers 70+ features
4. Trains XGBoost model → saves to `models/xgboost_model.joblib`
5. Generates 30-day forecasts → saves to `data/processed/forecasts_xgboost.csv`
6. Runs inventory optimization → saves to `data/processed/inventory_recommendations.csv`

Expected output:
```
=== PIPELINE RESULTS ===
Models trained: ['xgboost']
Forecasts generated: 1500
Optimization completed: True
```

Runtime: ~20 seconds.

### Step 6 — Launch the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser. If it doesn't open automatically, copy-paste the URL.

### Step 7 — Run the Tests (Optional)

```bash
pytest tests/test_pipeline.py -v
```

Expected: `25 passed` in ~30 seconds.

---

### Using Real Kaggle Data (Optional)

If you want to use the actual Favorita store sales dataset:

1. Install Kaggle CLI (if not already):
   ```bash
   pip install kaggle
   ```

2. Get your Kaggle API key from [kaggle.com/account](https://www.kaggle.com/account) → "Create New API Token". Place `kaggle.json` at:
   - macOS/Linux: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`

3. Download the dataset:
   ```bash
   kaggle competitions download -c store-sales-time-series-forecasting
   unzip store-sales-time-series-forecasting.zip -d data/raw/
   ```

4. Run the pipeline as normal:
   ```bash
   python main.py
   ```

The pipeline automatically detects the real files and uses them instead of generating synthetic data.

---

### Running Specific Pipeline Phases

If you want to run only part of the pipeline:

```bash
python main.py --phase data       # Load + clean + feature engineer only
python main.py --phase train      # Above + train model
python main.py --phase forecast   # Above + generate forecasts
python main.py --phase optimize   # Full pipeline (same as python main.py)
```

### Troubleshooting

**`ModuleNotFoundError`** — Virtual environment not activated. Run `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows).

**`streamlit: command not found`** — Streamlit installed but not on PATH while venv is active. Run `python -m streamlit run app.py` instead.

**Dashboard shows "No processed data found"** — Run `python main.py` first, then refresh the browser.

**Out of memory on training** — Reduce `n_estimators` in `config/config.yaml` from 1000 to 200.

**Tests fail** — Ensure you're in the project root directory and the venv is active before running pytest.

---

## Project Structure

```
├── main.py                    # Pipeline orchestrator — run this first
├── app.py                     # Streamlit dashboard — run after pipeline
├── config/
│   └── config.yaml            # All hyperparameters and settings
├── src/
│   ├── data/
│   │   ├── data_loader.py     # Loads CSV datasets
│   │   ├── data_cleaner.py    # Dedup, outlier removal, null handling
│   │   └── feature_engineer.py # 70+ features: lag, rolling, temporal, holiday
│   ├── models/
│   │   ├── base_model.py      # Abstract base with cross-validate, save/load
│   │   └── xgboost_model.py   # XGBoost implementation + feature importance
│   ├── optimization/
│   │   ├── safety_stock.py    # Service-level-based safety stock
│   │   ├── reorder_point.py   # Demand × lead_time + safety stock
│   │   └── inventory_optimizer.py # EOQ, ABC analysis, recommendations
│   └── evaluation/
│       └── metrics.py         # WMAPE, MAPE, RMSE, MAE, R², Ljung-Box
├── tests/
│   └── test_pipeline.py       # 25 unit/integration tests
├── data/
│   ├── raw/                   # Input data (Kaggle or auto-generated)
│   └── processed/             # Pipeline outputs (features, forecasts, inventory)
├── models/
│   └── xgboost_model.joblib   # Trained model (created by pipeline)
└── logs/
    └── pipeline.log           # Rotating logs (30-day retention)
```

---

## Comparable Tools — And How This Differs

### Commercial Platforms

| Tool | What It Does | Price | Key Difference |
|------|-------------|-------|----------------|
| **Oracle Demand Management** | Enterprise demand forecasting + supply chain | $100K+/yr | Full ERP integration but no customization; black box |
| **SAP IBP** | Integrated business planning for S/4HANA shops | $200K+/yr | Requires SAP ecosystem; enormous implementation cost |
| **Blue Yonder (JDA)** | End-to-end supply chain planning | $50K–$500K/yr | Proven at scale but not accessible to SME |
| **Anaplan** | Connected planning for finance and supply chain | $30K–$100K/yr | Strong collaboration features; weaker ML |

**This system vs. commercial:** Zero licensing cost, full source access, customizable features and models, deployable on a laptop or cloud VM in 20 seconds.

### Open Source / Academic Tools

| Tool | Strength | Limitation vs. This System |
|------|---------|---------------------------|
| **Facebook Prophet** | Simple seasonal decomposition; fast | Univariate only; no external features; poor multi-SKU scaling |
| **statsmodels SARIMA** | Rigorous statistical framework | Univariate; separate model per series; slow at scale |
| **Nixtla (StatsForecast)** | High-performance multi-series forecasting | Forecasting only; no inventory optimization layer |
| **sktime** | Unified API for time series ML | No inventory optimization; more complex API |
| **Amazon Forecast** | Managed AutoML for time series | Cloud-only; black box; pay-per-forecast pricing |

### Where This System Sits

```
                   Forecast Only ◄────────────────────► Full Supply Chain
                                │                               │
                             Prophet                       Oracle / SAP
                             SARIMA                       Blue Yonder
                             Nixtla
                                      ▲
                               THIS SYSTEM
                    (Forecast + Inventory Optimization,
                      open source, fully transparent,
                      deployable in one command)
```

**What makes this different:**

1. **Forecast → Inventory is a single pipeline.** Most tools stop at the forecast. This system's inventory optimizer consumes the forecast output directly — reorder points are forward-looking, not based on historical averages.

2. **Full feature transparency.** Every feature, lag, and holiday proximity calculation is in plain Python. You can read, modify, and understand exactly what the model is learning from.

3. **Production engineering, not just a notebook.** Proper data pipeline, unit tests, configuration file, logging with rotation, model serialization, and a live dashboard — not a Jupyter notebook demonstration.

4. **No cloud dependency.** Runs entirely locally. No API keys, no per-inference costs, no data leaving your environment.

5. **Zero data setup required.** The pipeline auto-generates realistic synthetic data with proper autocorrelation — so you can evaluate the full system immediately on any machine.

---

## Tests

The test suite (`tests/test_pipeline.py`) covers every major component:

```python
class TestDataCleaner:         # Shape preservation, null drop, dedup, outlier cap
class TestFeatureEngineer:     # Temporal features, lag correctness, holiday NaN = 0
class TestXGBoostModel:        # Fit/predict shape, non-negative outputs, save/load roundtrip
class TestMetrics:             # Perfect prediction → zero error, all metric keys present
class TestSafetyStock:         # Positive output, higher service level → more stock
class TestReorderPoint:        # Exceeds safety stock, higher demand → higher ROP
class TestInventoryOptimizer:  # EOQ positive, ABC class assignment, A > C value
```

Run with:
```bash
pytest tests/test_pipeline.py -v --tb=short
```

---

*Built with Python · XGBoost · Streamlit · Plotly · PuLP*
