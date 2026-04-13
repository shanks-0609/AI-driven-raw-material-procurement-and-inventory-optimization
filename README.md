♦ ProcureAI — AI-Driven Raw Material Procurement & Inventory Optimization
> *We built this from scratch — every model, every algorithm, every line of the dashboard. No ML libraries. Just Python, math, and a lot of debugging.*
---
🧠 What This Is
ProcureAI is an end-to-end AI-powered procurement intelligence system we built to answer three questions that every operations team struggles with:
What raw materials do we need to order?
When should we order them — should we buy now or wait for prices to drop?
From whom should we buy, based on real supplier performance data?
The system tracks 10 raw materials across 36 months of simulated market data, evaluates 36 suppliers, runs 5 custom AI/ML models, and surfaces everything through a live browser-based dashboard — no server, no internet, just open the HTML file.
---
🏗️ Project Structure
```
ProcureAI/
├── main.py                    # Entry point — runs the full pipeline
├── dashboard.html             # Live analytics dashboard (open in browser)
│
├── data/
│   ├── generate_data.py       # Simulated dataset generator
│   ├── procurement_data.json  # Generated raw data (auto-created)
│   └── dashboard_data.json    # Dashboard output (auto-created)
│
└── modules/
    ├── demand_forecast.py     # Holt-Winters + AR(p) ensemble
    ├── price_prediction.py    # ARIMA(3,1,2) + Gradient Boosted Trees
    ├── supplier_scoring.py    # KPI scoring + K-Means clustering
    ├── inventory_risk.py      # EOQ, safety stock, stockout probability
    └── recommendation_engine.py  # Multi-signal decision engine
```
---
⚙️ How to Run It
1. Clone the repo and run the pipeline:
```bash
git clone https://github.com/your-username/procureai.git
cd procureai
python main.py
```
No pip installs needed — we only use Python standard library (`json`, `math`, `random`).
2. Open the dashboard:
Just open `dashboard.html` in any modern browser. That's it.
---
🔬 What We Built — Module by Module
Module 1 — Realistic Dataset Generator (`generate_data.py`)
We didn't use any real-world dataset. Instead, we built our own simulator that produces genuinely complex, non-trivial data:
Prices follow an Ornstein-Uhlenbeck mean-reverting process — the same model used in quantitative finance for commodity prices
Demand follows an AR(2) auto-correlated process with regime changes every ~10 months and random demand spikes
8 market shock events are embedded (trade tariffs, port strikes, energy price spikes, geopolitical tension) that affect each material differently based on shock sensitivity coefficients
Multi-cycle seasonality — both annual and quarterly patterns layered on top of each other
The result is data that actually behaves like real supply chain data — messy, non-linear, and with structural breaks.
---
Module 2 — Demand Forecasting (`demand_forecast.py`)
We ensemble two fundamentally different approaches and let them vote:
Model	Weight	Why
Holt-Winters Triple Exponential Smoothing	60%	Excellent at capturing additive seasonality
AR(p) Auto-Regression	40%	Strong short-term autocorrelation capture
AR order selection is automatic using AIC (Akaike Information Criterion) — we test AR(1) through AR(6) and pick the order that minimises information loss without overfitting.
Prediction intervals are generated via bootstrap resampling — we resample residuals 200 times to give honest 90% confidence bands rather than assuming a parametric distribution.
---
Module 3 — Price Prediction (`price_prediction.py`)
This was the hardest module to get right. We use two completely different model families and blend them:
ARIMA(3,1,2) — 55% weight
First-order differencing to handle non-stationarity
ARMA(3,2) fitted on the differenced series using iterative OLS with Ridge regularisation
Residuals are updated iteratively so MA terms are properly estimated (not just OLS regression)
Gradient Boosted Decision Stumps — 45% weight
Built entirely from scratch using single-split regression stumps
Features: price lags (1, 2, 3, 6, 12 months), rolling mean, momentum, month index (seasonality proxy)
100 estimators with learning rate 0.05 — classic stagewise additive approach
On top of predictions, we compute technical indicators to generate BUY / WAIT / NEUTRAL signals:
RSI (14-period) — overbought/oversold detection
MACD — momentum and trend direction
Bollinger Bands — mean reversion and breakout signals
Volatility % — used to scale confidence interval width
A multi-factor signal score combines all four. Score ≤ -2 → BUY NOW, score ≥ 2 → WAIT, otherwise NEUTRAL.
---
Module 4 — Supplier Scoring (`supplier_scoring.py`)
We score every supplier across 6 KPIs using a weighted matrix:
KPI	Weight
On-Time Delivery	30%
Quality Score	25%
Price Competitiveness	20%
Responsiveness	12%
Fill Rate	8%
Defect Penalty	5%
On top of the base score, we apply:
Geopolitical risk adjustment based on the supplier's country (±3 points)
Vendor tenure bonus — experienced suppliers get a small reliability premium
Suppliers are then graded A+ / A / B / C / D and segmented into 3 tiers using K-Means clustering (initialised k-means++ style, 15 iterations) so the tier boundaries are data-driven, not arbitrary.
---
Module 5 — Inventory Risk Monitor (`inventory_risk.py`)
We calculate four metrics for each material in real time:
Economic Order Quantity (EOQ)
```
Q* = sqrt(2 × D × S / H)
```
where D = annual demand, S = ordering cost, H = holding cost per unit per year
Safety Stock
```
SS = Z × σ_demand × sqrt(lead_time_months)
```
Z = 1.645 for 95% service level (configurable to 90% or 99%)
Dynamic Days of Supply — calculated from current stock and actual forecast demand, not just historical averages
Stockout Probability — uses the Normal CDF approximation (Abramowitz & Stegun) to estimate P(demand during lead time > current stock). Above 40% triggers an immediate alert.
---
Module 6 — Recommendation Engine (`recommendation_engine.py`)
This is where all five modules come together. Every material gets a Decision Score (0–100):
```
Score = (Urgency × 0.40) + (Price Signal × 0.25) + (Demand Trend × 0.20) + (Inventory Level × 0.15) + Volatility Bonus
```
Recommendations are then priority-ranked and passed through a budget-constrained greedy optimiser — a knapsack-style algorithm that allocates across a ₹50L procurement cycle budget by benefit-per-rupee ratio. Split-order suggestions are generated when two suppliers have similar scores and the order quantity justifies splitting.
---
📊 Dashboard Features
The dashboard (`dashboard.html`) is a single self-contained HTML file with vanilla JS and Chart.js:
Overview — KPI cards, live alerts, top recommendations, price signals
Price Trends — Historical + 6-month forecast with 90% CI bands and BUY/WAIT/NEUTRAL banner
Demand Forecast — Holt-Winters + AR(p) ensemble charts with confidence intervals
Inventory Risk — Stock level bars, days of supply, critical alerts
Supplier Scores — Full sortable scorecard with grades, risk flags, and mini bar charts
Recommendations — Priority-ranked action list with timing, quantity, and supplier
Reports & Exports — One-click CSV/JSON download for every module
---
📈 Key Numbers
Metric	Value
Materials tracked	10
Suppliers evaluated	36
Historical data	36 months
AI models built	5
Python standard library only	✅
External ML dependencies	0
Dashboard server required	❌ (just open HTML)
---
💡 Design Decisions Worth Noting
Why no sklearn / statsmodels? — We wanted to actually understand what we were implementing. Building ARIMA differencing by hand, writing Gaussian elimination for OLS, and implementing bootstrap CI from scratch taught us far more than calling `.fit()` ever would.
Why simulate data? — Real procurement data is either proprietary or heavily messy in ways that obscure whether your models are working. Simulating with known properties (OU process, AR(2) demand, embedded shocks) lets us verify that our models are actually picking up the patterns we planted.
Why a static HTML dashboard? — We wanted anyone to be able to run this without setting up Flask, Node, or any server. Open one file, see everything.
---
🔧 Possible Extensions
Some things we thought about but didn't build:
Real-time price feeds via commodity APIs (LME, MCX)
Email/Slack alert integration for critical stockout events
Multi-plant inventory optimisation across locations
Supplier contract tracking and renewal reminders
LSTM / Transformer-based price forecasting (we deliberately avoided this to keep the math interpretable)
---
👥 Team
Built as a team project. Every module was designed, implemented, debugged, and integrated together.
---
📄 License
MIT — use it, learn from it, build on it.
---
If something's unclear, feel free to open an issue. We're happy to walk through any part of the code.
