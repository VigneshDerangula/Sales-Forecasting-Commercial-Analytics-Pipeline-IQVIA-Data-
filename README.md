#  Pharma Sales Forecasting & Demand Planning

> **Stack:** Python · SQL · pandas · scikit-learn · matplotlib · seaborn · SQLite

---

## 📌 Project Overview

End-to-end commercial analytics pipeline built on 3 years of synthetic **IQVIA-style** brand-level pharma sales data.  
Covers every KPI that pharma recruiters screen for — territory alignment, HCP engagement, market mix, demand visibility, and revenue forecasting.

---

## 🗂 Repository Structure

```
pharma-sales-forecasting/
├── data/
│   ├── pharma.db            ← SQLite database (auto-generated)
│   ├── sales.csv
│   ├── hcp_engagement.csv
│   └── market_mix.csv
├── sql/
│   └── pharma_analysis.sql  ← 10 production-grade SQL queries
├── src/
│   ├── generate_data.py     ← Synthetic data generator
│   ├── analysis.py          ← Python analytics + visualisations
│   └── sql_runner.py        ← SQL executor + SQL vs Python validation
├── outputs/
│   └── plots/               ← 11 publication-quality PNG charts
├── notebooks/
│   └── (add Jupyter notebooks here)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<your-handle>/pharma-sales-forecasting.git
cd pharma-sales-forecasting
pip install -r requirements.txt

# 2. Generate synthetic data
python src/generate_data.py

# 3. Run Python analysis + save all plots
python src/analysis.py

# 4. Run SQL queries + cross-validate against Python
python src/sql_runner.py
```

---

## 📊 Analyses Included

| # | Analysis | Business Question |
|---|----------|------------------|
| Q1 | Brand Annual Performance | Which brands are growing revenue YoY? |
| Q2 | Quarterly Revenue Trend | Are we hitting quarterly commercial targets? |
| Q3 | Regional Heatmap | Which regions over/under-index by brand? |
| Q4 | Territory Alignment (Top 10) | Which territories drive the most value? |
| Q5 | HCP Engagement vs Sales | Do more HCP visits correlate with higher units sold? |
| Q6 | Rolling 3-Month Market Share | Is our share trending up or eroding? |
| Q7 | YoY Revenue Growth | What is the compound growth story per brand? |
| Q8 | Market Mix Spend vs ROI | Which channels give the best revenue per dollar? |
| Q9 | Rep Productivity | Units per HCP visit by rep — who is most efficient? |
| Q10 | Demand Visibility Q4-2023 | What are the latest actuals for commercial planning? |
| F11 | 6-Month Revenue Forecast | Polynomial + seasonality model with 95% CI band |

---

## 🗄 Data Schema

### `sales`
| Column | Type | Description |
|--------|------|-------------|
| date | DATE | Monthly date (YYYY-MM-01) |
| brand | TEXT | Brand name (A–D) |
| therapy_area | TEXT | Cardiovascular / Oncology / CNS / Respiratory |
| region | TEXT | North / South / East / West |
| territory | TEXT | Territory code (e.g. N-101) |
| rep_id | TEXT | Field rep identifier |
| units_sold | INT | Monthly units sold |
| revenue_usd | REAL | Monthly revenue |
| market_share_pct | REAL | Brand market share (%) |

### `hcp_engagement`
| Column | Type | Description |
|--------|------|-------------|
| hcp_calls | INT | Phone/remote interactions |
| hcp_visits | INT | In-person visits |
| samples_distributed | INT | Samples left with HCPs |
| digital_interactions | INT | Email / portal touchpoints |

### `market_mix`
| Column | Type | Description |
|--------|------|-------------|
| tv_spend_usd | REAL | TV advertising spend |
| digital_spend_usd | REAL | Digital channel spend |
| print_spend_usd | REAL | Print/congress materials |
| congress_events | INT | Events attended |
| competitor_launches | INT | Competitive launches in period |

---

## 📈 Output Charts

| File | Chart |
|------|-------|
| `01_brand_annual_performance.png` | Grouped bar — revenue per brand per year |
| `02_quarterly_revenue_trend.png` | Multi-line trend across 12 quarters |
| `03_regional_heatmap.png` | Revenue heatmap (region × brand) |
| `04_top10_territories.png` | Horizontal bar — top territories |
| `05_hcp_sales_correlation.png` | Scatter — HCP visits vs units, calls vs revenue |
| `06_rolling_market_share.png` | Line — 3-month rolling share per brand |
| `07_yoy_revenue_growth.png` | Grouped bar — YoY growth % |
| `08_market_mix_analysis.png` | Stacked bar spend + ROI bar |
| `09_rep_productivity.png` | Bar — units per HCP visit |
| `10_demand_visibility_q4_2023.png` | Donut + bar — Q4 actuals |
| `11_brand_revenue_forecast.png` | Actual + fitted + 6-month forecast with CI |

---

## ✅ SQL vs Python Validation

`sql_runner.py` runs both the SQL and Python pipelines and cross-validates that total brand revenues match within $1 (float rounding only).  
This demonstrates production-grade consistency between the two analysis layers.

---

## 🧠 Skills Demonstrated

- **Time-Series Forecasting** — Polynomial trend + Fourier seasonality, sklearn pipeline
- **Commercial Analytics** — Revenue, market share, YoY growth, demand planning
- **Territory Alignment** — Rep productivity, territory ranking, HCP engagement
- **Market Mix Modelling** — Channel spend ROI, multi-channel attribution
- **SQL Analytics** — Window functions, CTEs, multi-table JOINs, aggregations
- **Data Engineering** — SQLite DB, CSV data lake, reproducible data generation
- **Visualisation** — 11 publication-quality matplotlib/seaborn charts

---

## 📄 License

MIT — free to use, adapt, and include in your portfolio.
