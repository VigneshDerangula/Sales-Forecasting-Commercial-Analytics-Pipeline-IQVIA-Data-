"""
generate_data.py
Synthetic pharma sales data generator — mimics IQVIA-style brand-level data.
"""

import pandas as pd
import numpy as np
import sqlite3
import os

np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD"]
REGIONS = ["North", "South", "East", "West"]
TERRITORIES = {
    "North": ["N-101", "N-102", "N-103"],
    "South": ["S-201", "S-202", "S-203"],
    "East":  ["E-301", "E-302", "E-303"],
    "West":  ["W-401", "W-402", "W-403"],
}
REPS = {t: f"REP_{t}" for region in TERRITORIES.values() for t in region}
THERAPY_AREAS = {
    "BrandA": "Cardiovascular",
    "BrandB": "Oncology",
    "BrandC": "CNS",
    "BrandD": "Respiratory",
}
START = pd.Timestamp("2021-01-01")
END   = pd.Timestamp("2023-12-31")
DATES = pd.date_range(START, END, freq="MS")  # monthly


def base_trend(n, growth=0.01):
    return np.array([1 + growth * i for i in range(n)])


def seasonality(dates):
    month = dates.month
    # Q4 pharma push + mid-year dip
    seasonal = 1 + 0.08 * np.sin(2 * np.pi * (month - 3) / 12)
    return seasonal


def generate_sales():
    rows = []
    n = len(DATES)
    seasonal = seasonality(DATES)

    brand_params = {
        "BrandA": dict(base=5000, growth=0.008, noise=0.06),
        "BrandB": dict(base=3200, growth=0.015, noise=0.08),
        "BrandC": dict(base=4100, growth=0.005, noise=0.07),
        "BrandD": dict(base=2800, growth=0.012, noise=0.09),
    }

    for brand, params in brand_params.items():
        for region, terrs in TERRITORIES.items():
            region_factor = {"North": 1.15, "South": 0.90, "East": 1.05, "West": 0.95}[region]
            for terr in terrs:
                trend = base_trend(n, params["growth"])
                noise = np.random.normal(1, params["noise"], n)
                units = (params["base"] * trend * seasonal * region_factor * noise).astype(int)
                units = np.clip(units, 100, None)
                price_per_unit = np.random.uniform(45, 95)
                revenue = (units * price_per_unit * np.random.normal(1, 0.02, n)).round(2)

                for i, date in enumerate(DATES):
                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "year": date.year,
                        "month": date.month,
                        "quarter": f"Q{date.quarter}",
                        "brand": brand,
                        "therapy_area": THERAPY_AREAS[brand],
                        "region": region,
                        "territory": terr,
                        "rep_id": REPS[terr],
                        "units_sold": int(units[i]),
                        "revenue_usd": float(revenue[i]),
                        "market_share_pct": round(np.random.uniform(8, 32), 2),
                    })

    return pd.DataFrame(rows)


def generate_hcp_engagement(sales_df):
    """HCP visits / calls linked to territories."""
    rows = []
    for _, grp in sales_df.groupby(["date", "territory", "brand"]):
        row = grp.iloc[0]
        rows.append({
            "date": row["date"],
            "territory": row["territory"],
            "brand": row["brand"],
            "rep_id": row["rep_id"],
            "hcp_calls": int(np.random.poisson(22)),
            "hcp_visits": int(np.random.poisson(8)),
            "samples_distributed": int(np.random.poisson(40)),
            "digital_interactions": int(np.random.poisson(55)),
        })
    return pd.DataFrame(rows)


def generate_market_mix(sales_df):
    rows = []
    for _, grp in sales_df.groupby(["date", "brand"]):
        row = grp.iloc[0]
        rows.append({
            "date": row["date"],
            "brand": row["brand"],
            "tv_spend_usd": round(np.random.uniform(10000, 80000), 2),
            "digital_spend_usd": round(np.random.uniform(5000, 40000), 2),
            "print_spend_usd": round(np.random.uniform(2000, 15000), 2),
            "congress_events": int(np.random.poisson(2)),
            "competitor_launches": int(np.random.binomial(3, 0.2)),
        })
    return pd.DataFrame(rows)


def save_to_sqlite(sales_df, hcp_df, mix_df, db_path):
    conn = sqlite3.connect(db_path)
    sales_df.to_sql("sales", conn, if_exists="replace", index=False)
    hcp_df.to_sql("hcp_engagement", conn, if_exists="replace", index=False)
    mix_df.to_sql("market_mix", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[✓] SQLite DB saved → {db_path}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating synthetic pharma data …")
    sales = generate_sales()
    hcp   = generate_hcp_engagement(sales)
    mix   = generate_market_mix(sales)

    sales.to_csv("data/sales.csv", index=False)
    hcp.to_csv("data/hcp_engagement.csv", index=False)
    mix.to_csv("data/market_mix.csv", index=False)
    save_to_sqlite(sales, hcp, mix, "data/pharma.db")

    print(f"[✓] Sales rows      : {len(sales):,}")
    print(f"[✓] HCP rows        : {len(hcp):,}")
    print(f"[✓] Market-mix rows : {len(mix):,}")
