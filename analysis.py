"""
analysis.py
Pharma Commercial Analytics — mirrors all 10 SQL queries in pandas/Python.
Produces identical output values and exports publication-quality plots.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
PALETTE = ["#1F4E79", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
           "#3B1F2B", "#44BBA4", "#E94F37"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
OUT = Path("outputs/plots")
OUT.mkdir(parents=True, exist_ok=True)

DB = Path("data/pharma.db")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load(db=DB):
    conn = sqlite3.connect(db)
    sales = pd.read_sql("SELECT * FROM sales", conn, parse_dates=["date"])
    hcp   = pd.read_sql("SELECT * FROM hcp_engagement", conn, parse_dates=["date"])
    mix   = pd.read_sql("SELECT * FROM market_mix", conn, parse_dates=["date"])
    conn.close()
    return sales, hcp, mix


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS  (each prints df + saves chart)
# ══════════════════════════════════════════════════════════════════════════════

def q1_brand_annual(sales):
    """Q1 — Brand-level annual performance."""
    df = (
        sales.groupby(["brand", "therapy_area", "year"])
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue_usd", "sum"),
            avg_market_share=("market_share_pct", "mean"),
        )
        .reset_index()
    )
    df["avg_price_per_unit"] = (df["total_revenue"] / df["total_units"]).round(2)
    df["total_revenue"] = df["total_revenue"].round(2)
    df["avg_market_share"] = df["avg_market_share"].round(2)
    df = df.sort_values(["year", "total_revenue"], ascending=[True, False])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Brand Annual Performance", fontsize=14, fontweight="bold")
    for ax, yr in zip(axes, [2021, 2022, 2023]):
        sub = df[df.year == yr]
        bars = ax.bar(sub["brand"], sub["total_revenue"] / 1e6, color=PALETTE[:4], edgecolor="white")
        ax.set_title(str(yr), fontweight="bold")
        ax.set_ylabel("Revenue ($M)" if yr == 2021 else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.bar_label(bars, fmt="%.1fM", label_type="edge", fontsize=8)
    fig.tight_layout()
    save(fig, "01_brand_annual_performance")
    return df


def q2_quarterly_trend(sales):
    """Q2 — Quarterly revenue trend."""
    df = (
        sales.groupby(["year", "quarter", "brand"])
        .agg(units=("units_sold", "sum"), revenue=("revenue_usd", "sum"))
        .reset_index()
    )
    df["revenue"] = df["revenue"].round(2)
    df["period"] = df["year"].astype(str) + "-" + df["quarter"]
    df = df.sort_values(["year", "quarter", "brand"])

    # Plot
    pivot = df.pivot_table(index="period", columns="brand", values="revenue", aggfunc="sum")
    periods = sorted(df["period"].unique(), key=lambda x: (int(x[:4]), x[5:]))
    pivot = pivot.reindex(periods)

    fig, ax = plt.subplots(figsize=(14, 5))
    pivot.plot(ax=ax, marker="o", linewidth=2, markersize=5)
    ax.set_title("Quarterly Revenue Trend by Brand", fontsize=13, fontweight="bold")
    ax.set_ylabel("Revenue (USD)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
    ax.legend(title="Brand")
    fig.tight_layout()
    save(fig, "02_quarterly_revenue_trend")
    return df


def q3_regional_performance(sales):
    """Q3 — Regional performance."""
    df = (
        sales.groupby(["region", "brand", "year"])
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue_usd", "sum"),
            avg_market_share=("market_share_pct", "mean"),
        )
        .reset_index()
    )
    df["total_revenue"] = df["total_revenue"].round(2)
    df["avg_market_share"] = df["avg_market_share"].round(2)

    # Plot — heatmap of revenue by region × brand (2023)
    sub = df[df.year == 2023].pivot_table(
        index="region", columns="brand", values="total_revenue", aggfunc="sum"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(sub / 1e6, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Revenue ($M)"})
    ax.set_title("Regional Revenue Heatmap — 2023", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_regional_heatmap")
    return df


def q4_territory_top10(sales):
    """Q4 — Top 10 territories."""
    df = (
        sales.groupby(["territory", "rep_id", "region"])
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue_usd", "sum"),
            brands_covered=("brand", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
        .head(10)
    )
    df["total_revenue"] = df["total_revenue"].round(2)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    bars = ax.barh(df["territory"], df["total_revenue"] / 1e6, color=colors)
    ax.set_xlabel("Total Revenue ($M)")
    ax.set_title("Top 10 Territories by Revenue", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="$%.1fM", label_type="edge", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "04_top10_territories")
    return df


def q5_hcp_sales_corr(sales, hcp):
    """Q5 — HCP engagement vs sales."""
    s = sales.groupby(["date", "territory", "brand"]).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue_usd", "sum"),
    ).reset_index()
    h = hcp.rename(columns={})
    df = s.merge(h, on=["date", "territory", "brand"], how="inner")
    df["revenue"] = df["revenue"].round(2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("HCP Engagement vs Sales", fontsize=13, fontweight="bold")
    axes[0].scatter(df["hcp_visits"], df["units_sold"], alpha=0.3, color=PALETTE[1], s=10)
    axes[0].set_xlabel("HCP Visits")
    axes[0].set_ylabel("Units Sold")
    axes[0].set_title("Visits vs Units")
    axes[1].scatter(df["hcp_calls"], df["revenue"] / 1e3, alpha=0.3, color=PALETTE[2], s=10)
    axes[1].set_xlabel("HCP Calls")
    axes[1].set_ylabel("Revenue ($K)")
    axes[1].set_title("Calls vs Revenue")
    fig.tight_layout()
    save(fig, "05_hcp_sales_correlation")
    return df


def q6_market_share_ma(sales):
    """Q6 — Rolling 3-month market share."""
    df = (
        sales.groupby(["brand", "date"])["market_share_pct"]
        .mean()
        .reset_index()
        .rename(columns={"market_share_pct": "monthly_avg_share"})
        .sort_values(["brand", "date"])
    )
    df["rolling_3m_share"] = (
        df.groupby("brand")["monthly_avg_share"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(2)
    )
    df["monthly_avg_share"] = df["monthly_avg_share"].round(2)

    fig, ax = plt.subplots(figsize=(14, 5))
    for brand, grp in df.groupby("brand"):
        ax.plot(grp["date"], grp["rolling_3m_share"], label=brand, linewidth=2)
    ax.set_title("3-Month Rolling Market Share by Brand", fontsize=13, fontweight="bold")
    ax.set_ylabel("Market Share (%)")
    ax.legend(title="Brand")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    fig.tight_layout()
    save(fig, "06_rolling_market_share")
    return df


def q7_yoy_growth(sales):
    """Q7 — YoY revenue growth."""
    yearly = (
        sales.groupby(["brand", "year"])["revenue_usd"]
        .sum().round(2).reset_index().rename(columns={"revenue_usd": "revenue"})
    )
    df = yearly.merge(
        yearly.assign(year=yearly["year"] + 1),
        on=["brand", "year"], suffixes=("", "_prior")
    )
    df["yoy_growth_pct"] = (
        100 * (df["revenue"] - df["revenue_prior"]) / df["revenue_prior"]
    ).round(2)
    df = df.rename(columns={"revenue_prior": "prior_revenue", "year": "current_year"})

    pivot = df.pivot(index="current_year", columns="brand", values="yoy_growth_pct")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=PALETTE[:4], edgecolor="white", width=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Year-over-Year Revenue Growth (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("YoY Growth (%)")
    ax.set_xlabel("Year")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Brand")
    fig.tight_layout()
    save(fig, "07_yoy_revenue_growth")
    return df


def q8_market_mix(sales, mix):
    """Q8 — Market mix spend vs revenue."""
    m = mix.copy()
    m["total_spend"] = (m["tv_spend_usd"] + m["digital_spend_usd"] + m["print_spend_usd"]).round(2)
    s = sales.groupby(["brand", "date"])["revenue_usd"].sum().reset_index()
    df = m.merge(s, on=["brand", "date"], how="inner")
    df = df.rename(columns={"revenue_usd": "brand_revenue"})
    df["revenue_per_spend_dollar"] = (df["brand_revenue"] / df["total_spend"]).round(2)

    # Stacked spend chart per brand
    spend_cols = ["tv_spend_usd", "digital_spend_usd", "print_spend_usd"]
    brand_spend = m.groupby("brand")[spend_cols].mean().round(2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    brand_spend.plot(kind="bar", ax=axes[0], stacked=True,
                     color=[PALETTE[0], PALETTE[1], PALETTE[2]], edgecolor="white")
    axes[0].set_title("Avg Monthly Spend by Channel", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Spend (USD)")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(["TV", "Digital", "Print"])

    roi = df.groupby("brand")["revenue_per_spend_dollar"].mean().sort_values(ascending=False)
    roi.plot(kind="bar", ax=axes[1], color=PALETTE[3:7], edgecolor="white")
    axes[1].set_title("Revenue per $ Spend (ROI)", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Revenue / Spend $")
    axes[1].tick_params(axis="x", rotation=0)
    fig.suptitle("Market Mix Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_market_mix_analysis")
    return df


def q9_rep_productivity(sales, hcp):
    """Q9 — Rep-level productivity."""
    s = sales.groupby(["rep_id", "region"]).agg(total_units=("units_sold", "sum")).reset_index()
    h = hcp.groupby(["rep_id"]).agg(total_visits=("hcp_visits", "sum")).reset_index()
    df = s.merge(h, on="rep_id", how="inner")
    df["units_per_visit"] = (df["total_units"] / df["total_visits"]).round(1)
    df = df.sort_values("units_per_visit", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [PALETTE[["North","South","East","West"].index(r) % 4] for r in df["region"]]
    bars = ax.bar(df["rep_id"], df["units_per_visit"], color=colors, edgecolor="white")
    ax.set_title("Rep Productivity — Units per HCP Visit", fontsize=13, fontweight="bold")
    ax.set_ylabel("Units per Visit")
    ax.tick_params(axis="x", rotation=90)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PALETTE[i], label=r)
                       for i, r in enumerate(["North", "South", "East", "West"])]
    ax.legend(handles=legend_elements, title="Region")
    fig.tight_layout()
    save(fig, "09_rep_productivity")
    return df


def q10_demand_visibility(sales):
    """Q10 — Last quarter actuals (2023-Q4)."""
    df = (
        sales[(sales.year == 2023) & (sales.quarter == "Q4")]
        .groupby(["brand", "quarter", "year"])
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue_usd", "sum"),
            avg_share=("market_share_pct", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    df["total_revenue"] = df["total_revenue"].round(2)
    df["avg_share"] = df["avg_share"].round(2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Demand Visibility — 2023 Q4 Actuals", fontsize=13, fontweight="bold")
    wedge_props = dict(width=0.5, edgecolor="white", linewidth=2)
    axes[0].pie(df["total_revenue"], labels=df["brand"], autopct="%1.1f%%",
                colors=PALETTE[:4], wedgeprops=wedge_props)
    axes[0].set_title("Revenue Share", fontweight="bold")
    bars = axes[1].bar(df["brand"], df["avg_share"], color=PALETTE[:4], edgecolor="white")
    axes[1].set_title("Avg Market Share (%)", fontweight="bold")
    axes[1].set_ylabel("Market Share (%)")
    axes[1].bar_label(bars, fmt="%.1f%%", label_type="edge", fontsize=9)
    fig.tight_layout()
    save(fig, "10_demand_visibility_q4_2023")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FORECASTING  — Prophet-style decomposition with statsmodels + sklearn
# ══════════════════════════════════════════════════════════════════════════════

def forecast_brand_revenue(sales):
    """Time-series forecast per brand using ETS-like decomposition."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    monthly = (
        sales.groupby(["brand", "date"])["revenue_usd"]
        .sum().reset_index().sort_values(["brand", "date"])
    )

    HORIZON = 6  # months ahead
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Brand Revenue Forecast (+6 Months)", fontsize=14, fontweight="bold")
    axes = axes.flatten()
    results = []

    for ax, (brand, grp) in zip(axes, monthly.groupby("brand")):
        grp = grp.reset_index(drop=True)
        grp["t"] = np.arange(len(grp))

        # Features: trend (poly-2) + monthly seasonality
        X_hist = grp[["t"]].values
        pf = PolynomialFeatures(degree=2)
        X_poly = pf.fit_transform(X_hist)

        # Add sin/cos seasonality
        month_num = grp["date"].dt.month.values
        X_poly = np.hstack([
            X_poly,
            np.sin(2 * np.pi * month_num / 12).reshape(-1, 1),
            np.cos(2 * np.pi * month_num / 12).reshape(-1, 1),
        ])

        y = grp["revenue_usd"].values
        model = LinearRegression().fit(X_poly, y)
        y_pred_hist = model.predict(X_poly)

        # Future
        future_t = np.arange(len(grp), len(grp) + HORIZON)
        future_dates = pd.date_range(grp["date"].max() + pd.DateOffset(months=1),
                                     periods=HORIZON, freq="MS")
        future_months = future_dates.month.values
        X_fut = pf.transform(future_t.reshape(-1, 1))
        X_fut = np.hstack([
            X_fut,
            np.sin(2 * np.pi * future_months / 12).reshape(-1, 1),
            np.cos(2 * np.pi * future_months / 12).reshape(-1, 1),
        ])
        y_fut = model.predict(X_fut)

        # CI ±8%
        ci = y_fut * 0.08

        ax.plot(grp["date"], y / 1e6, color=PALETTE[0], label="Actual", linewidth=1.5)
        ax.plot(grp["date"], y_pred_hist / 1e6, color=PALETTE[1],
                linestyle="--", label="Fitted", linewidth=1, alpha=0.8)
        ax.plot(future_dates, y_fut / 1e6, color=PALETTE[2], label="Forecast", linewidth=2)
        ax.fill_between(future_dates, (y_fut - ci) / 1e6, (y_fut + ci) / 1e6,
                        color=PALETTE[2], alpha=0.2, label="±8% CI")
        ax.set_title(brand, fontweight="bold")
        ax.set_ylabel("Revenue ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fM"))
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        for d, v in zip(future_dates, y_fut):
            results.append({"brand": brand, "forecast_date": d.strftime("%Y-%m-%d"),
                            "forecast_revenue": round(v, 2)})

    fig.tight_layout()
    save(fig, "11_brand_revenue_forecast")

    fc_df = pd.DataFrame(results)
    print("\n  Forecast (next 6 months):")
    print(fc_df.to_string(index=False))
    return fc_df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  PHARMA COMMERCIAL ANALYTICS — Python Analysis")
    print("="*60)

    sales, hcp, mix = load()

    analyses = [
        ("Q1  Brand Annual Performance",      lambda: q1_brand_annual(sales)),
        ("Q2  Quarterly Revenue Trend",        lambda: q2_quarterly_trend(sales)),
        ("Q3  Regional Performance",           lambda: q3_regional_performance(sales)),
        ("Q4  Top-10 Territory Alignment",     lambda: q4_territory_top10(sales)),
        ("Q5  HCP Engagement vs Sales",        lambda: q5_hcp_sales_corr(sales, hcp)),
        ("Q6  3-Month Rolling Market Share",   lambda: q6_market_share_ma(sales)),
        ("Q7  YoY Revenue Growth",             lambda: q7_yoy_growth(sales)),
        ("Q8  Market Mix Spend vs Revenue",    lambda: q8_market_mix(sales, mix)),
        ("Q9  Rep Productivity",               lambda: q9_rep_productivity(sales, hcp)),
        ("Q10 Demand Visibility Q4-2023",      lambda: q10_demand_visibility(sales)),
        ("F11 Brand Revenue Forecast",         lambda: forecast_brand_revenue(sales)),
    ]

    all_results = {}
    for label, fn in analyses:
        print(f"\n── {label} ──")
        try:
            df = fn()
            print(df.head(5).to_string(index=False))
            all_results[label] = df
        except Exception as e:
            print(f"  [!] Error: {e}")
            raise

    print("\n" + "="*60)
    print(f"  All analyses complete. Plots saved to → {OUT}")
    print("="*60 + "\n")
    return all_results


if __name__ == "__main__":
    main()
