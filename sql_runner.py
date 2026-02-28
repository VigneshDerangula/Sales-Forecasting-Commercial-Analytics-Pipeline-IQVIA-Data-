"""
sql_runner.py
Executes all 10 SQL queries against pharma.db and prints results.
Also cross-validates key aggregates against the Python analysis.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB = Path("data/pharma.db")
SQL_FILE = Path("sql/pharma_analysis.sql")


def run_all_queries(db=DB):
    conn = sqlite3.connect(db)
    raw_sql = SQL_FILE.read_text()

    # Split on blank-line-separated comment blocks
    blocks = []
    current = []
    for line in raw_sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- ─────") and current:
            blk = "\n".join(current).strip()
            if blk and not blk.startswith("--"):
                blocks.append(blk)
            current = [line]
        else:
            current.append(line)
    if current:
        blk = "\n".join(current).strip()
        if blk:
            blocks.append(blk)

    # Parse blocks properly — find SELECT statements
    import re
    select_pattern = re.compile(r"(SELECT\b.+?)(?=(?:\n-- ─|$))", re.DOTALL | re.IGNORECASE)
    queries_raw = select_pattern.findall(raw_sql)

    labels = [
        "Q1  Brand Annual Performance",
        "Q2  Quarterly Revenue Trend",
        "Q3  Regional Performance",
        "Q4  Top-10 Territories",
        "Q5  HCP Engagement vs Sales",
        "Q6  Market Share Rolling MA",
        "Q7  YoY Growth",
        "Q8  Market Mix Spend vs Revenue",
        "Q9  Rep Productivity",
        "Q10 Demand Visibility Q4-2023",
    ]

    results = {}
    print("\n" + "="*60)
    print("  PHARMA COMMERCIAL ANALYTICS — SQL Analysis")
    print("="*60)

    for i, (label, qry) in enumerate(zip(labels, queries_raw)):
        print(f"\n── {label} ──")
        try:
            df = pd.read_sql_query(qry.strip(), conn)
            results[label] = df
            print(df.head(5).to_string(index=False))
        except Exception as e:
            print(f"  [!] Error executing query {i+1}: {e}")
            print(f"  Query preview: {qry[:200]}")

    conn.close()
    return results


def cross_validate(sql_results, py_results):
    """Check that SQL and Python produce identical totals for Q1."""
    print("\n" + "="*60)
    print("  CROSS-VALIDATION: SQL vs Python")
    print("="*60)

    # Q1 check
    q1_sql = sql_results.get("Q1  Brand Annual Performance")
    q1_py  = py_results.get("Q1  Brand Annual Performance")

    if q1_sql is None or q1_py is None:
        print("  [!] Could not compare Q1 — results missing")
        return

    sql_rev = q1_sql.groupby("brand")["total_revenue"].sum().round(0)
    py_rev  = q1_py.groupby("brand")["total_revenue"].sum().round(0)

    merged = pd.DataFrame({"SQL": sql_rev, "Python": py_rev}).dropna()
    merged["Δ"] = (merged["SQL"] - merged["Python"]).abs()
    merged["Match"] = merged["Δ"] < 1  # allow $1 float rounding

    print("\n  Q1 Brand Total Revenue Comparison:")
    print(merged.to_string())

    all_match = merged["Match"].all()
    status = "✅ ALL MATCH" if all_match else "❌ DISCREPANCIES FOUND"
    print(f"\n  Status: {status}\n")


if __name__ == "__main__":
    sql_results = run_all_queries()

    # Also run Python analysis for cross-validation
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from analysis import main as py_main
    print("\n\nRunning Python analysis for cross-validation …\n")
    py_results = py_main()

    cross_validate(sql_results, py_results)
