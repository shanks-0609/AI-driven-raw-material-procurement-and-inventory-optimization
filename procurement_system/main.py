"""
╔══════════════════════════════════════════════════════════════╗
║     AI-Driven Raw Material Procurement Optimization System   ║
║                      v2.0 — Complete Build                   ║
╚══════════════════════════════════════════════════════════════╝

WHAT'S NEW in v2.0:
  ✅ Holt-Winters Triple Exponential Smoothing for demand
  ✅ AR(p) Auto-Regression with AIC-based order selection
  ✅ ARIMA(3,1,2) price forecasting (differencing + ARMA)
  ✅ Gradient Boosted Decision Stumps for price prediction
  ✅ MACD, Bollinger Bands, RSI technical indicators
  ✅ K-Means supplier clustering (3 tiers)
  ✅ EOQ + Safety Stock calculations
  ✅ Stockout probability estimation
  ✅ Budget-constrained greedy optimization
  ✅ Split-order supplier suggestions
  ✅ Cost savings estimation
  ✅ 10 materials, 36 months, complex dataset

HOW TO RUN:
  1. python main.py
  2. Open dashboard.html in any browser
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 65)
    print("  AI Procurement Optimization System — v2.0")
    print("=" * 65)

    data_path = os.path.join(os.path.dirname(__file__), "data", "procurement_data.json")

    # Step 1: Generate data
    print("\n[1/5] Generating complex simulation dataset...")
    from data.generate_data import build_full_dataset
    data = build_full_dataset()
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    with open(data_path.replace(".json", ".js"), "w") as f:
        f.write("window.invData_var = ")
        json.dump(data, f, indent=2)
        f.write(";")
    print(f"      ✓ {len(data['materials'])} materials | {len(data['suppliers'])} suppliers | 36 months history")

    # Step 2: Demand Forecasting
    print("\n[2/5] Running advanced demand forecasting...")
    print("      Models: Holt-Winters + AR(p) Ensemble")
    from modules.demand_forecast import run_all_forecasts
    demand_results = run_all_forecasts(data_path)
    ok = sum(1 for r in demand_results.values() if "error" not in r)
    print(f"      ✓ {ok}/{len(demand_results)} materials forecasted successfully")

    # Step 3: Price Prediction
    print("\n[3/5] Running ARIMA + Gradient Boost price predictions...")
    from modules.price_prediction import run_all_price_predictions
    price_results = run_all_price_predictions(data_path)
    buy_signals = sum(1 for r in price_results.values() if r.get("price_signal") == "BUY NOW")
    wait_signals = sum(1 for r in price_results.values() if r.get("price_signal") == "WAIT")
    neutral = sum(1 for r in price_results.values() if r.get("price_signal") == "NEUTRAL")
    print(f"      ✓ BUY NOW: {buy_signals} | WAIT: {wait_signals} | NEUTRAL: {neutral}")

    # Step 4: Supplier Evaluation
    print("\n[4/5] Evaluating suppliers with K-Means clustering...")
    from modules.supplier_scoring import generate_supplier_report
    ranked = generate_supplier_report(data_path)
    total_sups = sum(len(v) for v in ranked.values())
    print(f"      ✓ {total_sups} suppliers scored, graded, and clustered into 3 tiers")

    # Step 5: Recommendations
    print("\n[5/5] Generating budget-optimized procurement recommendations...")
    from modules.recommendation_engine import generate_recommendations, print_report
    recs = generate_recommendations(data_path)
    print_report(recs)

    urgent = [r for r in recs if r["priority"] in ("Critical", "High")]
    total_cost = sum(r.get("estimated_cost", 0) for r in recs if r.get("budget_allocated") is True)
    total_savings = sum(r.get("estimated_savings", 0) for r in recs if r.get("estimated_savings", 0) > 0)

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"  Urgent actions  : {len(urgent)}")
    print(f"  Total order cost: ₹{total_cost:,.0f}")
    print(f"  Estimated savings: ₹{total_savings:,.0f}")
    print(f"\n  Open dashboard.html in your browser to view the full UI.")
    print("=" * 65)

    # Write combined output for dashboard
    output = {
        "demand": demand_results,
        "prices": price_results,
        "recommendations": recs,
        "materials": data["materials"],
    }
    output_path = os.path.join(os.path.dirname(__file__), "data", "dashboard_data.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    with open(output_path.replace(".json", ".js"), "w") as f:
        f.write("window.dashData_var = ")
        json.dump(output, f, indent=2, default=str)
        f.write(";")
    print(f"\n  Dashboard data written to: {output_path}")


if __name__ == "__main__":
    main()
