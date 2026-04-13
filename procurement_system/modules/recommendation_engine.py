"""
Module 5: Smart Procurement Recommendation Engine
Enhanced with:
  - Multi-signal composite scoring (weighted decision matrix)
  - Budget-constrained greedy optimization (LP approximation)
  - Multi-supplier split order suggestions
  - Cost savings estimation
  - Risk-adjusted procurement timing
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.demand_forecast import forecast_demand
from modules.price_prediction import predict_prices
from modules.supplier_scoring import rank_suppliers_by_material, score_supplier
from modules.inventory_risk import assess_inventory_risk, compute_reorder_qty


BUDGET_LIMIT = 5_000_000  # INR / per cycle — adjustable


def compute_decision_score(urgency, price_signal, demand_trend, inv_level, volatility):
    """
    Weighted decision matrix to rank procurement priority.
    Returns score 0–100.
    """
    score = urgency * 0.40

    price_weight = {"BUY NOW": 15, "NEUTRAL": 7, "WAIT": 0}
    score += price_weight.get(price_signal, 7) * 0.25

    demand_weight = {"Increasing": 15, "Stable": 8, "Decreasing": 2}
    score += demand_weight.get(demand_trend, 8) * 0.20

    inv_weight = {"Critical": 20, "Low": 15, "Adequate": 5, "Overstocked": 0}
    score += inv_weight.get(inv_level, 5) * 0.15

    # Volatility bonus (high volatility → act sooner)
    vol_bonus = min(5, volatility / 4) if volatility else 0
    score += vol_bonus

    return round(min(100, score), 1)


def budget_constrained_optimization(recommendations, budget=BUDGET_LIMIT):
    """
    Greedy knapsack-style budget allocation.
    Prioritizes by decision_score / cost ratio (benefit per rupee).
    Returns recommendations with allocated quantities and budget flags.
    """
    remaining = budget
    for rec in sorted(recommendations, key=lambda x: x["decision_score"], reverse=True):
        cost = rec["recommended_qty"] * rec["current_price"]
        if cost <= 0:
            rec["budget_allocated"] = True
            rec["allocated_qty"] = rec["recommended_qty"]
            rec["estimated_cost"] = 0
            continue
        if remaining >= cost:
            rec["budget_allocated"] = True
            rec["allocated_qty"] = rec["recommended_qty"]
            rec["estimated_cost"] = round(cost, 2)
            remaining -= cost
        elif remaining > 0:
            # Partial order
            partial_qty = int(remaining / rec["current_price"])
            rec["budget_allocated"] = "Partial"
            rec["allocated_qty"] = partial_qty
            rec["estimated_cost"] = round(partial_qty * rec["current_price"], 2)
            remaining -= rec["estimated_cost"]
        else:
            rec["budget_allocated"] = False
            rec["allocated_qty"] = 0
            rec["estimated_cost"] = 0
        rec["remaining_budget_after"] = round(remaining, 2)
    return recommendations


def estimate_savings(price_signal, pct_change, qty, current_price):
    """Estimate potential savings or cost from waiting vs buying now."""
    if price_signal == "BUY NOW" and pct_change > 0:
        future_cost = qty * current_price * (1 + pct_change / 100)
        current_cost = qty * current_price
        return round(future_cost - current_cost, 2)
    elif price_signal == "WAIT" and pct_change < 0:
        future_cost = qty * current_price * (1 + pct_change / 100)
        current_cost = qty * current_price
        return round(current_cost - future_cost, 2)
    return 0


def generate_recommendations(data_path="data/procurement_data.json") -> list:
    """
    Full recommendation pipeline with budget optimization.
    """
    with open(data_path) as f:
        data = json.load(f)

    ranked_suppliers = rank_suppliers_by_material(data["suppliers"])
    recommendations = []

    for material in data["materials"]:
        # --- Inventory ---
        inv_item = next((i for i in data["inventory"] if i["material"] == material), None)
        demand_hist = data["demand_history"].get(material, [])
        inv_risk = assess_inventory_risk(inv_item, demand_history=demand_hist) if inv_item else {}

        # --- Demand ---
        demand_vals = [d["demand"] for d in demand_hist]
        demand_result = forecast_demand(demand_vals, periods_ahead=6) if len(demand_vals) >= 8 else {}
        avg_forecast = (
            sum(demand_result.get("forecasts", [0])[:3]) / 3
            if demand_result.get("forecasts") else 0
        )

        # --- Price ---
        price_vals = [p["price"] for p in data["price_history"].get(material, [])]
        price_result = predict_prices(price_vals) if len(price_vals) >= 14 else {}
        price_signal = price_result.get("price_signal", "NEUTRAL")
        current_price = price_result.get("current_price", 0)
        pct_change = price_result.get("pct_change_expected", 0)
        volatility = price_result.get("volatility_pct", 0)

        # --- Suppliers ---
        top_suppliers = ranked_suppliers.get(material, [])
        best_supplier = top_suppliers[0] if top_suppliers else None
        alt_supplier = top_suppliers[1] if len(top_suppliers) > 1 else None

        # --- Decision ---
        urgency = inv_risk.get("urgency_score", 0)
        demand_trend = demand_result.get("trend", "Stable")
        inv_level = inv_risk.get("risk_level", "Adequate")
        decision_score = compute_decision_score(urgency, price_signal, demand_trend, inv_level, volatility)

        if urgency >= 90 or inv_level == "Critical":
            action = "ORDER IMMEDIATELY"
            priority = "Critical"
            timing = "Today"
        elif urgency >= 70 or inv_level == "Low":
            if price_signal == "WAIT" and inv_level != "Low":
                action = "ORDER SOON — Monitor price 2–3 days"
                timing = "This Week"
            else:
                action = "PLACE ORDER"
                timing = "This Week"
            priority = "High"
        elif price_signal == "BUY NOW" and demand_trend in ("Increasing", "Stable"):
            action = "OPPORTUNISTIC BUY — Favourable price window"
            priority = "Medium"
            timing = "Next 2 Weeks"
        elif price_signal == "WAIT" and inv_level == "Adequate":
            action = "HOLD — Wait for better price"
            priority = "Low"
            timing = "Within Month"
        else:
            action = "MONITOR — No urgent action needed"
            priority = "Low"
            timing = "Within Month"

        reorder_qty = compute_reorder_qty(
            inv_item, avg_forecast, lead_time_days=best_supplier.get("lead_time_days", 14) if best_supplier else 14,
            demand_history=demand_hist
        ) if inv_item else 0

        savings = estimate_savings(price_signal, pct_change, reorder_qty, current_price)

        # Split order suggestion (if alt supplier available and score close)
        split_suggestion = None
        if alt_supplier and best_supplier:
            score_diff = best_supplier["composite_score"] - alt_supplier["composite_score"]
            if score_diff < 5 and reorder_qty > 200:
                split_qty_primary = int(reorder_qty * 0.65)
                split_qty_alt = reorder_qty - split_qty_primary
                split_suggestion = {
                    "primary": {"supplier": best_supplier["name"], "qty": split_qty_primary},
                    "alternate": {"supplier": alt_supplier["name"], "qty": split_qty_alt},
                }

        recommendations.append({
            "material": material,
            "action": action,
            "priority": priority,
            "timing": timing,
            "recommended_qty": reorder_qty,
            "recommended_supplier": best_supplier["name"] if best_supplier else "N/A",
            "supplier_score": best_supplier["composite_score"] if best_supplier else 0,
            "supplier_grade": best_supplier["grade"] if best_supplier else "N/A",
            "alt_supplier": alt_supplier["name"] if alt_supplier else None,
            "split_order_suggestion": split_suggestion,
            "current_price": current_price,
            "price_signal": price_signal,
            "price_trend_pct": pct_change,
            "volatility_pct": volatility,
            "demand_trend": demand_trend,
            "demand_forecast_3mo": [round(f) for f in demand_result.get("forecasts", [])[:3]],
            "inventory_level": inv_level,
            "urgency_score": urgency,
            "decision_score": decision_score,
            "dynamic_days_of_supply": inv_risk.get("dynamic_days_of_supply", 0),
            "stockout_probability": inv_risk.get("stockout_probability_pct", 0),
            "eoq": inv_risk.get("eoq", 0),
            "safety_stock": inv_risk.get("safety_stock", 0),
            "estimated_savings": savings,
            "reason": f"Inventory: {inv_level} | Price: {price_signal} | Demand: {demand_trend}",
            "lead_time_days": best_supplier.get("lead_time_days", 14) if best_supplier else 14,
        })

    # Budget optimization pass
    recommendations = budget_constrained_optimization(recommendations)
    recommendations.sort(key=lambda x: x["decision_score"], reverse=True)
    return recommendations


def print_report(recs: list):
    print("\n" + "=" * 65)
    print("  SMART PROCUREMENT RECOMMENDATIONS")
    print("=" * 65)
    for r in recs:
        print(f"\n📦 {r['material']}  [{r['priority']}] — Decision Score: {r['decision_score']}")
        print(f"   Action   : {r['action']}")
        print(f"   Timing   : {r['timing']} | Lead Time: {r['lead_time_days']}d")
        print(f"   Order Qty: {r['recommended_qty']} kg (EOQ={r['eoq']}, Safety={r['safety_stock']})")
        print(f"   Supplier : {r['recommended_supplier']} ({r['supplier_grade']})")
        if r.get("split_order_suggestion"):
            s = r["split_order_suggestion"]
            print(f"   Split    : {s['primary']['supplier']} {s['primary']['qty']}kg + "
                  f"{s['alternate']['supplier']} {s['alternate']['qty']}kg")
        print(f"   Signals  : {r['reason']}")
        print(f"   Stockout : {r['stockout_probability']}% probability | {r['dynamic_days_of_supply']} days left")
        if r.get("estimated_savings") and r["estimated_savings"] > 0:
            print(f"   💰 Est. Savings: ₹{r['estimated_savings']:,.0f}")
        budget_status = r.get("budget_allocated", "N/A")
        print(f"   Budget   : {'✓ Allocated' if budget_status is True else '⚠ Partial' if budget_status == 'Partial' else '✗ Over Budget'}")


if __name__ == "__main__":
    recs = generate_recommendations()
    print_report(recs)
