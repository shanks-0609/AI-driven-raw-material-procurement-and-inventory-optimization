"""
Module 4: Inventory Risk Monitoring
Enhanced with:
  - Dynamic days-of-supply based on forecast demand
  - Full EOQ (Economic Order Quantity) calculation
  - Safety stock using demand variability
  - Stockout probability estimation
  - Carrying cost analysis
"""

import json
import math


RISK_THRESHOLDS = {
    "critical_pct": 0.15,
    "low_pct":      0.30,
    "excess_pct":   0.85,
}


def compute_eoq(annual_demand, ordering_cost, holding_cost_per_unit):
    """
    Economic Order Quantity: Q* = sqrt(2DS/H)
    D = annual demand, S = ordering cost, H = holding cost per unit per year
    """
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    return round(math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit))


def compute_safety_stock(demand_values, lead_time_days, service_level=0.95):
    """
    Safety stock = Z * σ_demand * sqrt(lead_time)
    Z for 95% service level ≈ 1.645
    """
    if len(demand_values) < 3:
        return 0
    mean = sum(demand_values) / len(demand_values)
    variance = sum((d - mean)**2 for d in demand_values) / len(demand_values)
    std = math.sqrt(variance)
    z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(service_level, 1.645)
    lead_time_months = lead_time_days / 30.0
    safety_stock = z * std * math.sqrt(lead_time_months)
    return round(safety_stock)


def stockout_probability(current_stock, daily_usage, lead_time_days, std_daily_usage):
    """
    Estimate stockout probability during replenishment lead time.
    P(stockout) = P(demand > stock) using normal approximation.
    """
    if daily_usage <= 0:
        return 0.0
    demand_during_lt = daily_usage * lead_time_days
    demand_std_lt = std_daily_usage * math.sqrt(lead_time_days) if std_daily_usage > 0 else 1
    z = (current_stock - demand_during_lt) / demand_std_lt
    # Standard normal CDF approximation
    prob_no_stockout = _normal_cdf(z)
    return round((1 - prob_no_stockout) * 100, 1)


def _normal_cdf(z):
    """Abramowitz & Stegun approximation of normal CDF."""
    t = 1 / (1 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z**2) * poly
    return cdf if z >= 0 else 1 - cdf


def assess_inventory_risk(item: dict, demand_history: list = None, lead_time_days: int = 14) -> dict:
    """
    Comprehensive inventory risk assessment.
    """
    stock = item["current_stock"]
    capacity = item["max_capacity"]
    reorder = item["reorder_point"]
    pct = stock / capacity if capacity > 0 else 0

    # Daily usage from item or history
    daily_usage = item.get("daily_usage_rate", 0)
    if demand_history and len(demand_history) >= 3:
        monthly_vals = [h["demand"] for h in demand_history[-6:]]
        avg_monthly = sum(monthly_vals) / len(monthly_vals)
        daily_usage = round(avg_monthly / 30, 2)
        std_monthly = math.sqrt(sum((v - avg_monthly)**2 for v in monthly_vals) / len(monthly_vals))
        std_daily = std_monthly / 30
    else:
        std_daily = daily_usage * 0.15  # 15% coefficient of variation

    # Dynamic days of supply
    dynamic_dos = round(stock / max(daily_usage, 0.1)) if daily_usage > 0 else item.get("days_of_supply", 30)

    # Safety stock
    demand_vals = [h["demand"] for h in demand_history] if demand_history else []
    safety_stock_qty = compute_safety_stock(demand_vals, lead_time_days) if demand_vals else 0

    # EOQ
    ordering_cost = item.get("ordering_cost", 500)
    holding_cost = item.get("holding_cost_per_unit", 1.5)
    annual_demand = daily_usage * 365
    eoq = compute_eoq(annual_demand, ordering_cost, holding_cost)

    # Stockout probability
    stockout_prob = stockout_probability(stock, daily_usage, lead_time_days, std_daily)

    # Risk level
    if pct < RISK_THRESHOLDS["critical_pct"] or stockout_prob > 60:
        risk_level = "Critical"
        risk_color = "#ef4444"
        action = "URGENT: Place order immediately"
        urgency_score = 90 + min(9, int(stockout_prob / 10))
    elif stock < reorder or stockout_prob > 30:
        risk_level = "Low"
        risk_color = "#f97316"
        action = "Order this week — below reorder point"
        urgency_score = 65 + min(14, int(stockout_prob / 2))
    elif pct > RISK_THRESHOLDS["excess_pct"]:
        risk_level = "Overstocked"
        risk_color = "#8b5cf6"
        action = "Hold — review demand before ordering"
        urgency_score = 5
    else:
        risk_level = "Adequate"
        risk_color = "#22c55e"
        action = "Monitor — no immediate action required"
        urgency_score = max(10, int(30 - dynamic_dos / 2))

    carrying_cost_monthly = stock * holding_cost / 12

    return {
        **item,
        "stock_pct": round(pct * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommended_action": action,
        "urgency_score": urgency_score,
        "dynamic_days_of_supply": dynamic_dos,
        "safety_stock": safety_stock_qty,
        "eoq": eoq,
        "stockout_probability_pct": stockout_prob,
        "stockout_alert": dynamic_dos < 14 or stockout_prob > 40,
        "carrying_cost_monthly": round(carrying_cost_monthly, 2),
        "daily_usage": daily_usage,
    }


def compute_reorder_qty(item: dict, forecast_demand_monthly: float = None,
                        lead_time_days: int = 14, demand_history: list = None) -> int:
    """
    EOQ-based reorder quantity with safety stock.
    """
    capacity = item["max_capacity"]
    current = item["current_stock"]
    holding_cost = item.get("holding_cost_per_unit", 1.5)
    ordering_cost = item.get("ordering_cost", 500)

    if forecast_demand_monthly and forecast_demand_monthly > 0:
        annual_demand = forecast_demand_monthly * 12
        eoq = compute_eoq(annual_demand, ordering_cost, holding_cost)
        demand_vals = [h["demand"] for h in demand_history] if demand_history else []
        ss = compute_safety_stock(demand_vals, lead_time_days) if demand_vals else 0
        target = eoq + ss
    else:
        target = int(capacity * 0.70)

    reorder_qty = max(0, int(target - current))
    # Cap at remaining capacity
    reorder_qty = min(reorder_qty, capacity - current)
    return reorder_qty


def run_inventory_analysis(data_path="data/procurement_data.json") -> list:
    """Full inventory risk assessment."""
    with open(data_path) as f:
        data = json.load(f)

    inventory = data["inventory"]
    demand_history = data.get("demand_history", {})
    results = []

    for item in inventory:
        mat = item["material"]
        hist = demand_history.get(mat, [])
        result = assess_inventory_risk(item, demand_history=hist)
        results.append(result)

    results.sort(key=lambda x: x["urgency_score"], reverse=True)

    print("\n=== INVENTORY RISK REPORT ===")
    for item in results:
        dos = item["dynamic_days_of_supply"]
        prob = item["stockout_probability_pct"]
        alert = f" 🚨 Stockout risk {prob}%" if item["stockout_alert"] else ""
        print(f"  {item['material']}: {item['risk_level']} | "
              f"{item['stock_pct']}% stocked | {dos} days supply | EOQ={item['eoq']}{alert}")

    return results


if __name__ == "__main__":
    run_inventory_analysis()
