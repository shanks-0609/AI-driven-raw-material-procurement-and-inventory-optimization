"""
Complex Simulated Dataset Generator for ProcureAI
Generates realistic, non-linear supply chain data with:
 - Market shocks, supply disruptions, geopolitical events
 - Multi-cycle seasonality (monthly + quarterly + annual)
 - Mean-reverting price processes (Ornstein-Uhlenbeck)
 - Auto-correlated demand with regime changes
 - Realistic supplier degradation/improvement patterns
"""

import json
import random
import math
from datetime import datetime, timedelta

random.seed(2024)

MATERIALS = [
    "Steel Coils", "Aluminum Sheets", "Copper Wire",
    "Plastic Pellets", "Carbon Fiber", "Silicon Wafers",
    "Titanium Alloy", "Nickel Powder", "Graphite Electrodes", "Rare Earth Magnets"
]

SUPPLIERS = {
    "Steel Coils":          ["SteelMart Inc", "MetalPro Ltd", "AlloySource Co", "IronBridge Supply"],
    "Aluminum Sheets":      ["AluminumFirst", "LightMetal Corp", "AlloySource Co", "Nordic Metals"],
    "Copper Wire":          ["CopperKing", "WireWorks Ltd", "ElectraMat", "RedMetal GmbH"],
    "Plastic Pellets":      ["PolyBase Inc", "PlastiCore", "ChemSupply", "SynPoly"],
    "Carbon Fiber":         ["FiberTech", "CompositesPro", "AeroMat", "ToraySource"],
    "Silicon Wafers":       ["SiliconEdge", "WaferTech", "MicroMat", "ChipSource Asia"],
    "Titanium Alloy":       ["TitanWorks", "AeroAlloys", "PrecisionMetal Co"],
    "Nickel Powder":        ["NickelBase", "MetalPowders Ltd", "BatteryMat Supply"],
    "Graphite Electrodes":  ["GraphCo", "CarbonTech", "ElectrodeMax"],
    "Rare Earth Magnets":   ["MagnetCore", "REESource", "AsiaRareMetal"],
}

BASE_PRICES = {
    "Steel Coils":         820,
    "Aluminum Sheets":     2400,
    "Copper Wire":         8500,
    "Plastic Pellets":     1200,
    "Carbon Fiber":        22000,
    "Silicon Wafers":      15000,
    "Titanium Alloy":      35000,
    "Nickel Powder":       18000,
    "Graphite Electrodes": 4500,
    "Rare Earth Magnets":  85000,
}

BASE_DEMANDS = {
    "Steel Coils":         900,
    "Aluminum Sheets":     600,
    "Copper Wire":         450,
    "Plastic Pellets":     1200,
    "Carbon Fiber":        180,
    "Silicon Wafers":      320,
    "Titanium Alloy":      90,
    "Nickel Powder":       240,
    "Graphite Electrodes": 360,
    "Rare Earth Magnets":  60,
}

# Market shock events (month_index: description, price_multiplier)
SHOCK_EVENTS = {
    3:  ("Supply chain disruption", 1.18),
    7:  ("Global demand surge", 1.12),
    12: ("Trade tariff introduced", 1.22),
    17: ("Port strike resolved", 0.92),
    22: ("Energy price spike", 1.15),
    27: ("New supply source opened", 0.88),
    31: ("Geopolitical tension", 1.25),
    35: ("Market correction", 0.85),
}

def ornstein_uhlenbeck_path(n, mu, theta=0.15, sigma=0.08, x0=None):
    """
    Ornstein-Uhlenbeck mean-reverting process for realistic price simulation.
    theta: speed of mean reversion
    sigma: volatility
    mu: long-term mean (normalized to 1.0)
    """
    x0 = x0 or 1.0
    path = [x0]
    for _ in range(n - 1):
        dx = theta * (mu - path[-1]) + sigma * (random.gauss(0, 1))
        path.append(max(0.5, path[-1] + dx))
    return path


def generate_price_history(material, months=36):
    """
    Generate complex price history with:
    - OU mean-reverting base process
    - Multi-frequency seasonality
    - Market shocks
    - Structural breaks (regime changes)
    """
    base = BASE_PRICES[material]
    # OU process gives non-linear realistic fluctuation
    ou_path = ornstein_uhlenbeck_path(months, mu=1.0, theta=0.12, sigma=0.06)
    
    # Different materials have different shock sensitivities
    shock_sensitivity = {
        "Steel Coils": 0.9, "Aluminum Sheets": 1.0, "Copper Wire": 1.2,
        "Plastic Pellets": 0.7, "Carbon Fiber": 0.5, "Silicon Wafers": 1.3,
        "Titanium Alloy": 0.6, "Nickel Powder": 1.4, "Graphite Electrodes": 1.1,
        "Rare Earth Magnets": 1.6,
    }.get(material, 1.0)

    prices = []
    today = datetime.today()
    regime_multiplier = 1.0

    for i in range(months, 0, -1):
        date = today - timedelta(days=30 * i)
        month_idx = months - i

        # Multi-cycle seasonality
        annual_season = 1 + 0.07 * math.sin(2 * math.pi * date.month / 12)
        quarterly_season = 1 + 0.03 * math.cos(2 * math.pi * date.month / 3)
        season = annual_season * quarterly_season

        # Long-term upward trend (inflation)
        trend = 1 + 0.0018 * month_idx

        # OU process noise
        ou_factor = ou_path[month_idx]

        # Market shocks
        shock = 1.0
        if month_idx in SHOCK_EVENTS:
            _, mult = SHOCK_EVENTS[month_idx]
            # Apply shock partially (fades over next 2 months)
            shock = 1 + (mult - 1) * shock_sensitivity
            regime_multiplier = 0.6 + 0.4 * mult  # regime shift

        # Regime decay
        regime_multiplier = 0.95 * regime_multiplier + 0.05 * 1.0

        price = round(base * trend * season * ou_factor * shock * regime_multiplier, 2)
        prices.append({"date": date.strftime("%Y-%m"), "price": price})

    return prices


def generate_demand_history(material, months=36):
    """
    Generate demand with:
    - Auto-correlated AR(2) process
    - Regime changes (production ramp-ups, shutdowns)
    - Seasonal patterns
    - Random demand spikes
    """
    base = BASE_DEMANDS[material]
    demands = []
    today = datetime.today()

    prev1, prev2 = base, base
    regime = 1.0  # production regime multiplier

    for i in range(months, 0, -1):
        date = today - timedelta(days=30 * i)
        month_idx = months - i

        # AR(2) auto-correlation
        ar_component = 0.6 * prev1 + 0.2 * prev2 + random.gauss(0, base * 0.05)

        # Multi-cycle seasonality
        annual_season = 1 + 0.15 * math.sin(2 * math.pi * (date.month - 2) / 12)
        quarterly_dip = 1 - 0.04 * math.cos(2 * math.pi * date.month / 3)
        season = annual_season * quarterly_dip

        # Regime change every ~10 months
        if month_idx % 11 == 0:
            regime = random.uniform(0.80, 1.25)

        # Occasional demand spikes (new contracts, production surge)
        spike = 1.0
        if random.random() < 0.07:  # 7% chance of spike
            spike = random.uniform(1.3, 1.8)
        elif random.random() < 0.05:  # 5% chance of dip
            spike = random.uniform(0.5, 0.75)

        # Long-term growth trend
        trend = 1 + 0.002 * month_idx

        demand = max(20, int(ar_component * season * regime * spike * trend))
        demands.append({"date": date.strftime("%Y-%m"), "demand": demand})
        prev2 = prev1
        prev1 = demand

    return demands


def generate_supplier_data():
    """
    Generate suppliers with realistic KPI variations:
    - Some suppliers have consistent performance
    - Some are improving or degrading over time
    - Risk flags based on multiple thresholds
    """
    suppliers = []
    for material, sup_list in SUPPLIERS.items():
        for idx, sup in enumerate(sup_list):
            # First supplier in list is generally better
            quality_bias = max(0, 0.08 * (len(sup_list) - 1 - idx))

            on_time = round(random.uniform(68, 99) + quality_bias * 5, 1)
            quality = round(random.uniform(75, 99) + quality_bias * 4, 1)
            price_comp = round(random.uniform(72, 98), 1)
            resp = round(random.uniform(68, 98) + quality_bias * 3, 1)

            # Cap at 99
            on_time = min(99, on_time)
            quality = min(99, quality)
            price_comp = min(99, price_comp)
            resp = min(99, resp)

            score = round(on_time * 0.35 + quality * 0.30 + price_comp * 0.20 + resp * 0.15, 1)
            lead_time = random.randint(4, 28)

            # Certifications
            certs = []
            if quality > 90: certs.append("ISO 9001")
            if on_time > 92: certs.append("Lean Certified")
            if random.random() > 0.6: certs.append("ISO 14001")

            suppliers.append({
                "name": sup,
                "material": material,
                "on_time_delivery": on_time,
                "quality_score": quality,
                "price_competitiveness": price_comp,
                "responsiveness": resp,
                "overall_score": score,
                "lead_time_days": lead_time,
                "min_order_qty": random.choice([100, 200, 500, 1000]),
                "reliability": "High" if score > 88 else "Medium" if score > 78 else "Low",
                "certifications": certs,
                "country": random.choice(["India", "China", "Germany", "USA", "Japan", "South Korea"]),
                "years_as_vendor": random.randint(1, 15),
                "defect_rate_pct": round(random.uniform(0.2, 4.5), 2),
                "fill_rate_pct": round(random.uniform(85, 99.5), 1),
            })
    return suppliers


def generate_inventory_data():
    """Generate complex inventory with varying stock situations."""
    inventory = []
    for material in MATERIALS:
        max_stock = random.randint(3000, 12000)
        # Deliberately create diverse risk situations
        risk_scenario = random.choice(["critical", "low", "adequate", "adequate", "overstocked"])
        if risk_scenario == "critical":
            current = random.randint(int(max_stock * 0.05), int(max_stock * 0.14))
        elif risk_scenario == "low":
            current = random.randint(int(max_stock * 0.15), int(max_stock * 0.29))
        elif risk_scenario == "overstocked":
            current = random.randint(int(max_stock * 0.86), int(max_stock * 0.98))
        else:
            current = random.randint(int(max_stock * 0.35), int(max_stock * 0.75))

        reorder_point = int(max_stock * 0.25)
        daily_usage = round(BASE_DEMANDS[material] / 30, 1)
        days_of_supply = round(current / max(daily_usage, 1), 0)

        inventory.append({
            "material": material,
            "current_stock": current,
            "max_capacity": max_stock,
            "reorder_point": reorder_point,
            "unit": "kg",
            "daily_usage_rate": daily_usage,
            "days_of_supply": int(days_of_supply),
            "holding_cost_per_unit": round(random.uniform(0.5, 3.5), 2),
            "ordering_cost": random.randint(200, 1500),
            "last_updated": datetime.today().strftime("%Y-%m-%d"),
        })
    return inventory


def build_full_dataset():
    dataset = {
        "generated_at": datetime.today().isoformat(),
        "version": "2.0",
        "materials": MATERIALS,
        "price_history": {m: generate_price_history(m, months=36) for m in MATERIALS},
        "demand_history": {m: generate_demand_history(m, months=36) for m in MATERIALS},
        "suppliers": generate_supplier_data(),
        "inventory": generate_inventory_data(),
        "shock_events": {str(k): v[0] for k, v in SHOCK_EVENTS.items()},
    }
    return dataset


if __name__ == "__main__":
    import os
    data = build_full_dataset()
    os.makedirs("data", exist_ok=True)
    with open("data/procurement_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ Complex dataset generated: {len(data['materials'])} materials, {len(data['suppliers'])} suppliers, 36 months history")
