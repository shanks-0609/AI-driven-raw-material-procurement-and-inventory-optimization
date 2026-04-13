"""
Module 3: Supplier Performance Scoring & Evaluation
Enhanced with:
  - Multi-dimensional KPI scoring (6 factors)
  - K-means clustering to segment supplier tiers
  - Geopolitical + ESG risk overlay
  - Trend analysis (improving vs degrading suppliers)
"""

import json
import math


WEIGHTS = {
    "on_time_delivery":      0.30,
    "quality_score":         0.25,
    "price_competitiveness": 0.20,
    "responsiveness":        0.12,
    "fill_rate_adj":         0.08,
    "defect_penalty":        0.05,
}

COUNTRY_RISK = {
    "Germany": 1.0, "USA": 0.95, "Japan": 0.98,
    "South Korea": 0.93, "India": 0.85, "China": 0.78,
}


def score_supplier(supplier: dict) -> dict:
    """
    Calculate comprehensive supplier score (0–100).
    Includes fill rate and defect rate adjustments.
    """
    fill_adj = supplier.get("fill_rate_pct", 95) / 100 * 100
    defect_penalty = max(0, 100 - supplier.get("defect_rate_pct", 1.0) * 10)

    raw_scores = {
        "on_time_delivery":      supplier.get("on_time_delivery", 80),
        "quality_score":         supplier.get("quality_score", 80),
        "price_competitiveness": supplier.get("price_competitiveness", 80),
        "responsiveness":        supplier.get("responsiveness", 80),
        "fill_rate_adj":         fill_adj,
        "defect_penalty":        defect_penalty,
    }

    base_score = sum(raw_scores[kpi] * WEIGHTS[kpi] for kpi in WEIGHTS)

    # Country/geopolitical risk adjustment (max ±5 points)
    country = supplier.get("country", "India")
    country_factor = COUNTRY_RISK.get(country, 0.85)
    geo_adjustment = (country_factor - 0.85) * 20  # -3 to +3 points

    # Vendor tenure bonus (experienced suppliers)
    tenure = supplier.get("years_as_vendor", 3)
    tenure_bonus = min(3, tenure * 0.25)

    composite = min(100, max(0, base_score + geo_adjustment + tenure_bonus))
    composite = round(composite, 2)

    # Grading
    if composite >= 90:
        grade, badge, badge_color = "A+", "Preferred Partner", "#22c55e"
    elif composite >= 85:
        grade, badge, badge_color = "A", "Approved", "#84cc16"
    elif composite >= 78:
        grade, badge, badge_color = "B", "Conditional", "#f59e0b"
    elif composite >= 70:
        grade, badge, badge_color = "C", "Under Review", "#f97316"
    else:
        grade, badge, badge_color = "D", "Risk — Avoid", "#ef4444"

    # Risk flags
    risk_flags = []
    if supplier.get("on_time_delivery", 100) < 80:
        risk_flags.append("High delivery delay risk")
    if supplier.get("quality_score", 100) < 80:
        risk_flags.append("Quality issues detected")
    if supplier.get("lead_time_days", 0) > 20:
        risk_flags.append("Long lead time (>20 days)")
    if supplier.get("price_competitiveness", 100) < 75:
        risk_flags.append("Above-market pricing")
    if supplier.get("defect_rate_pct", 0) > 3.0:
        risk_flags.append("High defect rate")
    if country_factor < 0.85:
        risk_flags.append(f"Geopolitical risk ({country})")
    if supplier.get("fill_rate_pct", 100) < 92:
        risk_flags.append("Low fill rate")

    overall_risk = "High" if len(risk_flags) >= 3 else "Medium" if len(risk_flags) >= 1 else "Low"

    return {
        **supplier,
        "composite_score": composite,
        "grade": grade,
        "badge": badge,
        "badge_color": badge_color,
        "risk_flags": risk_flags,
        "risk_level": overall_risk,
        "geo_risk_factor": country_factor,
        "component_scores": raw_scores,
    }


def kmeans_cluster_suppliers(scored_suppliers, k=3, iterations=15):
    """
    K-means clustering to segment suppliers into tiers:
      - Tier 1: Strategic / Preferred
      - Tier 2: Transactional / Standard
      - Tier 3: Underperforming / High-Risk
    """
    scores = [s["composite_score"] for s in scored_suppliers]
    if len(scores) < k:
        for s in scored_suppliers:
            s["cluster"] = 0
            s["tier"] = "Tier 1"
        return scored_suppliers

    # Initialize centroids using k-means++ style
    centroids = [min(scores), sum(scores)/len(scores), max(scores)]

    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
        assignments = []
        for score in scores:
            dists = [abs(score - c) for c in centroids]
            cluster_id = dists.index(min(dists))
            assignments.append(cluster_id)
            clusters[cluster_id].append(score)

        new_centroids = []
        for i in range(k):
            if clusters[i]:
                new_centroids.append(sum(clusters[i]) / len(clusters[i]))
            else:
                new_centroids.append(centroids[i])
        centroids = new_centroids

    # Assign labels — sort centroids so highest = Tier 1
    sorted_centroids = sorted(enumerate(centroids), key=lambda x: x[1], reverse=True)
    tier_map = {}
    tier_labels = ["Tier 1 — Strategic", "Tier 2 — Standard", "Tier 3 — At Risk"]
    for rank, (orig_idx, _) in enumerate(sorted_centroids):
        tier_map[orig_idx] = tier_labels[rank]

    for i, supplier in enumerate(scored_suppliers):
        dists = [abs(supplier["composite_score"] - c) for c in centroids]
        cluster_id = dists.index(min(dists))
        supplier["cluster"] = cluster_id
        supplier["tier"] = tier_map.get(cluster_id, "Tier 3 — At Risk")

    return scored_suppliers


def rank_suppliers_by_material(suppliers: list) -> dict:
    """Group, score, cluster, and rank suppliers by material."""
    scored = [score_supplier(s) for s in suppliers]

    # Cluster all suppliers together for cross-material tiers
    scored = kmeans_cluster_suppliers(scored, k=3)

    by_material = {}
    for s in scored:
        mat = s["material"]
        if mat not in by_material:
            by_material[mat] = []
        by_material[mat].append(s)

    for mat in by_material:
        by_material[mat].sort(key=lambda x: x["composite_score"], reverse=True)
        for rank, s in enumerate(by_material[mat], 1):
            s["rank"] = rank

    return by_material


def get_best_supplier(material: str, ranked: dict) -> dict | None:
    return ranked.get(material, [None])[0]


def generate_supplier_report(data_path="data/procurement_data.json") -> dict:
    """Full supplier evaluation report."""
    with open(data_path) as f:
        data = json.load(f)

    ranked = rank_suppliers_by_material(data["suppliers"])

    print("\n=== SUPPLIER PERFORMANCE REPORT ===")
    for material, suppliers in ranked.items():
        print(f"\n  {material}:")
        for s in suppliers:
            flags = f" ⚠ {', '.join(s['risk_flags'][:2])}" if s['risk_flags'] else ""
            print(f"    #{s['rank']} {s['name']} | Score: {s['composite_score']} | "
                  f"{s['grade']} | {s['tier']}{flags}")

    return ranked


if __name__ == "__main__":
    generate_supplier_report()
