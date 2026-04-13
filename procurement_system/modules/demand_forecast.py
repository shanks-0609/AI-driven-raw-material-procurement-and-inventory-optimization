"""
Module 1: Advanced Demand Forecasting
Implements:
  - Holt-Winters Triple Exponential Smoothing (additive seasonality)
  - AR(p) Auto-Regression with automatic order selection (AIC)
  - Ensemble: weighted average of both models
  - Proper prediction intervals via bootstrap residuals
"""

import json
import math
import random


# ─────────────────────────────────────────────
# Holt-Winters Triple Exponential Smoothing
# ─────────────────────────────────────────────

def holt_winters(series, season_period=12, alpha=None, beta=None, gamma=None, periods_ahead=6):
    """
    Triple Exponential Smoothing (Holt-Winters) with additive seasonality.
    Automatically optimizes alpha, beta, gamma via grid search if not provided.
    """
    n = len(series)
    if n < season_period * 2:
        return None  # insufficient data

    def _run(series, a, b, g, period, ahead):
        """Run Holt-Winters and return fitted values + forecasts."""
        # Initialization
        level = sum(series[:period]) / period
        trend = (sum(series[period:2*period]) - sum(series[:period])) / (period ** 2)
        seasonal = [series[i] - level for i in range(period)]

        levels, trends, seasonals = [level], [trend], seasonal[:]
        fitted = []

        for i in range(1, n):
            s_idx = i - period
            prev_s = seasonals[s_idx] if s_idx >= 0 else 0
            new_level = a * (series[i] - prev_s) + (1 - a) * (levels[-1] + trends[-1])
            new_trend = b * (new_level - levels[-1]) + (1 - b) * trends[-1]
            new_seasonal = g * (series[i] - new_level) + (1 - g) * prev_s
            levels.append(new_level)
            trends.append(new_trend)
            seasonals.append(new_seasonal)
            fitted.append(new_level + new_trend + prev_s)

        forecasts = []
        for h in range(1, ahead + 1):
            s_idx = n - period + ((h - 1) % period)
            forecast = levels[-1] + h * trends[-1] + seasonals[s_idx]
            forecasts.append(max(0, round(forecast, 1)))
        return fitted, forecasts

    def _sse(a, b, g):
        try:
            fitted, _ = _run(series, a, b, g, season_period, 0)
            return sum((series[i+1] - fitted[i])**2 for i in range(len(fitted)))
        except:
            return float('inf')

    # Grid search for optimal parameters
    if alpha is None or beta is None or gamma is None:
        best_sse = float('inf')
        best_params = (0.3, 0.1, 0.2)
        for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
            for b in [0.05, 0.1, 0.15, 0.2, 0.3]:
                for g in [0.1, 0.2, 0.3, 0.4, 0.6]:
                    sse = _sse(a, b, g)
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (a, b, g)
        alpha, beta, gamma = best_params

    fitted, forecasts = _run(series, alpha, beta, gamma, season_period, periods_ahead)
    residuals = [series[i+1] - fitted[i] for i in range(len(fitted))]
    return {
        "forecasts": forecasts,
        "fitted": [round(f, 1) for f in fitted],
        "residuals": residuals,
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma},
    }


# ─────────────────────────────────────────────
# Auto-Regressive (AR) Model
# ─────────────────────────────────────────────

def fit_ar(series, max_p=6):
    """
    Fit AR(p) model using OLS, select order by AIC.
    Returns coefficients, intercept, order, and in-sample fitted values.
    """
    n = len(series)
    best_aic = float('inf')
    best_result = None

    for p in range(1, min(max_p + 1, n // 3)):
        # Build design matrix
        X, y = [], []
        for i in range(p, n):
            X.append([series[i - j - 1] for j in range(p)])
            y.append(series[i])

        # OLS: coefficients via normal equations
        m = len(y)
        # Add intercept column
        X_mat = [[1] + row for row in X]
        k = len(X_mat[0])

        # Xt @ X with Ridge regularization to avoid singular matrices
        XtX = [[sum(X_mat[r][c1] * X_mat[r][c2] for r in range(m)) for c2 in range(k)] for c1 in range(k)]
        # Add Ridge penalty to diagonal
        for i in range(k):
            XtX[i][i] += 1e-6
            
        Xty = [sum(X_mat[r][c] * y[r] for r in range(m)) for c in range(k)]

        try:
            coeffs = _solve_linear(XtX, Xty)
        except:
            continue

        # Fitted values
        fitted = [sum(coeffs[j] * X_mat[i][j] for j in range(k)) for i in range(m)]
        residuals = [y[i] - fitted[i] for i in range(m)]
        sse = sum(r ** 2 for r in residuals)
        sigma2 = sse / max(m - k, 1)

        # AIC = n*ln(sse/n) + 2k
        try:
            aic = m * math.log(sse / m) + 2 * k
        except:
            continue

        if aic < best_aic:
            best_aic = aic
            best_result = {
                "order": p,
                "coeffs": coeffs,
                "fitted": fitted,
                "residuals": residuals,
                "sse": sse,
                "sigma2": sigma2,
                "aic": round(aic, 2),
            }

    return best_result


def _solve_linear(A, b):
    """Gaussian elimination for small systems."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-12:
            raise ValueError("Singular matrix")
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            M[row] = [M[row][j] - factor * M[col][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def ar_forecast(series, ar_result, periods_ahead=6):
    """Generate multi-step forecasts from AR model."""
    if ar_result is None:
        return None
    p = ar_result["order"]
    coeffs = ar_result["coeffs"]
    history = list(series)
    forecasts = []
    for _ in range(periods_ahead):
        x_row = [1] + [history[-(j + 1)] for j in range(p)]
        pred = sum(coeffs[j] * x_row[j] for j in range(len(coeffs)))
        pred = max(0, pred)
        forecasts.append(round(pred, 1))
        history.append(pred)
    return forecasts


# ─────────────────────────────────────────────
# Ensemble + Prediction Intervals
# ─────────────────────────────────────────────

def bootstrap_prediction_intervals(residuals, forecasts, n_boot=200, confidence=0.90):
    """
    Bootstrap residuals to generate prediction intervals.
    """
    boot_forecasts = []
    for _ in range(n_boot):
        sample = [forecasts[i] + random.choice(residuals) for i in range(len(forecasts))]
        boot_forecasts.append(sample)

    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100

    ci_lower, ci_upper = [], []
    for i in range(len(forecasts)):
        col = sorted(b[i] for b in boot_forecasts)
        lo_idx = int(lower_pct / 100 * n_boot)
        hi_idx = int(upper_pct / 100 * n_boot)
        ci_lower.append(max(0, round(col[lo_idx], 1)))
        ci_upper.append(round(col[min(hi_idx, n_boot - 1)], 1))

    return ci_lower, ci_upper


def forecast_demand(history_values, periods_ahead=6):
    """
    Main demand forecasting function.
    Ensemble of:
      1. Holt-Winters (triple exponential smoothing)
      2. AR(p) auto-regression
    """
    if not history_values or len(history_values) < 8:
        return {"error": "Insufficient data for forecasting"}

    # --- Holt-Winters ---
    hw_result = holt_winters(history_values, season_period=12, periods_ahead=periods_ahead)
    hw_forecasts = hw_result["forecasts"] if hw_result else None

    # --- AR Model ---
    ar_result = fit_ar(history_values)
    ar_forecasts = ar_forecast(history_values, ar_result, periods_ahead) if ar_result else None

    # --- Ensemble (weighted average) ---
    if hw_forecasts and ar_forecasts:
        # Weight HW more (better for seasonal), AR for short-term
        ensemble = [round(0.60 * hw_forecasts[i] + 0.40 * ar_forecasts[i], 1) for i in range(periods_ahead)]
        method = "Ensemble: Holt-Winters (60%) + AR({p}) (40%)".format(p=ar_result["order"])
        residuals = hw_result["residuals"][-24:] if hw_result.get("residuals") else [0]
    elif hw_forecasts:
        ensemble = hw_forecasts
        method = "Holt-Winters Triple Exponential Smoothing"
        residuals = hw_result.get("residuals", [0])[-24:]
    elif ar_forecasts:
        ensemble = ar_forecasts
        method = "AR({p}) Auto-Regression".format(p=ar_result["order"])
        residuals = ar_result.get("residuals", [0])[-24:]
    else:
        return {"error": "All models failed"}

    ci_lower, ci_upper = bootstrap_prediction_intervals(residuals, ensemble)

    avg_recent = sum(history_values[-3:]) / 3
    avg_forecast = sum(ensemble) / len(ensemble)
    trend_label = (
        "Increasing" if avg_forecast > avg_recent * 1.04 else
        "Decreasing" if avg_forecast < avg_recent * 0.96 else "Stable"
    )

    return {
        "forecasts": ensemble,
        "hw_forecasts": hw_forecasts,
        "ar_forecasts": ar_forecasts,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "trend": trend_label,
        "method": method,
        "ar_order": ar_result["order"] if ar_result else None,
        "ar_aic": ar_result["aic"] if ar_result else None,
        "hw_params": hw_result["params"] if hw_result else None,
    }


def run_all_forecasts(data_path="data/procurement_data.json"):
    """Run demand forecasting for all materials."""
    with open(data_path) as f:
        data = json.load(f)

    results = {}
    for material, history in data["demand_history"].items():
        values = [h["demand"] for h in history]
        dates = [h["date"] for h in history]
        result = forecast_demand(values, periods_ahead=6)
        result["history_dates"] = dates
        result["history_values"] = values
        results[material] = result
        if "error" not in result:
            avg3 = round(sum(result["forecasts"][:3]) / 3, 0)
            print(f"  [Demand] {material}: Trend={result.get('trend')} | "
                  f"Method={result['method'][:30]}... | Next 3mo avg={avg3}")
        else:
            print(f"  [Demand] {material}: {result['error']}")

    return results


if __name__ == "__main__":
    results = run_all_forecasts()
    print(f"\n✓ Forecasting complete for {len(results)} materials.")
