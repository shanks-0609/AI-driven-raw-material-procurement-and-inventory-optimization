"""
Module 2: Advanced Raw Material Price Prediction
Implements:
  - ARIMA(p,d,q)-style model (differencing + AR + MA)
  - Gradient Boosting analog using stagewise additive trees (pure Python)
  - Ensemble of both models
  - Technical indicators: RSI, MACD, Bollinger Bands
  - Volatility-adjusted prediction intervals
"""

import json
import math
import random


# ─────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────

def ema(values, span):
    """Exponential Moving Average."""
    alpha = 2.0 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(alpha * v + (1 - alpha) * result[-1])
    return [round(e, 4) for e in result]


def macd(values, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    if len(values) < slow + signal:
        return None, None, None
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(values))]
    signal_line = ema(macd_line, signal)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(macd_line))]
    return (
        round(macd_line[-1], 4),
        round(signal_line[-1], 4),
        round(histogram[-1], 4)
    )


def rsi(values, period=14):
    """Relative Strength Index."""
    if len(values) < period + 1:
        return 50.0
    gains = [max(0, values[i] - values[i-1]) for i in range(1, len(values))]
    losses = [max(0, values[i-1] - values[i]) for i in range(1, len(values))]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss < 1e-9:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def bollinger_bands(values, window=20, num_std=2):
    """Bollinger Bands: upper, middle, lower."""
    if len(values) < window:
        return None, None, None
    recent = values[-window:]
    mean = sum(recent) / window
    std = math.sqrt(sum((v - mean)**2 for v in recent) / window)
    return round(mean + num_std * std, 2), round(mean, 2), round(mean - num_std * std, 2)


def compute_volatility(values, window=6):
    """Annualized volatility proxy."""
    if len(values) < window:
        return 0.0
    recent = values[-window:]
    mean = sum(recent) / window
    variance = sum((v - mean)**2 for v in recent) / window
    return round(math.sqrt(variance) / mean * 100, 2)  # percent


# ─────────────────────────────────────────────
# ARIMA-style model (ARMA on differenced series)
# ─────────────────────────────────────────────

def difference(series, d=1):
    """Apply d-th order differencing."""
    diff = series[:]
    for _ in range(d):
        diff = [diff[i] - diff[i-1] for i in range(1, len(diff))]
    return diff


def undifference(original, forecasts, d=1):
    """Invert differencing to get price-level forecasts."""
    history = original[:]
    result = []
    for f in forecasts:
        val = history[-1] + f
        result.append(val)
        history.append(val)
    return result


def _solve_linear(A, b):
    """Gaussian elimination."""
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


def fit_arma(series, p=2, q=1):
    """
    Fit ARMA(p,q) model on (already differenced) series via OLS approximation.
    MA terms use lagged residuals.
    """
    n = len(series)
    if n < p + q + 5:
        return None

    # Iterative OLS for ARMA
    residuals = [0.0] * n
    best_coeffs = None
    best_sse = float('inf')

    for iteration in range(3):
        X, y = [], []
        for i in range(max(p, q), n):
            row = [series[i - j - 1] for j in range(p)] + \
                  [residuals[i - j - 1] for j in range(q)]
            X.append([1.0] + row)
            y.append(series[i])

        m = len(y)
        k = len(X[0])
        XtX = [[sum(X[r][c1] * X[r][c2] for r in range(m)) for c2 in range(k)] for c1 in range(k)]
        # Add Ridge penalty to diagonal
        for i in range(k):
            XtX[i][i] += 1e-6
        Xty = [sum(X[r][c] * y[r] for r in range(m)) for c in range(k)]
        try:
            coeffs = _solve_linear(XtX, Xty)
        except:
            break

        # Update residuals
        for i in range(max(p, q), n):
            fitted = sum(coeffs[j] * X[i - max(p, q)][j] for j in range(k))
            residuals[i] = series[i] - fitted

        sse = sum(r**2 for r in residuals[max(p,q):])
        if sse < best_sse:
            best_sse = sse
            best_coeffs = coeffs

    if best_coeffs is None:
        return None

    return {
        "coeffs": best_coeffs,
        "p": p, "q": q,
        "residuals": residuals,
        "sse": best_sse,
    }


def arima_forecast(original_series, periods_ahead=6, d=1, p=3, q=2):
    """
    ARIMA(p,d,q) forecast pipeline:
    1. Difference series d times
    2. Fit ARMA(p,q) on differenced series
    3. Forecast on differenced scale
    4. Invert differencing
    """
    diff_series = difference(original_series, d)
    if len(diff_series) < p + q + 5:
        return None, None

    arma = fit_arma(diff_series, p=p, q=q)
    if arma is None:
        return None, None

    coeffs = arma["coeffs"]
    res = list(arma["residuals"])
    hist = list(diff_series)

    diff_forecasts = []
    for _ in range(periods_ahead):
        p_ = arma["p"]
        q_ = arma["q"]
        row = [1.0] + [hist[-(j+1)] for j in range(p_)] + [res[-(j+1)] for j in range(q_)]
        pred = sum(coeffs[j] * row[j] for j in range(len(coeffs)))
        diff_forecasts.append(pred)
        hist.append(pred)
        res.append(0.0)  # future residuals = 0

    # Invert differencing
    price_forecasts = undifference(original_series, diff_forecasts, d)
    price_forecasts = [max(0, round(p, 2)) for p in price_forecasts]

    return price_forecasts, arma


# ─────────────────────────────────────────────
# Gradient Boosted Trees Analog (Feature-based)
# ─────────────────────────────────────────────

class SimpleDecisionStump:
    """Single-feature, single-split regression stump."""
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X, y):
        best_sse = float('inf')
        m = len(y)
        
        # Random sub-sampling of features
        num_feats = len(X[0])
        sampled_features = random.sample(range(num_feats), max(2, int(num_feats * 0.8)))
        
        for feat in sampled_features:
            vals = sorted(set(row[feat] for row in X))
            for thresh in vals[:-1]:
                left_y = [y[i] for i in range(m) if X[i][feat] <= thresh]
                right_y = [y[i] for i in range(m) if X[i][feat] > thresh]
                if not left_y or not right_y:
                    continue
                lv = sum(left_y) / len(left_y)
                rv = sum(right_y) / len(right_y)
                sse = sum((v - lv)**2 for v in left_y) + sum((v - rv)**2 for v in right_y)
                if sse < best_sse:
                    best_sse = sse
                    self.feature = feat
                    self.threshold = thresh
                    self.left_val = lv
                    self.right_val = rv

    def predict_one(self, x):
        if self.feature is None:
            return 0.0
        return self.left_val if x[self.feature] <= self.threshold else self.right_val


def build_price_features(series, i, lag_steps=[1, 2, 3, 6, 12]):
    """Build feature vector for price at index i."""
    feats = []
    for lag in lag_steps:
        feats.append(series[i - lag] if i >= lag else series[0])
    # Month index (seasonality proxy)
    feats.append(i % 12)
    # Rolling mean
    w = min(6, i)
    feats.append(sum(series[max(0,i-w):i]) / max(w, 1))
    # Momentum
    feats.append(series[i-1] - series[max(0,i-3)] if i >= 3 else 0)
    return feats


def gradient_boost_price_forecast(series, periods_ahead=6, n_estimators=100, learning_rate=0.05):
    """
    Gradient Boosted regression stumps for price forecasting.
    Uses lag features, rolling stats, and momentum.
    """
    n = len(series)
    min_idx = 12  # need at least 12 lags

    X = [build_price_features(series, i) for i in range(min_idx, n)]
    y = [series[i] for i in range(min_idx, n)]

    # Initial prediction = mean
    f = [sum(y) / len(y)] * len(y)
    trees = []

    for _ in range(n_estimators):
        residuals = [y[i] - f[i] for i in range(len(y))]
        stump = SimpleDecisionStump()
        stump.fit(X, residuals)
        update = [stump.predict_one(x) for x in X]
        f = [f[i] + learning_rate * update[i] for i in range(len(y))]
        trees.append(stump)

    # Forecast future periods
    extended = list(series)
    forecasts = []
    for _ in range(periods_ahead):
        i = len(extended)
        x = build_price_features(extended, i)
        pred = sum(y) / len(y)  # base
        for stump in trees:
            pred += learning_rate * stump.predict_one(x)
        pred = max(0, round(pred, 2))
        forecasts.append(pred)
        extended.append(pred)

    return forecasts


# ─────────────────────────────────────────────
# Main Prediction Function
# ─────────────────────────────────────────────

def predict_prices(history_values, periods_ahead=6):
    """
    Ensemble price prediction:
      - ARIMA(3,1,2): captures autocorrelation + differenced trend
      - Gradient Boosted Stumps: captures non-linear feature interactions
    """
    if len(history_values) < 14:
        return {"error": "Insufficient price history (need >= 14 months)"}

    # --- ARIMA ---
    arima_preds, arima_model = arima_forecast(history_values, periods_ahead=periods_ahead)

    # --- Gradient Boosting ---
    try:
        gb_preds = gradient_boost_price_forecast(history_values, periods_ahead=periods_ahead)
    except Exception as e:
        gb_preds = None

    # --- Ensemble ---
    if arima_preds and gb_preds:
        ensemble = [round(0.55 * arima_preds[i] + 0.45 * gb_preds[i], 2) for i in range(periods_ahead)]
        method = "Ensemble: ARIMA(3,1,2) + Gradient Boosted Trees"
    elif arima_preds:
        ensemble = arima_preds
        method = "ARIMA(3,1,2)"
    elif gb_preds:
        ensemble = gb_preds
        method = "Gradient Boosted Trees"
    else:
        # Fallback: EMA extrapolation
        e = ema(history_values, 6)
        slope = (e[-1] - e[-6]) / 5 if len(e) >= 6 else 0
        ensemble = [round(max(0, e[-1] + slope * (i + 1)), 2) for i in range(periods_ahead)]
        method = "EMA Extrapolation (fallback)"

    # Technical indicators
    rsi_val = rsi(history_values)
    macd_line, macd_signal, macd_hist = macd(history_values)
    bb_upper, bb_mid, bb_lower = bollinger_bands(history_values)
    vol = compute_volatility(history_values)
    current_price = history_values[-1]
    predicted_avg = sum(ensemble) / len(ensemble)
    pct_change = round((predicted_avg - current_price) / current_price * 100, 2)

    # Volatility-adjusted confidence intervals
    std_est = current_price * (vol / 100)
    ci_lower = [round(max(0, p - 1.5 * std_est * math.sqrt(i+1)), 2) for i, p in enumerate(ensemble)]
    ci_upper = [round(p + 1.5 * std_est * math.sqrt(i+1), 2) for i, p in enumerate(ensemble)]

    # --- Signal Logic (multi-factor) ---
    signal_score = 0
    reasons = []

    # Price trend
    if pct_change > 5:
        signal_score -= 2
        reasons.append(f"Price projected to rise {pct_change:.1f}%")
    elif pct_change < -5:
        signal_score += 2
        reasons.append(f"Price projected to fall {abs(pct_change):.1f}%")

    # RSI
    if rsi_val > 70:
        signal_score += 1
        reasons.append(f"RSI={rsi_val} (overbought — may correct)")
    elif rsi_val < 30:
        signal_score -= 1
        reasons.append(f"RSI={rsi_val} (oversold — attractive entry)")

    # MACD
    if macd_hist is not None:
        if macd_hist > 0 and macd_line > macd_signal:
            signal_score -= 1
            reasons.append("MACD bullish crossover")
        elif macd_hist < 0:
            signal_score += 1
            reasons.append("MACD bearish — downward momentum")

    # Bollinger Band position
    if bb_upper and current_price > bb_upper:
        signal_score += 1
        reasons.append("Price above Bollinger upper band (overextended)")
    elif bb_lower and current_price < bb_lower:
        signal_score -= 1
        reasons.append("Price below Bollinger lower band (undervalued)")

    if signal_score <= -2:
        signal = "BUY NOW"
        signal_reason = " | ".join(reasons)
        signal_color = "red"
    elif signal_score >= 2:
        signal = "WAIT"
        signal_reason = " | ".join(reasons)
        signal_color = "green"
    else:
        signal = "NEUTRAL"
        signal_reason = "Mixed signals — monitor closely"
        signal_color = "yellow"

    return {
        "predictions": ensemble,
        "arima_predictions": arima_preds,
        "gb_predictions": gb_preds,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "current_price": current_price,
        "ema_values": ema(history_values, 6),
        "volatility_pct": vol,
        "rsi": rsi_val,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "price_signal": signal,
        "signal_reason": signal_reason,
        "signal_color": signal_color,
        "pct_change_expected": pct_change,
        "method": method,
    }


def run_all_price_predictions(data_path="data/procurement_data.json"):
    """Run price predictions for all materials."""
    with open(data_path) as f:
        data = json.load(f)

    results = {}
    for material, history in data["price_history"].items():
        values = [h["price"] for h in history]
        dates = [h["date"] for h in history]
        result = predict_prices(values)
        result["history_dates"] = dates
        result["history_values"] = values
        results[material] = result
        if "error" not in result:
            print(f"  [Price] {material}: Signal={result['price_signal']} | "
                  f"RSI={result['rsi']} | Δ={result['pct_change_expected']}% | Vol={result['volatility_pct']}%")
        else:
            print(f"  [Price] {material}: {result['error']}")

    return results


if __name__ == "__main__":
    run_all_price_predictions()
