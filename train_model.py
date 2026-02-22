"""
ARIA XGBoost Trainer
====================
يولّد بيانات تدريب واقعية مبنية على ديناميكيات السوق الحقيقية،
ثم يدرّب نموذجين منفصلين للتنبؤ بـ SL و TP بالنقاط (pips).

المدخلات (Features):
  fast_ema, slow_ema, ema_diff, adx, di_plus, di_minus,
  atr, rsi, close1, close2, close3, spread, direction,
  volatility_ratio, trend_strength, momentum

المخرجات (Targets):
  sl_pips  — المسافة المثلى لوقف الخسارة
  tp_pips  — المسافة المثلى لجني الأرباح
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

RANDOM_SEED = 42
N_SAMPLES   = 50_000
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. توليد بيانات تدريب واقعية
# ─────────────────────────────────────────────
def generate_training_data(n: int) -> pd.DataFrame:
    """يولّد بيانات تحاكي ظروف السوق الحقيقية."""

    # محاكاة سعر إغلاق بـ Random Walk
    price_base = 1.1000
    returns    = np.random.normal(0, 0.0005, n)
    prices     = price_base + np.cumsum(returns)
    prices     = np.abs(prices)  # لا أسعار سالبة

    # ATR — التقلب الحقيقي
    atr_base = np.random.uniform(0.0005, 0.003, n)
    atr      = atr_base * (1 + 0.5 * np.abs(np.sin(np.arange(n) / 200)))

    # RSI — 14 فترة مبسّطة
    rsi = np.random.uniform(20, 80, n)
    # تحيّز: في الاتجاهات القوية يميل RSI للأطراف
    trend_factor = np.sin(np.arange(n) / 100)
    rsi = np.clip(rsi + trend_factor * 15, 10, 90)

    # EMA
    fast_ema = prices + np.random.normal(0, 0.0002, n)
    slow_ema = prices + np.random.normal(0, 0.0004, n)
    ema_diff = fast_ema - slow_ema

    # ADX و DI
    adx      = np.random.uniform(10, 60, n)
    di_plus  = np.random.uniform(10, 40, n)
    di_minus = np.random.uniform(10, 40, n)
    # ربط ADX بقوة الاتجاه
    adx = adx + np.abs(ema_diff) / 0.0001 * 2
    adx = np.clip(adx, 10, 80)

    # Spread
    spread = np.random.uniform(0.5, 3.0, n)

    # Close prices
    close1 = prices
    close2 = prices - returns
    close3 = prices - returns * 2

    # Direction: 1=BUY, -1=SELL
    direction = np.where(ema_diff > 0, 1, -1).astype(float)

    # Volatility ratio: ATR / price
    volatility_ratio = atr / prices

    # Trend strength: |ema_diff| / atr
    trend_strength = np.abs(ema_diff) / (atr + 1e-10)

    # Momentum: change over 3 bars
    momentum = close1 - close3

    # ─────────────────────────────────────────
    # حساب SL/TP المثلى (الهدف الحقيقي)
    # ─────────────────────────────────────────
    # SL المثلى: مبنية على ATR مع تعديل بناءً على قوة الاتجاه
    # كلما كان الاتجاه أقوى (ADX عالٍ) → SL أوسع قليلاً
    # كلما كان التقلب أعلى (ATR كبير) → SL أوسع
    sl_base = atr * 10_000 * 1.5   # تحويل لـ pips (EURUSD: 1 pip = 0.0001)
    sl_adx_factor = 1 + (adx - 25) / 100  # ADX 25 = عامل 1.0
    sl_pips = sl_base * sl_adx_factor
    sl_pips = np.clip(sl_pips + np.random.normal(0, 2, n), 8, 80)

    # TP المثلى: نسبة Risk/Reward مبنية على قوة الاتجاه
    # اتجاه قوي (ADX > 40) → RR أعلى (1:2.5)
    # اتجاه ضعيف (ADX < 25) → RR أقل (1:1.5)
    rr_ratio = 1.5 + (adx - 20) / 40  # من 1.5 إلى 2.5
    rr_ratio = np.clip(rr_ratio, 1.2, 3.0)
    tp_pips  = sl_pips * rr_ratio
    tp_pips  = np.clip(tp_pips + np.random.normal(0, 3, n), 15, 200)

    # إضافة ضوضاء واقعية
    sl_pips = sl_pips * np.random.uniform(0.85, 1.15, n)
    tp_pips = tp_pips * np.random.uniform(0.85, 1.15, n)

    df = pd.DataFrame({
        "fast_ema":        fast_ema,
        "slow_ema":        slow_ema,
        "ema_diff":        ema_diff,
        "adx":             adx,
        "di_plus":         di_plus,
        "di_minus":        di_minus,
        "atr":             atr,
        "rsi":             rsi,
        "close1":          close1,
        "close2":          close2,
        "close3":          close3,
        "spread":          spread,
        "direction":       direction,
        "volatility_ratio": volatility_ratio,
        "trend_strength":  trend_strength,
        "momentum":        momentum,
        "sl_pips":         sl_pips,
        "tp_pips":         tp_pips,
    })
    return df


# ─────────────────────────────────────────────
# 2. تدريب النموذج
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "fast_ema", "slow_ema", "ema_diff", "adx", "di_plus", "di_minus",
    "atr", "rsi", "close1", "close2", "close3", "spread", "direction",
    "volatility_ratio", "trend_strength", "momentum"
]

def train_and_save():
    print("=" * 60)
    print("ARIA XGBoost Trainer — بدء التدريب")
    print("=" * 60)

    print(f"\n[1/5] توليد {N_SAMPLES:,} عينة تدريب...")
    df = generate_training_data(N_SAMPLES)
    print(f"      SL range: {df['sl_pips'].min():.1f} – {df['sl_pips'].max():.1f} pips")
    print(f"      TP range: {df['tp_pips'].min():.1f} – {df['tp_pips'].max():.1f} pips")

    X = df[FEATURE_COLS].values
    y_sl = df["sl_pips"].values
    y_tp = df["tp_pips"].values

    print("\n[2/5] تقسيم البيانات (80% تدريب / 20% اختبار)...")
    X_train, X_test, y_sl_train, y_sl_test = train_test_split(
        X, y_sl, test_size=0.2, random_state=RANDOM_SEED
    )
    _, _, y_tp_train, y_tp_test = train_test_split(
        X, y_tp, test_size=0.2, random_state=RANDOM_SEED
    )

    print("\n[3/5] تدريب نموذج SL...")
    model_sl = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model_sl.fit(
        X_train, y_sl_train,
        eval_set=[(X_test, y_sl_test)],
        verbose=False,
    )
    sl_pred  = model_sl.predict(X_test)
    sl_mae   = mean_absolute_error(y_sl_test, sl_pred)
    sl_r2    = r2_score(y_sl_test, sl_pred)
    print(f"      MAE = {sl_mae:.2f} pips  |  R² = {sl_r2:.4f}")

    print("\n[4/5] تدريب نموذج TP...")
    model_tp = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model_tp.fit(
        X_train, y_tp_train,
        eval_set=[(X_test, y_tp_test)],
        verbose=False,
    )
    tp_pred = model_tp.predict(X_test)
    tp_mae  = mean_absolute_error(y_tp_test, tp_pred)
    tp_r2   = r2_score(y_tp_test, tp_pred)
    print(f"      MAE = {tp_mae:.2f} pips  |  R² = {tp_r2:.4f}")

    print("\n[5/5] حفظ النماذج والبيانات الوصفية...")
    model_sl.save_model(os.path.join(MODELS_DIR, "sl_model.json"))
    model_tp.save_model(os.path.join(MODELS_DIR, "tp_model.json"))

    # حفظ أسماء المميزات وإحصائيات التدريب
    meta = {
        "feature_cols": FEATURE_COLS,
        "n_samples": N_SAMPLES,
        "sl_model": {
            "mae_pips": round(sl_mae, 3),
            "r2_score": round(sl_r2, 4),
            "n_estimators": 500,
        },
        "tp_model": {
            "mae_pips": round(tp_mae, 3),
            "r2_score": round(tp_r2, 4),
            "n_estimators": 500,
        },
        "version": "3.0.0",
        "symbol": "EURUSD",
        "pip_value": 0.0001,
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ التدريب اكتمل بنجاح!")
    print(f"   النماذج محفوظة في: {MODELS_DIR}/")
    print(f"   SL Model: MAE={sl_mae:.2f} pips, R²={sl_r2:.4f}")
    print(f"   TP Model: MAE={tp_mae:.2f} pips, R²={tp_r2:.4f}")
    print("=" * 60)
    return meta


if __name__ == "__main__":
    train_and_save()
