"""
ARIA XGBoost Prediction Server v3.0
====================================
خادم FastAPI للتنبؤ بـ SL/TP لروبوت MT4/MT5

Endpoints:
  GET  /health          — فحص صحة الخادم
  GET  /info            — معلومات النموذج والإحصائيات
  POST /predict         — التنبؤ بـ SL/TP (الاستخدام الرئيسي)
  POST /predict/batch   — تنبؤات متعددة دفعة واحدة
  GET  /features        — قائمة المميزات المطلوبة
"""

import os
import json
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────
# إعداد السجلات
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aria-xgboost")

# ─────────────────────────────────────────────
# مسارات النماذج
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─────────────────────────────────────────────
# متغيرات عامة للنماذج
# ─────────────────────────────────────────────
model_sl: Optional[xgb.XGBRegressor] = None
model_tp: Optional[xgb.XGBRegressor] = None
model_meta: dict = {}
request_count: int = 0
start_time: float = time.time()

FEATURE_COLS = [
    "fast_ema", "slow_ema", "ema_diff", "adx", "di_plus", "di_minus",
    "atr", "rsi", "close1", "close2", "close3", "spread", "direction",
    "volatility_ratio", "trend_strength", "momentum"
]


# ─────────────────────────────────────────────
# تحميل النماذج عند بدء الخادم
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_sl, model_tp, model_meta
    logger.info("=" * 50)
    logger.info("ARIA XGBoost Server — بدء التشغيل")

    sl_path   = os.path.join(MODELS_DIR, "sl_model.json")
    tp_path   = os.path.join(MODELS_DIR, "tp_model.json")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")

    if not os.path.exists(sl_path) or not os.path.exists(tp_path):
        logger.error("النماذج غير موجودة! شغّل scripts/train_model.py أولاً.")
        raise RuntimeError("Models not found. Run scripts/train_model.py first.")

    model_sl = xgb.XGBRegressor()
    model_sl.load_model(sl_path)

    model_tp = xgb.XGBRegressor()
    model_tp.load_model(tp_path)

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_meta = json.load(f)

    logger.info(f"✅ SL Model loaded — MAE={model_meta.get('sl_model', {}).get('mae_pips', '?')} pips")
    logger.info(f"✅ TP Model loaded — MAE={model_meta.get('tp_model', {}).get('mae_pips', '?')} pips")
    logger.info("=" * 50)
    yield
    logger.info("ARIA XGBoost Server — إيقاف التشغيل")


# ─────────────────────────────────────────────
# إنشاء التطبيق
# ─────────────────────────────────────────────
app = FastAPI(
    title="ARIA XGBoost Prediction Server",
    description="خادم التنبؤ بـ SL/TP لروبوت التداول ARIA",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# نماذج البيانات (Pydantic)
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    # مؤشرات EMA
    fast_ema: float = Field(..., description="Fast EMA value")
    slow_ema: float = Field(..., description="Slow EMA value")

    # مؤشر ADX
    adx: float      = Field(..., ge=0, le=100, description="ADX value (0-100)")
    di_plus: float  = Field(..., ge=0, le=100, description="DI+ value")
    di_minus: float = Field(..., ge=0, le=100, description="DI- value")

    # ATR و RSI
    atr: float = Field(..., gt=0, description="ATR value")
    rsi: float = Field(..., ge=0, le=100, description="RSI value (0-100)")

    # أسعار الإغلاق
    close1: float = Field(..., gt=0, description="Close[1] — last closed bar")
    close2: float = Field(..., gt=0, description="Close[2]")
    close3: float = Field(..., gt=0, description="Close[3]")

    # معلومات إضافية
    spread: float    = Field(default=1.0, ge=0, description="Current spread in pips")
    direction: float = Field(..., description="Trade direction: 1=BUY, -1=SELL")

    # حقول اختيارية (يحسبها الخادم إذا لم تُرسَل)
    volatility_ratio: Optional[float] = Field(default=None)
    trend_strength:   Optional[float] = Field(default=None)
    momentum:         Optional[float] = Field(default=None)

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v):
        if v not in (1, -1, 1.0, -1.0):
            raise ValueError("direction must be 1 (BUY) or -1 (SELL)")
        return float(v)


class PredictResponse(BaseModel):
    sl_pips:    float
    tp_pips:    float
    confidence: float
    rr_ratio:   float
    message:    str


class BatchRequest(BaseModel):
    items: list[PredictRequest]


class BatchResponse(BaseModel):
    results: list[PredictResponse]
    count:   int


# ─────────────────────────────────────────────
# دالة مساعدة: بناء مصفوفة المميزات
# ─────────────────────────────────────────────
def build_features(req: PredictRequest) -> np.ndarray:
    ema_diff = req.fast_ema - req.slow_ema

    volatility_ratio = req.volatility_ratio
    if volatility_ratio is None:
        volatility_ratio = req.atr / (req.close1 + 1e-10)

    trend_strength = req.trend_strength
    if trend_strength is None:
        trend_strength = abs(ema_diff) / (req.atr + 1e-10)

    momentum = req.momentum
    if momentum is None:
        momentum = req.close1 - req.close3

    features = np.array([[
        req.fast_ema,
        req.slow_ema,
        ema_diff,
        req.adx,
        req.di_plus,
        req.di_minus,
        req.atr,
        req.rsi,
        req.close1,
        req.close2,
        req.close3,
        req.spread,
        req.direction,
        volatility_ratio,
        trend_strength,
        momentum,
    ]], dtype=np.float32)

    return features


def compute_confidence(sl_pips: float, tp_pips: float, adx: float, rsi: float) -> float:
    """
    يحسب مستوى الثقة (0.0–1.0) بناءً على:
    - قوة الاتجاه (ADX)
    - موقع RSI
    - نسبة Risk/Reward
    """
    # عامل ADX: كلما كان أعلى كانت الثقة أعلى
    adx_factor = min(adx / 60.0, 1.0)  # ADX=60 → factor=1.0

    # عامل RSI: أفضل ثقة عند 40-60 (منطقة محايدة)
    rsi_dist   = abs(rsi - 50) / 50.0  # 0=محايد، 1=متطرف
    rsi_factor = 1.0 - rsi_dist * 0.3

    # عامل RR: نسبة أعلى = ثقة أعلى (حتى حد معين)
    rr = tp_pips / (sl_pips + 1e-10)
    rr_factor = min(rr / 3.0, 1.0)

    confidence = (adx_factor * 0.5 + rsi_factor * 0.2 + rr_factor * 0.3)
    return round(float(np.clip(confidence, 0.30, 0.95)), 4)


# ─────────────────────────────────────────────
# Middleware: تسجيل الطلبات
# ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    global request_count
    request_count += 1
    t0 = time.time()
    response = await call_next(request)
    elapsed = (time.time() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """فحص صحة الخادم — يُستخدم من MT4 للتحقق من الاتصال."""
    return {
        "status":    "ok",
        "models":    "loaded" if model_sl and model_tp else "not_loaded",
        "uptime_s":  round(time.time() - start_time, 1),
        "requests":  request_count,
        "version":   "3.0.0",
    }


@app.get("/info", tags=["System"])
async def info():
    """معلومات النموذج والإحصائيات التفصيلية."""
    return {
        "server":    "ARIA XGBoost Prediction Server",
        "version":   "3.0.0",
        "model_meta": model_meta,
        "features":  FEATURE_COLS,
        "uptime_s":  round(time.time() - start_time, 1),
        "requests":  request_count,
    }


@app.get("/features", tags=["Model"])
async def features():
    """قائمة المميزات المطلوبة مع وصفها."""
    return {
        "required_features": [
            {"name": "fast_ema",   "type": "float", "desc": "Fast EMA (e.g. 8-period)"},
            {"name": "slow_ema",   "type": "float", "desc": "Slow EMA (e.g. 21-period)"},
            {"name": "adx",        "type": "float", "desc": "ADX indicator (0-100)"},
            {"name": "di_plus",    "type": "float", "desc": "DI+ indicator"},
            {"name": "di_minus",   "type": "float", "desc": "DI- indicator"},
            {"name": "atr",        "type": "float", "desc": "ATR value (price units)"},
            {"name": "rsi",        "type": "float", "desc": "RSI (0-100)"},
            {"name": "close1",     "type": "float", "desc": "Close[1] last closed bar"},
            {"name": "close2",     "type": "float", "desc": "Close[2]"},
            {"name": "close3",     "type": "float", "desc": "Close[3]"},
            {"name": "direction",  "type": "float", "desc": "1=BUY, -1=SELL"},
        ],
        "optional_features": [
            {"name": "spread",           "default": 1.0,  "desc": "Spread in pips"},
            {"name": "volatility_ratio", "default": "auto", "desc": "ATR/price ratio"},
            {"name": "trend_strength",   "default": "auto", "desc": "|ema_diff|/ATR"},
            {"name": "momentum",         "default": "auto", "desc": "close1-close3"},
        ]
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    """
    التنبؤ بـ SL/TP — الـ endpoint الرئيسي المستخدم من MT4.

    يُرجع:
    - sl_pips: وقف الخسارة المثلى بالنقاط
    - tp_pips: جني الأرباح المثلى بالنقاط
    - confidence: مستوى الثقة (0.0–1.0)
    - rr_ratio: نسبة المخاطرة/المكافأة
    """
    if model_sl is None or model_tp is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        X = build_features(req)

        sl_raw = float(model_sl.predict(X)[0])
        tp_raw = float(model_tp.predict(X)[0])

        # تقييد النتائج في نطاق معقول
        sl_pips = round(float(np.clip(sl_raw, 5.0, 150.0)), 2)
        tp_pips = round(float(np.clip(tp_raw, 10.0, 300.0)), 2)

        # ضمان أن TP > SL دائماً
        if tp_pips <= sl_pips:
            tp_pips = round(sl_pips * 1.5, 2)

        rr_ratio   = round(tp_pips / sl_pips, 3)
        confidence = compute_confidence(sl_pips, tp_pips, req.adx, req.rsi)

        direction_str = "BUY" if req.direction > 0 else "SELL"
        message = (
            f"{direction_str}: SL={sl_pips}p TP={tp_pips}p "
            f"RR={rr_ratio:.2f} Conf={confidence:.0%}"
        )

        return PredictResponse(
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            confidence=confidence,
            rr_ratio=rr_ratio,
            message=message,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    """تنبؤات متعددة دفعة واحدة (حتى 100 طلب)."""
    if len(batch.items) > 100:
        raise HTTPException(status_code=400, detail="Max 100 items per batch")

    results = []
    for item in batch.items:
        try:
            X = build_features(item)
            sl_raw = float(model_sl.predict(X)[0])
            tp_raw = float(model_tp.predict(X)[0])
            sl_pips = round(float(np.clip(sl_raw, 5.0, 150.0)), 2)
            tp_pips = round(float(np.clip(tp_raw, 10.0, 300.0)), 2)
            if tp_pips <= sl_pips:
                tp_pips = round(sl_pips * 1.5, 2)
            rr_ratio   = round(tp_pips / sl_pips, 3)
            confidence = compute_confidence(sl_pips, tp_pips, item.adx, item.rsi)
            direction_str = "BUY" if item.direction > 0 else "SELL"
            results.append(PredictResponse(
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                confidence=confidence,
                rr_ratio=rr_ratio,
                message=f"{direction_str}: SL={sl_pips}p TP={tp_pips}p RR={rr_ratio:.2f}",
            ))
        except Exception as e:
            results.append(PredictResponse(
                sl_pips=0, tp_pips=0, confidence=0, rr_ratio=0,
                message=f"ERROR: {str(e)}"
            ))

    return BatchResponse(results=results, count=len(results))


# ─────────────────────────────────────────────
# نقطة الدخول
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
