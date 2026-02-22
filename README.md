# ARIA XGBoost Prediction Server v3.0

خادم التنبؤ بـ SL/TP لروبوت التداول الذكي ARIA — مبني على FastAPI + XGBoost.

---

## نتائج التدريب

| النموذج | دقة MAE | R² Score | عدد العينات |
|---------|---------|----------|------------|
| SL Model | **3.49 pips** | **0.9444** | 50,000 |
| TP Model | **7.84 pips** | **0.9527** | 50,000 |

---

## التشغيل المحلي السريع

```bash
# 1. تثبيت المتطلبات
pip install -r requirements.txt

# 2. تدريب النماذج (مرة واحدة فقط)
python scripts/train_model.py

# 3. تشغيل الخادم
python main.py
# أو
uvicorn main:app --host 0.0.0.0 --port 8000
```

الخادم يعمل على: http://localhost:8000
التوثيق التفاعلي: http://localhost:8000/docs

---

## النشر على Render.com (مجاني)

1. ارفع المجلد على GitHub
2. اذهب إلى [render.com](https://render.com) → New Web Service
3. اختر المستودع
4. Build Command: `pip install -r requirements.txt && python scripts/train_model.py`
5. Start Command: `python main.py`
6. انسخ الـ URL وضعه في إعدادات الروبوت

---

## النشر على Railway

```bash
# تثبيت Railway CLI
npm install -g @railway/cli

# تسجيل الدخول والنشر
railway login
railway init
railway up
```

---

## النشر بـ Docker

```bash
# بناء الصورة
docker build -t aria-xgboost .

# تشغيل الحاوية
docker run -p 8000:8000 aria-xgboost

# أو بـ docker-compose
docker-compose up -d
```

---

## الـ Endpoints

### GET /health
فحص صحة الخادم — استخدمه من MT4 للتحقق من الاتصال.

```json
{
  "status": "ok",
  "models": "loaded",
  "uptime_s": 120.5,
  "requests": 45,
  "version": "3.0.0"
}
```

### POST /predict
التنبؤ الرئيسي بـ SL/TP.

**الطلب:**
```json
{
  "fast_ema": 1.10850,
  "slow_ema": 1.10620,
  "adx": 42.5,
  "di_plus": 28.3,
  "di_minus": 14.7,
  "atr": 0.00085,
  "rsi": 58.2,
  "close1": 1.10900,
  "close2": 1.10820,
  "close3": 1.10750,
  "spread": 1.2,
  "direction": 1
}
```

**الاستجابة:**
```json
{
  "sl_pips": 14.68,
  "tp_pips": 29.06,
  "confidence": 0.7423,
  "rr_ratio": 1.98,
  "message": "BUY: SL=14.68p TP=29.06p RR=1.98 Conf=74%"
}
```

**direction:** `1` = BUY، `-1` = SELL

### GET /info
معلومات النموذج والإحصائيات.

### GET /features
قائمة المميزات المطلوبة مع وصفها.

### POST /predict/batch
تنبؤات متعددة دفعة واحدة (حتى 100 طلب).

---

## إعداد الروبوت للاتصال بالخادم

في إعدادات الروبوت `ARIA_SYAF_MT4.mq4`:

```
InpEnableXGBoost    = true
InpXGBoostAPIURL    = https://your-server.onrender.com/predict
InpXGBoostMinConf   = 0.65
```

في MT4: **Tools → Options → Expert Advisors → Allow WebRequest**
أضف: `https://your-server.onrender.com`

---

## إعادة تدريب النموذج ببياناتك الحقيقية

```python
# في scripts/train_model.py، استبدل generate_training_data() بـ:
df = pd.read_csv('your_eurusd_data.csv')
# تأكد من وجود أعمدة: fast_ema, slow_ema, adx, di_plus, di_minus,
#                       atr, rsi, close1, close2, close3, spread,
#                       direction, sl_pips, tp_pips
```

---

## هيكل الملفات

```
xgboost_server/
├── main.py                  ← الخادم الرئيسي
├── requirements.txt         ← المتطلبات
├── Dockerfile               ← للنشر بـ Docker
├── docker-compose.yml       ← للتشغيل المحلي بـ Docker
├── render.yaml              ← للنشر على Render.com
├── railway.json             ← للنشر على Railway
├── Procfile                 ← للنشر على Heroku
├── README.md                ← هذا الملف
├── models/
│   ├── sl_model.json        ← نموذج SL المدرّب
│   ├── tp_model.json        ← نموذج TP المدرّب
│   └── model_meta.json      ← بيانات وصفية للنموذج
└── scripts/
    └── train_model.py       ← سكريبت التدريب
```
