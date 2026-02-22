FROM python:3.11-slim

WORKDIR /app

# تثبيت المتطلبات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ الكود والنماذج
COPY main.py .
COPY models/ ./models/
COPY scripts/ ./scripts/

# المنفذ الافتراضي
EXPOSE 8000

# تشغيل الخادم
CMD ["python", "main.py"]
