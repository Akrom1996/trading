FROM python:3.11-slim

WORKDIR /app

# Ensure standard output and error are flushed immediately and python path is root
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default runtime configuration (overridden per-container in docker-compose.yml)
ENV SYMBOL=SOL/USDT
ENV ROLE=coin_trader
ENV STATE_DIR=/app/state
ENV DB_PATH=/app/data/bot_data.db

CMD ["python", "-u", "main.py"]