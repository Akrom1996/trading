FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Overridden per-container in docker-compose.yml (SOL/USDT, ZEC/USDT, BTC/USDT)
ENV SYMBOL=ZEC/USDT
# Where model files + positions_*.json get written -- mount this as a
# volume per container so state survives image rebuilds/redeploys.
ENV STATE_DIR=/app/state

CMD ["python", "-u", "main.py"]