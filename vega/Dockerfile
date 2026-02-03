# Vega - Conversational Agent
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-base.txt requirements-vega.txt ./
RUN pip install --no-cache-dir -r requirements-vega.txt

# Copy application code
COPY shared/ ./shared/
COPY vega/ ./vega/

# Run Vega bot
CMD ["python", "-m", "vega.bot"]
