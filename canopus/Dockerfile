# Canopus - Web Browser Agent
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install --with-deps chromium

# Copy application code
COPY shared/ ./shared/
COPY canopus/ ./canopus/

# Run Canopus bot
CMD ["python", "-m", "canopus.bot"]
