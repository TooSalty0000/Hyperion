# Polaris - Calendar/Scheduling Agent
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-base.txt requirements-polaris.txt ./
RUN pip install --no-cache-dir -r requirements-polaris.txt

# Create directory for Google token persistence
RUN mkdir -p /root/.hyperion/polaris

# Copy application code
COPY shared/ ./shared/
COPY polaris/ ./polaris/

# Run Polaris bot
CMD ["python", "-m", "polaris.bot"]
