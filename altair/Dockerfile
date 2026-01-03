# Altair - CLI/Terminal Specialist
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including tmux for terminal sessions
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for session persistence
RUN mkdir -p /root/.hyperion

# Copy application code
COPY shared/ ./shared/
COPY altair/ ./altair/

# Run Altair bot
CMD ["python", "-m", "altair.bot"]
