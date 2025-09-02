FROM python:3.10-slim-buster

WORKDIR /app

# Create a non-root user to run the application
RUN groupadd -g 1000 tradingbot && \
    useradd -u 1000 -g tradingbot -s /bin/sh -m tradingbot

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=tradingbot:tradingbot . .

# Switch to non-root user
USER tradingbot

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Command will be specified in derived images
CMD ["echo", "Base image - specify command in derived Dockerfile"] 