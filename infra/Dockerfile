FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ procps netcat-openbsd \
                                             curl gnupg git make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Add MongoDB repository and install MongoDB client tools
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | \
    gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor && \
    echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] http://repo.mongodb.org/apt/debian bullseye/mongodb-org/6.0 main" | \
    tee /etc/apt/sources.list.d/mongodb-org-6.0.list && \
    apt-get update && \
    apt-get install -y mongodb-org-tools mongodb-mongosh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only pyproject.toml first for better caching
COPY pyproject.toml .

# Install package in development mode and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/config /app/cache

# Copy application code
COPY . .

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MONGODB_URI=mongodb://mongodb:27017/
ENV MONGODB_DATABASE=bensbot_trading
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_PREFIX=bensbot:
ENV LOG_LEVEL=INFO

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Health check - to verify the process is still alive (not API health check)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD pgrep -f "python.*trading_bot" || exit 1

# Default command - will be overridden by docker-compose
CMD ["python", "-m", "trading_bot.run_bot", "--config", "/app/config/config.yaml"] 