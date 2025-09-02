#!/bin/bash
set -e

# Function to check if MongoDB is ready
wait_for_mongodb() {
    echo "Waiting for MongoDB to be ready..."
    until mongosh --eval "db.adminCommand('ping')" "$MONGODB_URI" &>/dev/null; do
        echo "MongoDB not ready yet, waiting..."
        sleep 2
    done
    echo "MongoDB is ready!"
}

# Function to check if Redis is ready
wait_for_redis() {
    echo "Waiting for Redis to be ready..."
    until nc -z "$REDIS_HOST" "$REDIS_PORT"; do
        echo "Redis not ready yet, waiting..."
        sleep 2
    done
    echo "Redis is ready!"
}

# Wait for dependencies to be ready
if [[ -n "$MONGODB_URI" ]]; then
    wait_for_mongodb
fi

if [[ -n "$REDIS_HOST" && -n "$REDIS_PORT" ]]; then
    wait_for_redis
fi

# Set up config directory and ensure it exists
mkdir -p /app/config /app/logs /app/data

# Create a basic config file if none exists
if [ ! -f "/app/config/config.yaml" ] && [ ! -f "/app/config/config.json" ]; then
    echo "Creating default configuration file..."
    python -m trading_bot.config.migrate_configs --base-dir /app/config --output /app/config/config.yaml
    echo "Configuration file created at /app/config/config.yaml"
fi

# Execute the main command
exec "$@" 