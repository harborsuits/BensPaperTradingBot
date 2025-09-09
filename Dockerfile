# Multi-stage Docker build for BenBot
FROM node:18-alpine AS base

# Install dependencies for native modules
RUN apk add --no-cache python3 make g++

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Production stage
FROM node:18-alpine AS production

# Install Python and pip for any Python dependencies
RUN apk add --no-cache python3 py3-pip

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S benbot -u 1001

WORKDIR /app

# Copy installed dependencies from base stage
COPY --from=base --chown=benbot:nodejs /app/node_modules ./node_modules

# Copy application code
COPY --chown=benbot:nodejs . .

# Create data directories
RUN mkdir -p /app/data && chown -R benbot:nodejs /app/data

# Switch to non-root user
USER benbot

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:4000/health', (res) => { \
    process.exit(res.statusCode === 200 ? 0 : 1) \
  }).on('error', () => process.exit(1))"

# Expose ports
EXPOSE 3003 4000

# Start the application
CMD ["node", "live-api/server.js"]

