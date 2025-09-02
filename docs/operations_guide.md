# BensBot Operations Guide

## Overview

This operations guide provides comprehensive instructions for setting up, running, monitoring, and troubleshooting the BensBot trading system in a production environment. The guide focuses on the enhanced reliability and efficiency components:

1. Persistence Layer (MongoDB)
2. Watchdog & Fault Tolerance
3. Dynamic Capital Scaling
4. Strategy Retirement & Promotion
5. Execution Quality Modeling

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)
  - [Docker Deployment](#docker-deployment)
  - [Kubernetes Deployment](#kubernetes-deployment)
- [Monitoring System Health](#monitoring-system-health)
  - [Using the Dashboard](#using-the-dashboard)
  - [API Health Endpoints](#api-health-endpoints)
  - [Log Monitoring](#log-monitoring)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Recovery Procedures](#recovery-procedures)
  - [Data Backups](#data-backups)
- [Maintenance](#maintenance)
  - [Scheduled Tasks](#scheduled-tasks)
  - [Database Management](#database-management)
  - [System Updates](#system-updates)

## System Requirements

- Python 3.9+
- MongoDB 5.0+
- Docker 20.10+ (for container deployment)
- Kubernetes 1.22+ (for orchestrated deployment)
- Minimum 4GB RAM, 2 CPU cores, 20GB storage
- Network access to exchange/broker APIs
- Internet connection for market data feeds

## Installation

### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourorganization/bensbot.git
   cd bensbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install MongoDB:
   - macOS (using Homebrew):
     ```bash
     brew tap mongodb/brew
     brew install mongodb-community
     brew services start mongodb-community
     ```
   - Linux (Ubuntu/Debian):
     ```bash
     wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
     echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
     sudo apt-get update
     sudo apt-get install -y mongodb-org
     sudo systemctl start mongod
     ```

5. Configure MongoDB connection in your environment:
   ```bash
   export MONGODB_URI="mongodb://localhost:27017/"
   export MONGODB_DATABASE="bensbot"
   ```

### Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t bensbot:latest .
   ```

2. Run with MongoDB:
   ```bash
   docker network create bensbot-network
   docker run -d --name mongodb --network bensbot-network -v mongodb_data:/data/db mongo:5.0
   docker run -d --name bensbot --network bensbot-network -p 8000:8000 -p 8050:8050 -e MONGODB_URI="mongodb://mongodb:27017/" bensbot:latest
   ```

## Configuration

The BensBot system is configured using environment variables and configuration files. Key configuration parameters include:

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGODB_URI` | MongoDB connection string | `mongodb://localhost:27017/` |
| `MONGODB_DATABASE` | MongoDB database name | `bensbot` |
| `INITIAL_CAPITAL` | Initial trading capital | `100000` |
| `WATCHDOG_INTERVAL` | Watchdog check interval (seconds) | `30` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENABLE_PERSISTENCE` | Enable persistence layer | `true` |
| `ENABLE_WATCHDOG` | Enable watchdog monitoring | `true` |
| `ENABLE_DYNAMIC_CAPITAL` | Enable dynamic capital scaling | `true` |
| `ENABLE_STRATEGY_RETIREMENT` | Enable strategy lifecycle management | `true` |
| `ENABLE_EXECUTION_MODEL` | Enable execution quality modeling | `true` |

### Configuration Files

Advanced configuration can be modified in the following files:

1. **Enhanced Components Configuration**:
   - `config/enhanced_components.json` - Contains detailed settings for all enhanced components

2. **Strategy Configuration**:
   - `config/strategies.json` - Strategy parameters and lifecycle settings

3. **Execution Model Settings**:
   - `config/execution_model.json` - Execution quality parameters per symbol and session

## Production Deployment

### Docker Deployment

The BensBot system can be deployed using Docker for simplified deployment and management.

1. **Prerequisites**:
   - Docker and Docker Compose installed
   - Access to a Docker registry (optional, for production)

2. **Configuration**:
   Create a `.env` file with your production settings:
   ```
   MONGODB_URI=mongodb://mongodb:27017/
   MONGODB_DATABASE=bensbot
   INITIAL_CAPITAL=100000
   JWT_SECRET_KEY=your_secure_jwt_key
   LOG_LEVEL=INFO
   ```

3. **Deployment**:
   ```bash
   docker-compose up -d
   ```

4. **Scaling** (for higher availability):
   ```bash
   docker-compose up -d --scale bensbot=2
   ```

5. **Monitoring**:
   ```bash
   docker-compose logs -f
   ```

### Kubernetes Deployment

For production-grade deployment with high availability, fault tolerance, and scalability, use Kubernetes.

1. **Prerequisites**:
   - Kubernetes cluster (e.g., AKS, EKS, GKE, or self-managed)
   - `kubectl` configured to access your cluster
   - Container registry access

2. **Deployment Steps**:

   a. **Create Namespace**:
   ```bash
   kubectl create namespace bensbot
   ```

   b. **Create Secrets**:
   ```bash
   kubectl create secret generic bensbot-secrets \
     --from-literal=JWT_SECRET_KEY=your_secure_jwt_key \
     --from-literal=API_KEY=your_api_key \
     -n bensbot
   ```

   c. **Deploy MongoDB**:
   ```bash
   kubectl apply -f kubernetes/mongodb.yaml -n bensbot
   ```

   d. **Deploy BensBot**:
   ```bash
   kubectl apply -f kubernetes/bensbot.yaml -n bensbot
   ```

   e. **Verify Deployment**:
   ```bash
   kubectl get pods -n bensbot
   kubectl get services -n bensbot
   ```

3. **Accessing the Application**:
   ```bash
   kubectl port-forward service/bensbot 8000:8000 8050:8050 -n bensbot
   ```

4. **Scaling**:
   ```bash
   kubectl scale deployment bensbot --replicas=3 -n bensbot
   ```

## Monitoring System Health

### Using the Dashboard

The enhanced monitoring dashboard provides a comprehensive view of system health, performance, and execution metrics.

1. **Accessing the Dashboard**:
   - Local: `http://localhost:8050/`
   - Production: Through your configured ingress/domain

2. **Dashboard Sections**:
   - **Overview**: System health, components status, recent issues
   - **Persistence Monitor**: MongoDB stats, recent trades, system logs
   - **Watchdog Monitor**: Service health, recovery history, dependencies
   - **Capital Management**: Capital metrics, risk parameters, scaling factors
   - **Strategy Performance**: Strategy metrics, lifecycle events, rankings
   - **Execution Quality**: Slippage, latency, spread, market impact

3. **Using Filters**:
   The dashboard supports filtering by date range, symbol, strategy, and other parameters depending on the view.

### API Health Endpoints

BensBot provides REST API endpoints for monitoring system health:

1. **System Health**:
   - `GET /health` - Overall system health status
   - `GET /health/detailed` - Detailed component health information

2. **Watchdog Status**:
   - `GET /watchdog/status` - Current watchdog and monitored services status

3. **Metrics**:
   - `GET /metrics/capital` - Capital metrics
   - `GET /metrics/strategies` - Strategy performance metrics
   - `GET /metrics/execution` - Execution quality metrics

### Log Monitoring

Logs are stored in both the filesystem and MongoDB for comprehensive monitoring.

1. **Log Locations**:
   - Filesystem: `/app/logs/` (within the container)
   - MongoDB: `system_logs` collection

2. **Log Levels**:
   - `ERROR`: Critical issues requiring immediate attention
   - `WARNING`: Potential issues that might require attention
   - `INFO`: General operational information
   - `DEBUG`: Detailed debugging information (high volume)

3. **Log Aggregation**:
   In production, consider using a log aggregation system like ELK Stack (Elasticsearch, Logstash, Kibana) or Graylog.

## Troubleshooting

### Common Issues

1. **Connectivity Issues**:
   - **Symptoms**: "Not connected" errors in logs or dashboard
   - **Solution**: 
     - Check network connectivity to MongoDB
     - Verify broker/exchange API connectivity
     - Ensure required ports are open in firewalls

2. **Watchdog Recovery Loops**:
   - **Symptoms**: Services repeatedly failing and recovering
   - **Solution**:
     - Check dependent services' health
     - Increase recovery cooldown times
     - Verify credential validity

3. **Memory Issues**:
   - **Symptoms**: OOM errors, slow performance
   - **Solution**:
     - Increase container memory limits
     - Optimize MongoDB queries
     - Reduce data retention periods

4. **Database Performance**:
   - **Symptoms**: Slow queries, timeouts
   - **Solution**:
     - Create appropriate indexes
     - Implement data archiving
     - Upgrade MongoDB resources

### Recovery Procedures

1. **System Crash Recovery**:
   ```bash
   # Restart the BensBot container/pod
   docker restart bensbot
   # or
   kubectl rollout restart deployment bensbot -n bensbot
   
   # Verify recovery
   docker logs bensbot | grep "Restored"
   # or
   kubectl logs -l app=bensbot -n bensbot | grep "Restored"
   ```

2. **MongoDB Recovery**:
   ```bash
   # Check MongoDB status
   docker exec -it mongodb mongosh --eval "db.serverStatus()"
   
   # Repair database if needed
   docker exec -it mongodb mongod --repair
   
   # Restore from backup if available
   docker exec -it mongodb mongorestore --db bensbot /app/backups/bensbot-backup-YYYY-MM-DD/
   ```

3. **Manual State Reset**:
   ```bash
   # Connect to MongoDB
   docker exec -it mongodb mongosh
   
   # Clear problematic states
   use bensbot
   db.strategy_states.updateOne({strategy_id: "problematic_strategy"}, {$set: {status: "INACTIVE"}})
   
   # Restart the component
   docker exec -it bensbot python -c "from trading_bot.core.enhanced_integration import reload_component; reload_component('strategy_manager')"
   ```

### Data Backups

1. **Automated Backups**:
   The system performs daily backups of the MongoDB database.

2. **Manual Backup**:
   ```bash
   # Create a backup
   docker exec -it mongodb mongodump --db bensbot --out /app/backups/bensbot-backup-$(date +%Y-%m-%d)/
   
   # Extract backup from container
   docker cp mongodb:/app/backups/bensbot-backup-YYYY-MM-DD/ ./backups/
   ```

3. **Restore from Backup**:
   ```bash
   # Copy backup to container
   docker cp ./backups/bensbot-backup-YYYY-MM-DD/ mongodb:/app/backups/
   
   # Restore
   docker exec -it mongodb mongorestore --db bensbot /app/backups/bensbot-backup-YYYY-MM-DD/bensbot/
   ```

## Maintenance

### Scheduled Tasks

Set up the following scheduled maintenance tasks:

1. **Daily Backups**:
   ```bash
   # Add to crontab
   0 1 * * * docker exec mongodb mongodump --db bensbot --out /app/backups/bensbot-backup-$(date +\%Y-\%m-\%d)/
   ```

2. **Log Rotation**:
   ```bash
   # Add to crontab
   0 0 * * * docker exec bensbot logrotate /etc/logrotate.d/bensbot
   ```

3. **Performance Report**:
   ```bash
   # Add to crontab
   0 7 * * * docker exec bensbot python -m scripts.generate_performance_report
   ```

### Database Management

1. **Index Optimization**:
   ```bash
   # Connect to MongoDB
   docker exec -it mongodb mongosh
   
   # Create indexes for common queries
   use bensbot
   db.trades.createIndex({timestamp: -1})
   db.trades.createIndex({strategy_id: 1, timestamp: -1})
   db.performance.createIndex({metric_type: 1, timestamp: -1})
   db.system_logs.createIndex({timestamp: -1, level: 1})
   ```

2. **Data Archiving**:
   ```bash
   # Archive old data
   docker exec -it bensbot python -m scripts.archive_old_data --days 90
   ```

### System Updates

1. **Update Process**:
   ```bash
   # Pull latest code
   git pull
   
   # Build new Docker image
   docker build -t bensbot:latest .
   
   # Update running containers
   docker-compose down
   docker-compose up -d
   ```

2. **Kubernetes Updates**:
   ```bash
   # Update container image
   kubectl set image deployment/bensbot bensbot=bensbot:latest -n bensbot
   
   # Monitor rollout
   kubectl rollout status deployment/bensbot -n bensbot
   ```

3. **Rollback Process**:
   ```bash
   # Docker rollback
   docker tag bensbot:previous bensbot:rollback
   docker-compose down
   docker-compose up -d
   
   # Kubernetes rollback
   kubectl rollout undo deployment/bensbot -n bensbot
   ```
