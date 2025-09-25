#!/bin/bash
# BenBot VPS Deployment Script
# Usage: ./deploy-to-vps.sh <server-ip>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server IP is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide server IP${NC}"
    echo "Usage: ./deploy-to-vps.sh <server-ip>"
    exit 1
fi

SERVER_IP=$1
SERVER_USER=${2:-ubuntu}
DEPLOY_PATH="/home/$SERVER_USER/benbot"

echo -e "${GREEN}🚀 Starting BenBot deployment to $SERVER_IP${NC}"

# Step 1: Create deployment archive
echo -e "${YELLOW}📦 Creating deployment archive...${NC}"
tar -czf benbot-deploy.tar.gz \
    --exclude='node_modules' \
    --exclude='*.log' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='data/*.json' \
    --exclude='config/credentials' \
    live-api new-trading-dashboard package.json

# Step 2: Copy to server
echo -e "${YELLOW}📤 Copying files to server...${NC}"
scp benbot-deploy.tar.gz $SERVER_USER@$SERVER_IP:/tmp/

# Step 3: Deploy on server
echo -e "${YELLOW}🛠️ Deploying on server...${NC}"
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
set -e

# Install dependencies if not present
if ! command -v node &> /dev/null; then
    echo "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

if ! command -v pm2 &> /dev/null; then
    echo "Installing PM2..."
    sudo npm install -g pm2
fi

# Create deployment directory
mkdir -p ~/benbot
cd ~/benbot

# Extract files
tar -xzf /tmp/benbot-deploy.tar.gz
rm /tmp/benbot-deploy.tar.gz

# Install dependencies
echo "Installing dependencies..."
cd ~/benbot
npm install

cd ~/benbot/live-api
npm install

cd ~/benbot/new-trading-dashboard
npm install
npm run build

# Create necessary directories
mkdir -p ~/benbot/live-api/logs
mkdir -p ~/benbot/live-api/data
mkdir -p ~/benbot/live-api/config/credentials

# Setup PM2
cd ~/benbot/live-api
pm2 delete benbot-api 2>/dev/null || true
pm2 start ecosystem.config.js

# Setup PM2 startup
pm2 startup systemd -u $USER --hp /home/$USER
pm2 save

# Setup nginx (if installed)
if command -v nginx &> /dev/null; then
    echo "Configuring nginx..."
    sudo tee /etc/nginx/sites-available/benbot << 'EOF'
server {
    listen 80;
    server_name _;

    # Frontend
    location / {
        root /home/ubuntu/benbot/new-trading-dashboard/dist;
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://localhost:4000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://localhost:4000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
EOF
    sudo ln -sf /etc/nginx/sites-available/benbot /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
fi

echo "✅ Deployment complete!"
ENDSSH

# Step 4: Create .env template
echo -e "${YELLOW}📝 Creating .env template...${NC}"
cat > .env.template << 'EOF'
# Tradier API (Paper Trading)
TRADIER_API_KEY=KU2iUnOZIUFre0wypgyOn8TgmGxI
TRADIER_ACCOUNT_ID=VA1201776

# Market Data APIs
FINNHUB_API_KEY=your_finnhub_key_here
MARKETAUX_API_TOKEN=your_marketaux_key_here

# Optional: Crypto exchange (if using)
# COINBASE_API_KEY=your_key_here
# COINBASE_API_SECRET=your_secret_here
EOF

echo -e "${GREEN}✅ Deployment script complete!${NC}"
echo -e "${YELLOW}⚠️  Next steps:${NC}"
echo "1. Copy .env.template to server: scp .env.template $SERVER_USER@$SERVER_IP:~/benbot/live-api/.env"
echo "2. SSH to server and update API keys in .env file"
echo "3. Restart PM2: pm2 restart benbot-api"
echo "4. Check logs: pm2 logs benbot-api"
echo "5. Access dashboard at: http://$SERVER_IP"

# Cleanup
rm -f benbot-deploy.tar.gz
