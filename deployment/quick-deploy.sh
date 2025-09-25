#!/bin/bash
# Quick deployment script for Benbot to VPS

echo "🚀 Benbot Cloud Deployment Script"
echo "=================================="

# Check if server IP is provided
if [ -z "$1" ]; then
    echo "Usage: ./quick-deploy.sh <server-ip>"
    echo "Example: ./quick-deploy.sh 165.232.45.123"
    exit 1
fi

SERVER_IP=$1
SERVER_USER=${2:-root}

echo "📦 Preparing deployment package..."

# Create deployment directory
mkdir -p ~/Desktop/benbot-deploy
cd ~/Desktop/benbot

# Create .env file for production
cat > ~/Desktop/benbot-deploy/.env << EOF
# Benbot Production Environment
NODE_ENV=production
PORT=4000

# Trading Configuration
AUTOLOOP_ENABLED=1
STRATEGIES_ENABLED=1
AI_ORCHESTRATOR_ENABLED=1
AUTO_EVOLUTION_ENABLED=1
OPTIONS_ENABLED=1

# API Keys (Replace with your actual keys)
TRADIER_API_KEY=${TRADIER_API_KEY}
TRADIER_ACCOUNT_ID=${TRADIER_ACCOUNT_ID}
FINNHUB_API_KEY=${FINNHUB_API_KEY}
MARKETAUX_API_TOKEN=${MARKETAUX_API_TOKEN}

# Frontend
VITE_API_BASE=
VITE_USE_MSW=false
EOF

# Create PM2 ecosystem file
cat > ~/Desktop/benbot-deploy/ecosystem.config.js << 'EOF'
module.exports = {
  apps: [{
    name: 'benbot-backend',
    script: './live-api/minimal_server.js',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production'
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_file: './logs/combined.log',
    time: true
  }, {
    name: 'benbot-frontend',
    script: 'npx',
    args: 'vite preview --port 3003 --host',
    cwd: './new-trading-dashboard',
    instances: 1,
    autorestart: true,
    watch: false,
    env: {
      NODE_ENV: 'production'
    }
  }]
};
EOF

# Create setup script for server
cat > ~/Desktop/benbot-deploy/server-setup.sh << 'EOF'
#!/bin/bash
echo "🔧 Setting up Benbot on server..."

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install PM2 globally
sudo npm install -g pm2

# Install Nginx
sudo apt-get install -y nginx

# Create benbot user
sudo useradd -m -s /bin/bash benbot || true

# Setup directories
sudo mkdir -p /home/benbot/app
sudo mkdir -p /home/benbot/logs
sudo mkdir -p /home/benbot/data

# Set permissions
sudo chown -R benbot:benbot /home/benbot

echo "✅ Server setup complete!"
EOF

# Create nginx config
cat > ~/Desktop/benbot-deploy/nginx.conf << 'EOF'
server {
    listen 80;
    server_name _;

    # Frontend
    location / {
        proxy_pass http://localhost:3003;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:4000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket
    location /ws {
        proxy_pass http://localhost:4000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Tar the application (excluding node_modules)
echo "📦 Creating deployment package..."
tar -czf ~/Desktop/benbot-deploy/benbot.tar.gz \
    --exclude='node_modules' \
    --exclude='.git' \
    --exclude='data' \
    --exclude='logs' \
    --exclude='*.log' \
    .

echo "📤 Uploading to server..."
scp ~/Desktop/benbot-deploy/server-setup.sh $SERVER_USER@$SERVER_IP:/tmp/
scp ~/Desktop/benbot-deploy/benbot.tar.gz $SERVER_USER@$SERVER_IP:/tmp/
scp ~/Desktop/benbot-deploy/.env $SERVER_USER@$SERVER_IP:/tmp/
scp ~/Desktop/benbot-deploy/ecosystem.config.js $SERVER_USER@$SERVER_IP:/tmp/
scp ~/Desktop/benbot-deploy/nginx.conf $SERVER_USER@$SERVER_IP:/tmp/

echo "🔧 Running server setup..."
ssh $SERVER_USER@$SERVER_IP 'bash /tmp/server-setup.sh'

echo "📦 Deploying application..."
ssh $SERVER_USER@$SERVER_IP << 'ENDSSH'
# Extract application
cd /home/benbot/app
tar -xzf /tmp/benbot.tar.gz
cp /tmp/.env .
cp /tmp/ecosystem.config.js .

# Install dependencies
cd live-api && npm install --production
cd ../new-trading-dashboard && npm install && npm run build

# Setup Nginx
sudo cp /tmp/nginx.conf /etc/nginx/sites-available/benbot
sudo ln -sf /etc/nginx/sites-available/benbot /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

# Start with PM2
cd /home/benbot/app
pm2 start ecosystem.config.js
pm2 save
pm2 startup systemd -u benbot --hp /home/benbot

echo "✅ Deployment complete!"
echo "🌐 Access your bot at: http://$SERVER_IP"
ENDSSH

echo "
🎉 Deployment Complete!
=======================
📍 Server: http://$SERVER_IP
📊 Dashboard: http://$SERVER_IP
🔧 API: http://$SERVER_IP/api

To check status:
ssh $SERVER_USER@$SERVER_IP 'pm2 status'

To view logs:
ssh $SERVER_USER@$SERVER_IP 'pm2 logs'
"
