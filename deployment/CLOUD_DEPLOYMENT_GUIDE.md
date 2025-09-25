# BenBot Cloud Deployment Guide 🚀

## Quick Start (DigitalOcean/AWS/Vultr)

### 1. Create a VPS
- **Recommended Specs**: 2 CPU, 4GB RAM, 80GB SSD
- **OS**: Ubuntu 22.04 LTS
- **Cost**: ~$20-40/month

### 2. Initial Server Setup
```bash
# SSH into your server
ssh root@your-server-ip

# Create user
adduser ubuntu
usermod -aG sudo ubuntu

# Setup firewall
ufw allow 22
ufw allow 80
ufw allow 443
ufw enable

# Switch to ubuntu user
su - ubuntu
```

### 3. Deploy BenBot
From your local machine:
```bash
cd ~/Desktop/benbot
./deployment/deploy-to-vps.sh your-server-ip
```

### 4. Configure API Keys
```bash
# SSH to server
ssh ubuntu@your-server-ip

# Edit environment file
cd ~/benbot/live-api
nano .env

# Add your keys:
TRADIER_API_KEY=KU2iUnOZIUFre0wypgyOn8TgmGxI
TRADIER_ACCOUNT_ID=VA1201776
FINNHUB_API_KEY=your_key_here
MARKETAUX_API_TOKEN=your_key_here

# Save and exit (Ctrl+X, Y, Enter)

# Restart application
pm2 restart benbot-api
```

### 5. Monitor Your Bot
```bash
# View logs
pm2 logs benbot-api

# Check status
pm2 status

# Monitor resources
pm2 monit
```

## Cloud Provider Quick Links

### DigitalOcean (Easiest)
1. Create Droplet: https://cloud.digitalocean.com/droplets/new
2. Choose: Ubuntu 22.04, Basic, $24/mo (4GB RAM)
3. Add SSH key during creation

### AWS EC2
1. Launch Instance: https://console.aws.amazon.com/ec2/
2. Choose: Ubuntu 22.04, t3.medium
3. Configure Security Group (ports 22, 80, 443)

### Vultr
1. Deploy Server: https://my.vultr.com/deploy/
2. Choose: Ubuntu 22.04, $24/mo plan
3. Enable Auto-backups

## Monitoring & Maintenance

### Health Checks
```bash
# Check if bot is running
curl http://your-server-ip/api/health

# Check paper account
curl http://your-server-ip/api/paper/account

# View active strategies
curl http://your-server-ip/api/strategies/active
```

### Auto-restart on Crash
PM2 automatically restarts the bot if it crashes. The configuration includes:
- Memory limit: 2GB
- Auto-restart: Enabled
- Daily restart: 3 AM

### Backup Your Data
```bash
# Create backup script on server
cat > ~/backup-benbot.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf ~/benbot-backup-$DATE.tar.gz \
    ~/benbot/live-api/data \
    ~/benbot/live-api/logs \
    ~/benbot/live-api/.env
# Keep only last 7 days
find ~/ -name "benbot-backup-*.tar.gz" -mtime +7 -delete
EOF

chmod +x ~/backup-benbot.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * ~/backup-benbot.sh") | crontab -
```

## SSL Certificate (Optional)
```bash
# Install certbot
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
```

## Troubleshooting

### Bot not starting?
```bash
pm2 logs benbot-api --lines 100
```

### Orders not executing?
- Check market hours
- Verify API keys in .env
- Check account balance

### High memory usage?
```bash
pm2 restart benbot-api
```

## Cost Optimization

- Start with smallest VPS that meets requirements
- Use spot/reserved instances on AWS for savings
- Monitor actual resource usage after 1 week
- Scale up only if needed

## Security Notes

1. **Never commit .env file to git**
2. Use SSH keys, not passwords
3. Keep firewall enabled
4. Regular security updates:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## Next Steps

1. Monitor for 24-48 hours
2. Check logs for any errors
3. Verify trades are executing at market open
4. Set up alerts for failures (optional)
