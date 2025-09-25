#!/bin/bash
# Setup PM2 to auto-start on system boot

echo "Setting up PM2 auto-start..."

# Generate startup script
sudo env PATH=$PATH:/opt/homebrew/Cellar/node/23.11.0/bin /opt/homebrew/lib/node_modules/pm2/bin/pm2 startup launchd -u bendickinson --hp /Users/bendickinson

# Save current process list
pm2 save

echo "✅ PM2 auto-start configured!"
echo "Your bot will now start automatically when your Mac boots up."
