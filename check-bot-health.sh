#!/bin/bash
# Quick health check for BenBot

echo "🤖 BenBot Health Check"
echo "======================"

# Check PM2 processes
echo -e "\n📊 Process Status:"
pm2 status

# Check if backend is responding
echo -e "\n🔌 Backend API:"
if curl -s http://localhost:4000/api/health > /dev/null; then
    echo "✅ Backend is responding"
else
    echo "❌ Backend not responding"
fi

# Check recent trades
echo -e "\n📈 Recent Activity:"
TRADES=$(curl -s http://localhost:4000/api/trades | python3 -c "import json,sys; data=json.loads(sys.stdin.read()); print(f'Total trades: {len(data.get(\"items\", []))}')" 2>/dev/null)
echo "$TRADES"

# Check portfolio
echo -e "\n💰 Portfolio Status:"
curl -s http://localhost:4000/api/paper/account | python3 -c "
import json,sys
try:
    data = json.loads(sys.stdin.read())
    equity = data.get('balances', {}).get('total_equity', 0)
    print(f'Total Equity: \${equity:,.2f}')
except:
    print('Unable to fetch portfolio')
" 2>/dev/null

echo -e "\n✨ Bot is configured to auto-restart on crashes"
echo "🔄 To enable auto-start on boot, run: ./setup-autostart.sh"
