-- StopDashboard.applescript
-- Script to stop the trading dashboard

tell application "Terminal"
    -- Open a new terminal window
    do script ""
    -- Navigate to the dashboard directory
    do script "cd /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard" in front window
    -- Run the stop dashboard script
    do script "./stop_dashboard.sh" in front window
    -- Close terminal after 3 seconds
    delay 3
    close front window
end tell
