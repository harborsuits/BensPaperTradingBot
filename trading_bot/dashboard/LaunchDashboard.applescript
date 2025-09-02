-- LaunchDashboard.applescript
-- Script to launch the trading dashboard

tell application "Terminal"
    -- Open a new terminal window
    do script ""
    -- Navigate to the dashboard directory
    do script "cd /Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard" in front window
    -- Run the start dashboard script
    do script "./start_dashboard.sh" in front window
    -- Minimize the terminal window (optional)
    -- set miniaturized of front window to true
end tell

-- Open the browser after a short delay
delay 5
do shell script "open http://localhost:8501"
