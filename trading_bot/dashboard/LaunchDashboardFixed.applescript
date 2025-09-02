-- LaunchDashboardFixed.applescript
-- Script to launch the trading dashboard with virtual environment in a safe location

tell application "Terminal"
    -- Open a new terminal window
    do script ""
    -- Navigate to the dashboard directory (using quoted path to handle the colon)
    do script "cd \"/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard\"" in front window
    -- Run the fixed setup and run script
    do script "./setup_and_run_fixed.sh" in front window
end tell

-- Open the browser after a longer delay to allow setup
delay 8
do shell script "open http://localhost:8501"
