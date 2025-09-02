-- DirectLaunchDashboard.applescript
-- Script to launch the trading dashboard directly with Streamlit

tell application "Terminal"
    -- Open a new terminal window
    do script ""
    -- Navigate to the dashboard directory (using quoted path to handle the colon)
    do script "cd \"/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard\"" in front window
    -- Run the direct start dashboard script
    do script "./start_dashboard_direct.sh" in front window
end tell

-- Open the browser after a short delay
delay 3
do shell script "open http://localhost:8501"
