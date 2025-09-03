-- BenBot Minimal Launcher.applescript
-- This script launches the BenBot trading dashboard with the minimal server

on run
	set projectPath to "/Users/bendickinson/Desktop/benbot"
	
	-- Display a startup message
	display dialog "Starting BenBot Trading Dashboard..." buttons {"OK"} default button "OK" with title "BenBot Launcher" with icon note giving up after 2
	
	-- Run the minimal launcher script
	tell application "Terminal"
		activate
		do script "cd " & quoted form of projectPath & " && ./start_benbot_minimal.sh"
	end tell
	
	-- Wait a moment for servers to start
	delay 5
	
	-- Notify user
	display notification "BenBot Trading Dashboard is now running" with title "BenBot Launcher" subtitle "Dashboard ready" sound name "Glass"
end run
