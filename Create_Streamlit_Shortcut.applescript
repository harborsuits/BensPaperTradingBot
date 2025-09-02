-- AppleScript to create a desktop shortcut for the Trading Bot Dashboard with Streamlit

tell application "Finder"
	set desktopPath to path to desktop folder as string
	set projectPath to (path to desktop folder as string) & "Trading:"
	
	-- Create the applescript that will open the terminal and run the dashboard
	set appContent to "tell application \"Terminal\"
	activate
	do script \"cd " & projectPath & " && ./run_streamlit_dashboard.command\"
end tell"
	
	-- Save the script as an application on the desktop
	set appFile to (desktopPath & "Trading Dashboard (Streamlit).app")
	
	-- Check if application already exists
	if exists appFile as POSIX file then
		display dialog "The shortcut already exists on your desktop. Do you want to replace it?" buttons {"Cancel", "Replace"} default button "Replace"
		if button returned of result is "Cancel" then
			return
		else
			do shell script "rm -rf " & quoted form of (appFile as string)
		end if
	end if
	
	-- Create the new application
	tell application "Script Editor"
		activate
		make new document
		set text of document 1 to appContent
		tell document 1
			save as "application" in POSIX file (appFile as string)
		end tell
		quit
	end tell
	
	display dialog "Trading Dashboard (Streamlit) shortcut has been created on your desktop." buttons {"OK"} default button "OK"
end tell 