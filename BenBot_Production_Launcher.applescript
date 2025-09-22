#!/usr/bin/osascript

-- BenBot Production Launcher App
-- This creates a double-clickable app that launches the production trading bot

on run
    set appPath to (path to me) as text
    set containerPath to POSIX path of appPath
    
    -- Get the directory containing this app
    set AppleScript's text item delimiters to "/"
    set pathItems to text items of containerPath
    set pathItems to items 1 thru -2 of pathItems
    set basePath to pathItems as string
    set AppleScript's text item delimiters to ""
    
    -- Launch the shell script
    set scriptPath to basePath & "/BenBot_Production_Launcher.sh"
    
    tell application "Terminal"
        activate
        do script "cd " & quoted form of basePath & " && bash " & quoted form of scriptPath
    end tell
    
    display notification "Starting BenBot Trading Dashboard..." with title "BenBot Launcher"
end run
