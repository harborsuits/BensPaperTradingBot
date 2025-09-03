-- Script to create a new BenBot Launcher app

tell application "Automator"
    set newDocument to make new document with properties {document type:"application"}
    
    -- Add a Run Shell Script action
    tell newDocument
        set shellAction to make new action with properties {name:"Run Shell Script"}
        tell shellAction
            set properties to {input:"to stdin", shell:"/bin/bash", source:"cd /Users/bendickinson/Desktop/benbot && ./BenBot_Fixed_Launcher.sh"}
        end tell
        add shellAction
        
        -- Save the document as an application
        save as "BenBot Launcher Fixed" in "/Users/bendickinson/Desktop"
    end tell
    
    -- Close Automator
    quit
end tell

-- Display a notification
tell application "System Events"
    display notification "New BenBot Launcher app has been created on your Desktop." with title "BenBot Launcher" sound name "Glass"
end tell
