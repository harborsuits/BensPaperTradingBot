#!/bin/bash
# DO NOT COMMIT THIS FILE TO GIT!
# Run this script to set up your environment variables

echo "Setting up environment variables for BenBot..."

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
    echo "# BenBot Environment Variables" > .env
    echo "# DO NOT COMMIT THIS FILE!" >> .env
    echo "" >> .env
fi

# Function to add or update env var
add_env_var() {
    local key=$1
    local value=$2
    if grep -q "^${key}=" .env; then
        # Update existing
        sed -i '' "s|^${key}=.*|${key}=${value}|" .env
    else
        # Add new
        echo "${key}=${value}" >> .env
    fi
    echo "✓ Set ${key}"
}

# Trading APIs
echo "Setting up Trading APIs..."
# Add your keys here (replace the placeholders)
# add_env_var "TRADIER_TOKEN" "your_token_here"
# add_env_var "TRADIER_ACCOUNT_ID" "your_account_here"
add_env_var "TRADIER_BASE_URL" "https://sandbox.tradier.com/v1"

# News APIs (uncomment and add the ones you want to use)
echo "Setting up News APIs..."
# add_env_var "MARKETAUX_API_KEY" "your_key_here"
# add_env_var "NEWSDATA_API_KEY" "your_key_here"
# add_env_var "GNEWS_API_KEY" "your_key_here"

# System Settings
echo "Setting up System Settings..."
add_env_var "FORCE_NO_MOCKS" "true"
add_env_var "PAPER_MOCK_MODE" "false"

echo ""
echo "Environment setup complete!"
echo "Remember to:"
echo "1. Edit this file and add your actual API keys"
echo "2. Run: source .env"
echo "3. Restart your backend: pm2 restart benbot-backend"
echo ""
echo "SECURITY REMINDER:"
echo "- NEVER commit .env or this script with real keys to git"
echo "- Add .env to your .gitignore file"
