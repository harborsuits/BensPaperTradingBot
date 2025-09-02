#!/bin/bash
# This script sets temporary environment variables for Coinbase API testing
# and runs the read-only test script to ensure safety

# Set API credentials as environment variables (only for this script session)
export COINBASE_API_KEY="adb53c0e-35a0-4171-b237-a19fec741363"
export COINBASE_API_SECRET="eavv3nYSkAWN9kRS1xnBJLmXgN74plaOvWlmVJhOCjeBdK6XL4zlV5OKk+GaELoGwAGy/rEf+9RnOLxzF34LqQ=="

# Run the read-only test to verify connection without trading
echo "Running read-only Coinbase API test..."
python3 examples/coinbase_read_only_test.py

# Clear the environment variables when done
unset COINBASE_API_KEY
unset COINBASE_API_SECRET

echo "Test complete. Environment variables cleared."
