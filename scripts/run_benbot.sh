#!/bin/bash
# Run script for BensBot with proper environment

# Activate the virtual environment
source "/Users/bendickinson/benbot_env/bin/activate"

# Set Python path to include current directory
export PYTHONPATH=$PYTHONPATH:/Users/bendickinson/Desktop/Trading:BenBot

# Function to run a Python script
run_script() {
  echo "Running $1..."
  python $1 $2 $3 $4 $5
}

# Run requested script
if [ -z "$1" ]; then
  echo "Please specify a script to run, e.g.:"
  echo "./run_benbot.sh app.py"
  echo "./run_benbot.sh test_autonomous_core.py"
  echo "./run_benbot.sh demo_autonomous_pipeline.py"
else
  run_script $1 $2 $3 $4 $5
fi
