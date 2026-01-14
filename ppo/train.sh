#!/bin/bash

# Load your shell environment to activate your Conda environment
source /home/kspiv/.bashrc
conda activate rl
cd /om/user/kspiv/protein-evolution

START_TIME=$(date +%s) # Get current time in seconds since epoch

echo "Running command..."
python ppo/train.py
echo "Command completed."

END_TIME=$(date +%s) # Get current time again

ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Total time taken: ${ELAPSED_TIME} seconds."
