#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=sample
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=a2c/logs/%j-%x.out

# Load your shell environment to activate your Conda environment
source /home/jeanshe/.bashrc
conda activate plm
cd /home/jeanshe/orcd/pool/protein-evolution

START_TIME=$(date +%s) # Get current time in seconds since epoch

echo "Running command..."
python a2c/sample_variants_from_policy.py
echo "Command completed."

END_TIME=$(date +%s) # Get current time again

ELAPSED_TIME=$((END_TIME - START_TIME))
echo "Total time taken: ${ELAPSED_TIME} seconds."
