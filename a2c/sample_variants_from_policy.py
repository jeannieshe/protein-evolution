import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import sys

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

from protein_evolution.fitness_functions import fitness_ESM_DMS
from protein_evolution.environments import ProteinEnv

from datetime import datetime

MODEL_PATH = 'a2c/a2c_pretraining'
NUM_VARIANTS = 10

if __name__ == "__main__":
    # Allow model path to be passed as command line argument
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    
    if not MODEL_PATH:
        print("Error: MODEL_PATH must be set. Either edit the script or pass it as an argument.")
        print("Usage: python sample_variants_from_policy.py [model_path]")
        sys.exit(1)
    
    # Load wild-type sequence
    with open('data/aav_wt.txt', 'r') as file:
        wt = file.readline().strip()
    
    # Load DMS dataset for fitness computation
    DMS = pd.read_csv('data/aav_dms.csv')
    
    # Create a single environment for sampling
    def make_env():
        return ProteinEnv(wt, fitness_ESM_DMS, 'data/aav_dms.csv')
    
    env = make_env()
    
    # Load the trained model
    print(f"Loading model from {MODEL_PATH}...")
    model = A2C.load(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Sample variants
    variants = []
    print(f"\nSampling {NUM_VARIANTS} variants from the policy...")
    
    for i in range(NUM_VARIANTS):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        # Run episode until truncated (6 steps)
        while not truncated:
            action, _states = model.predict(obs, deterministic=False)  # Use deterministic=False for sampling
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract final sequence
        final_seq = env.idxs_to_letters(env.state)
        variants.append(final_seq)
        print(f"Variant {i+1}/{NUM_VARIANTS}: {final_seq}")
    
    # Compute fitness for each variant
    print("\n" + "="*80)
    print("Computing fitness scores...")
    print("="*80)
    
    results = []
    for i, variant_seq in enumerate(variants):
        fitness, dataset_used = fitness_ESM_DMS(wt, variant_seq, DMS)
        results.append({
            'variant_id': i + 1,
            'sequence': variant_seq,
            'fitness': fitness,
            'dataset_used': dataset_used
        })
        print(f"\nVariant {i+1}:")
        print(f"  Sequence: {variant_seq}")
        print(f"  Fitness: {fitness:.4f}")
        print(f"  Dataset used: {dataset_used}")
    
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = f'logs/{formatted_date}_sampled_variants.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    print(f"{'='*80}")


