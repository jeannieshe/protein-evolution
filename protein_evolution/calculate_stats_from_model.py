import pandas as pd
import sys
import pickle
import numpy as np

MODEL_PATH = '/om/user/kspiv/protein-evolution/logs/2025-12-09_19:01:01/ppo_metrics.pkl' # a .pkl file

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     MODEL_PATH = sys.argv[1]
    # else:
    #     print("Error: MODEL_PATH must be set. Pass it as an argument.")
    #     print("Usage: python calculate_stats_from_model.py [model_path]")
    #     sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        metrics = pickle.load(f)

    print(f"Calculating statistics from this path: {MODEL_PATH}")
    print(f"The average reward during the last rollout was {metrics['all_rewards'][-1]}.\nThe max reward during the last rollout was {metrics['top_rewards'][-1]}.")

    print(f"The ratio of surrogate:DMS queries during training was: {metrics['dataset_used'].count('surrogate')/metrics['dataset_used'].count('DMS')}")
    print(f"The position that was mutated the most was position {np.argmax(metrics['mutation_counts'])}, mutated {max(metrics['mutation_counts'])} times")

    print(f"The lines below can be pasted into the presentation:")
    print(f"Ratio: {metrics['dataset_used'].count('surrogate')/metrics['dataset_used'].count('DMS'):.4f}")
    print(f"Tuple: ({metrics['top_rewards'][-1]:.4f}, {metrics['all_rewards'][-1]:.4f})")
    print(f"Position: {np.argmax(metrics['mutation_counts'])}\nMutated times: {max(metrics['mutation_counts'])}")

# actions: list of length #timesteps? each item is a tuple (pos, aa_idx)
# dataset_used: list of length #timesteps? each item is either ‘DMS’ or ‘surrogate’ (representing if that variant required surrogate model or DMS lookup)
# mutation_counts: length 735 array of integers 0, 1, 2… representing how many times over all training that position was mutated
# num_mutations_per_variant: list of length #timesteps? each item is an int representing how many mutations exist per variant (like, 2 means that the variants has 2 different point mutations across that sequence)
# all_rewards and top_rewards: list of length #rollouts, representing rewards per rollout