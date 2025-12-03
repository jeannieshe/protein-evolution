import pandas as pd
import numpy as np

DMS = pd.read_csv("data/aav_dms.csv")

# DMS statistics
dms_mean = DMS['DMS_score'].mean()
dms_std = DMS['DMS_score'].std()
print(f"DMS: mean={dms_mean:.3f}, std={dms_std:.3f}")

# ESM statistics (sample a subset of 100 variants)
from protein_evolution.fitness_functions import esm_pseudo_log_likelihood

with open('data/aav_wt.txt', 'r') as f:
    wt = f.readline().strip()

esm_scores = []
for mut_seq in DMS['mutated_sequence'].sample(100):
    esm_scores.append(esm_pseudo_log_likelihood(wt, mut_seq))

esm_mean = np.mean(esm_scores)
esm_std = np.std(esm_scores)
print(f"ESM sampled 100 mutants: mean={esm_mean:.3f}, std={esm_std:.3f}")