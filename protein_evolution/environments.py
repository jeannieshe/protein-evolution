import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class ProteinEnv(gym.Env):
    """
    This would be used in a real setting where you're not confined to a DMS dataset!
    State: amino acid sequence (string or int array)
    Action: mutate position i to amino acid j
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, seq, fitness_fn, DMS_path):
        ''' Requires the wild-type aa sequence (string), 
                fitness_fn (defined in fitness_functions.py),
            and DMS dataset (path to csv)
        '''
        super().__init__()
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        self.actions = []

        self.L = len(seq)
        self.fitness_fn = fitness_fn
        self.DMS = pd.read_csv(DMS_path)
        
        # convert sequence string → array of indices
        self.wt = np.array([self.aa_to_idx[a] for a in seq], dtype=np.int32)

        # action = choose a position to mutate, and choose an aa to mutate to
        # self.action_space = spaces.Discrete(self.L * 20)   # talk about indel limitation in write-up
        self.action_space = spaces.MultiDiscrete([588-561+1, 20])  # DMS dataset only mutates in positions 561-588

        # observation = vector of length L with values in [0,19]
        self.observation_space = spaces.MultiDiscrete([20] * self.L)

        self.state = None
        self.step_count = 0
    
    def idxs_to_letters(self, seq):
        ''' convert string of indexes to string of aa letters '''
        return ''.join([self.idx_to_aa[i] for i in seq])

    def _decode_action(self, action):
        pos, aa_idx = action
        pos += 560
        self.actions.append((pos, aa_idx))
        return pos, aa_idx

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.wt.copy()  # back to wild-type
        self.step_count = 0
        obs = self.state.copy()
        return obs, {}

    def step(self, action):
        pos, aa_idx = self._decode_action(action)
        print(f'Pos: {pos} and aa_idx: {aa_idx}')

        # Apply mutation
        new_state = self.state.copy()
        new_state[pos] = aa_idx

        # Reward from fitness function
        reward, dataset_used = self.fitness_fn(self.idxs_to_letters(self.wt), self.idxs_to_letters(new_state), self.DMS)

        # You can choose episode termination rule:
        # e.g., fixed length episode of mutations
        terminated = False # there are no natural terminal states for the model; the protein is never "done" evolving
        truncated = (self.step_count >= 3) # we want to artificially create a limit of 3 steps. 

        # Key distinction: truncated allows for future bootstrapping, meaning that the model could continue to receive reward in the future.

        self.state = new_state
        self.step_count += 1
        # print(self.idxs_to_letters(new_state))
        info = {'dataset_used': dataset_used, 
        'mutation_count': (self.wt != new_state),
        'num_mutations_per_variant': (self.wt != new_state).sum()}

        return new_state.copy(), reward, terminated, truncated, info

    def render(self):
        seq_str = "".join(self.idx_to_aa[i] for i in self.state)
        print(seq_str)
