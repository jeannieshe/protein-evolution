import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import cProfile
import pstats

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO

import sys
sys.path.append('/om/user/kspiv/protein-evolution')
sys.path.append('/om/user/kspiv/protein-evolution/protein_evolution')
sys.path.append('/om/user/kspiv/protein-evolution/protein_evolution/ppo')

from protein_evolution.fitness_functions import fitness_ESM, fitness_ESM_DMS
from protein_evolution.callbacks import *
from protein_evolution.environments import ProteinEnv

from datetime import datetime

NUM_ENVS = 64

if __name__ == "__main__":
    with open('data/aav_wt.txt', 'r') as file:
        wt = file.readline().strip()

    def make_env():
        # Provide your own initial sequence + fitness_fn
        return ProteinEnv(wt, fitness_ESM_DMS, 'data/aav_dms.csv')

    vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=6, 
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        verbose=1,
        device="cpu"
    )

    total_timesteps = 10000
    tqdm_cb = TQDMCallback(total_timesteps=total_timesteps, algo='PPO')
    logger_cb = ProteinRLLogger(check_freq=1)
    callback = CallbackList([tqdm_cb, logger_cb])
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("/om/user/kspiv/protein-evolution/ppo/ppo_01122026")