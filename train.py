import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3 import PPO


from fitness_functions import fitness_ESM, fitness_ESM_DMS
from callbacks import *
from environments import ProteinEnv

if __name__ == "__main__":
    with open('aav_wt.txt', 'r') as file:
        wt = file.readline().strip()

    def make_env():
        # Provide your own initial sequence + fitness_fn
        return ProteinEnv(wt, fitness_ESM, 'aav_dms.csv')

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512, 
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        clip_range=0.2,
        verbose=1,
        device="cpu"
    )

    total_timesteps = 20_000
    tqdm_cb = TQDMCallback(total_timesteps=total_timesteps, algo='PPO')
    logger_cb = ProteinRLLogger(check_freq=1)
    callback = CallbackList([tqdm_cb, logger_cb])
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_pretraining")