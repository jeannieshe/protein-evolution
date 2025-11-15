from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

class TQDMCallback(BaseCallback):
    def __init__(self, total_timesteps: int, verbose: int = 0, algo='PPO'):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self.algo = algo

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc=f"Training {self.algo}")

    def _on_step(self) -> bool:
        # `self.model.num_timesteps` is updated by SB3 internally
        self.pbar.n = self.model.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.n = self.total_timesteps
            self.pbar.close()

class ProteinRLLogger(BaseCallback):
    """
    Callback to log protein RL metrics:
      - average reward per rollout
      - top-k sequence fitness
      - mutation frequency per position
    """
    def __init__(self, check_freq=1, save_path="./protein_logs", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.all_rewards = []
        self.top_rewards = []
        self.mutation_counts = None  # will initialize based on env L

    def _on_training_start(self):
        # Initialize mutation counts
        env = self.training_env.envs[0]  # assume DummyVecEnv
        self.L = env.L
        self.mutation_counts = np.zeros(self.L)

    def _on_rollout_end(self):
        # Called after each rollout (n_steps)
        env = self.training_env.envs[0]
        reward = np.mean(self.locals["rewards"])  # rollout mean reward
        self.all_rewards.append(reward)

        # Track top reward
        top_reward = np.max(self.locals["rewards"])
        self.top_rewards.append(top_reward)

        # Track mutation frequency
        state = env.state
        self.mutation_counts += (state != env.wt).astype(int)

        # Optional: print/log
        if self.n_calls % self.check_freq == 0 and self.verbose > 0:
            print(f"[Rollout {len(self.all_rewards)}] avg_reward={reward:.2f}, top_reward={top_reward:.2f}")

    def _on_training_end(self):
        import pickle
        with open('debug.pkl', 'wb') as file:
            dicty = {'mut': self.mutation_counts,
            'all_rewards': self.all_rewards,
            'top_rewards': self.top_rewards}
            pickle.dump(dicty, file)
        # Plot metrics
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(self.all_rewards, label="avg reward")
        plt.plot(self.top_rewards, label="top reward")
        plt.xlabel("Rollout")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Reward over training")

        plt.subplot(1,2,2)
        plt.bar(range(self.L), self.mutation_counts)
        plt.xlabel("Position")
        plt.ylabel("Mutation count")
        plt.title("Mutation frequency per position")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "protein_training_metrics.png"))
        plt.show()

    def _on_step(self) -> bool:
        # Required by BaseCallback; do nothing
        return True
