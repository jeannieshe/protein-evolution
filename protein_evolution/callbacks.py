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
    def __init__(self, check_freq=1, save_path="./logs", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.all_rewards = []
        self.top_rewards = []
        self.actions = []
        self.dataset_used = []
        self.mutation_counts = None
        self.num_mutations_per_variant = []

    def _on_training_start(self):
        # Initialize mutation counts
        env = self.training_env.envs[0]  # assume DummyVecEnv
        self.L = env.L
        self.mutation_counts = np.zeros(self.L)

    def _on_rollout_end(self):
        # Called after each rollout (n_steps) - this is true katie checked
        self.actions += self.locals["actions"].tolist()
        self.dataset_used += list(info['dataset_used'] for info in self.locals['infos'])

        # env = self.training_env.envs[0]
        reward = np.mean(self.locals["rewards"])  # rollout mean reward
        self.all_rewards.append(reward)

        # Track top reward
        top_reward = np.max(self.locals["rewards"])
        self.top_rewards.append(top_reward)

        # Track mutation frequency
        for info in self.locals['infos']:
            self.mutation_counts += info['mutation_count']
            self.num_mutations_per_variant.append(info['num_mutations_per_variant'].item())

        # Optional: print/log
        if self.n_calls % self.check_freq == 0 and self.verbose > 0:
            print(f"[Rollout {len(self.all_rewards)}] avg_reward={reward:.2f}, top_reward={top_reward:.2f}")

    def _on_training_end(self):
        import pickle
        with open(os.path.join(self.save_path, "ppo_metrics.pkl"), 'wb') as file:
            dicty = {
                'actions': self.actions,
                'dataset_used': self.dataset_used,
                'mutation_counts': self.mutation_counts,
                'num_mutations_per_variant': self.num_mutations_per_variant,
                'all_rewards': self.all_rewards,
                'top_rewards': self.top_rewards
            }
            pickle.dump(dicty, file)

        # -----------------------
        # Rewards Over Training
        # -----------------------
        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.plot(self.all_rewards, label="avg reward")
        plt.plot(self.top_rewards, label="top reward")
        plt.xlabel("Rollout")
        plt.ylabel("Reward")
        plt.legend()
        plt.title("Reward over training")

        # -----------------------
        # Mutation Counts Per Position
        # -----------------------
        plt.subplot(1,3,2)
        # plt.bar(range(561,589), self.mutation_counts[560:588])
        try:
            plt.bar(range(283), self.mutation_counts)
            plt.xlabel("Position")
            plt.ylabel("Mutation count")
            plt.title("Mutation frequency per position")
        except:
            plt.plot(self.all_rewards, label="avg reward")
            plt.plot(self.top_rewards, label="top reward")
            plt.xlabel("Rollout")
            plt.ylabel("Reward")
            plt.legend()
            plt.title("Reward over training")

        # -----------------------
        # Distribution: Mutations per Variant
        # -----------------------
        plt.subplot(1,3,3)
        # import pdb;pdb.set_trace()
        plt.hist(self.num_mutations_per_variant, bins=range(0, max(self.num_mutations_per_variant)+2), align='left')
        plt.xlabel("Mutations per variant")
        plt.ylabel("Frequency")
        plt.title("Numbers of mutations per variant")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "ppo_metrics.png"))
        plt.close()

        # -----------------------
        # 20 × 28 Heatmap of Actions
        # -----------------------
        # heat = np.zeros((20, 28), dtype=int)
        heat = np.zeros((20, 238), dtype=int)
        for (pos, aa_idx) in self.actions:
            heat[aa_idx, pos] += 1

        plt.figure(figsize=(10,6))
        plt.imshow(heat, aspect='auto', origin='lower')
        plt.colorbar(label="Mutation Count")
        # plt.xlabel("Position (561 → 588)")
        plt.xlabel("Position (1-238")
        plt.ylabel("AA Index (0–19)")

        # X-axis ticks → actual sequence positions
        # positions = np.arange(561, 588 + 1)
        positions = np.arange(1, 238 + 1)
        # plt.xticks(np.arange(28), positions, rotation=90)
        plt.xticks(np.arange(238), positions, rotation=90)

        # Y-axis ticks → amino acids
        aa_labels = [self.locals['env'].envs[0].idx_to_aa[i] for i in range(20)]
        plt.yticks(np.arange(20), aa_labels)

        plt.title("Heatmap of Mutations (Amino Acid × Position)")
        plt.savefig(os.path.join(self.save_path, "action_heatmap.png"))
        plt.close()


    def _on_step(self) -> bool:
        # Required by BaseCallback; do nothing
        return True