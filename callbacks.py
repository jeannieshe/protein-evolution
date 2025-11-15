from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

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