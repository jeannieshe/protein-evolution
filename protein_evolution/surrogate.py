from fitness_functions import esm_pseudo_log_likelihood, FastEmbedder
import torch, os
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

torch.manual_seed(67)

import sys;sys.path.append('/om/user/kspiv/protein-evolution/protein_evolution/')
sys.path.append('/om/user/kspiv/protein-evolution/models/')
sys.path.append('/om/user/kspiv/protein-evolution/data/')
sys.path.append('/om/user/kspiv/protein-evolution/')

# DMS_path = '/om/user/kspiv/protein-evolution/data/aav_dms.csv'
# wt_path = '/om/user/kspiv/protein-evolution/data/aav_wt.txt'
# wt_name = 'aav'
DMS_path = '/om/user/kspiv/protein-evolution/data/Somermeyer2022_avGFP_dms_filtered.csv'
wt_path = '/om/user/kspiv/protein-evolution/data/avgfp_wt.txt'
wt_name = 'avgfp'

with open(wt_path, 'r') as file:
    wt = file.readline().strip()

class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(320, 320)
        self.l2 = nn.Linear(320, 128)
        self.l3 = nn.Linear(128, 1)
    
    def forward(self, embeddings):
        # embeddings = 16 x 320 or similar batch size x esm embedding size
        x = self.l1(embeddings)
        x = F.relu(x).square()
        x = self.l2(x)
        x = F.relu(x).square()
        x = self.l3(x)
        return x.squeeze(-1)   # shape [16]

class DMSDataset(Dataset):
    """
    PyTorch Dataset for DMS data with pre-computed embeddings.
    """
    def __init__(self, df, device="cpu", precompute=True):
        self.df = df.reset_index(drop=True)
        self.sequences = df["mutated_sequence"].tolist()
        self.scores = torch.tensor(df["DMS_score"].values, dtype=torch.float32)
        self.device = device

        self.embedder = FastEmbedder(wt, wt_name)
        
        if precompute:
            print("Pre-computing embeddings for all sequences...")
            self.embeddings = self._precompute_embeddings()
            print(f"Done! Cached {len(self.embeddings)} embeddings")
        else:
            self.embeddings = None
    
    def _precompute_embeddings(self):
        """Compute all embeddings once and cache them"""
        embeddings = []
        print('Within DMSDataset _precompute_embeddings: setting wildtype.')
        self.embedder.set_wildtype()

        save_path = f'/om/user/kspiv/protein-evolution/protein_evolution/{wt_name}_embeddings.pkl'

        if os.path.exists(save_path):
            print(f"Found {save_path}. Loading embeddings from pickle...")
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        else:
            print('Within DMSDataset _precompute_embeddings: no embeddings found. Doing it the long way.')
        
        # Process in batches for efficiency
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(self.sequences), batch_size):
                batch_seqs = self.sequences[i:i+batch_size]
                
                # Batch embedding computation
                batch_embeds = []
                for seq in batch_seqs:
                    emb = self.embedder.embed_mutant(seq)  
                    batch_embeds.append(emb)
                
                # Stack and move to device
                batch_embeds = torch.tensor(np.array(batch_embeds), 
                                           dtype=torch.float32).to(self.device)
                embeddings.append(batch_embeds)
                
                if i % 64 == 0:
                    print(f"  Processed {i}/{len(self.sequences)} sequences")
        
        # Concatenate all batches
        with open(save_path, 'wb') as file:
            pickle.dump(torch.cat(embeddings, dim=0), file)
        return torch.cat(embeddings, dim=0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            # Use cached embeddings - instant!
            x = self.embeddings[idx]
        else:
            # Fallback to on-the-fly computation
            x = torch.tensor(self.embedder.embed_mutant(self.sequences[idx]), 
                           dtype=torch.float32).to(self.device)
        
        y = self.scores[idx].to(self.device)
        return x, y

        
if __name__ == '__main__':
    df = pd.read_csv(DMS_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("NO CUDA WHATTT")

    # Create dataset
    dataset = DMSDataset(df, device=device)

    # Train/test split
    test_frac = 0.2
    n_test = int(len(dataset) * test_frac)
    n_train = len(dataset) - n_test
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    model = Surrogate().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    metrics = {}

    # Training loop
    n_epochs = 30
    from sklearn.model_selection import KFold

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    # Convert dataset into full arrays for kfold indices
    X = dataset.embeddings.cpu().numpy() if dataset.embeddings is not None else np.stack([dataset[i][0].cpu().numpy() for i in range(len(dataset))])
    y = dataset.scores.cpu().numpy()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")
        train_X, test_X = X[train_idx], X[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]

        # make TensorDatasets for loaders
        train_tensor = torch.tensor(train_X, dtype=torch.float32).to(device)
        train_targets = torch.tensor(train_y, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(test_X, dtype=torch.float32).to(device)
        test_targets = torch.tensor(test_y, dtype=torch.float32).to(device)
        train_dataset_fold = torch.utils.data.TensorDataset(train_tensor, train_targets)
        test_dataset_fold = torch.utils.data.TensorDataset(test_tensor, test_targets)

        train_loader = DataLoader(train_dataset_fold, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset_fold, batch_size=16, shuffle=False)

        model = Surrogate().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Train
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f"Fold {fold+1}, Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_loss:.4f}")

        # Eval
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = model(batch_x)
                all_preds.append(preds.cpu())
                all_targets.append(batch_y.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        mse = mean_squared_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        print(f"Fold {fold+1}: Test MSE: {mse:.4f}, R2: {r2:.4f}")

        # Compute Spearman correlation for train set
        all_train_preds, all_train_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                preds = model(batch_x)
                all_train_preds.append(preds.cpu())
                all_train_targets.append(batch_y.cpu())
        all_train_preds = torch.cat(all_train_preds).numpy()
        all_train_targets = torch.cat(all_train_targets).numpy()
        train_spearman, _ = spearmanr(all_train_targets, all_train_preds)
        print(f"Fold {fold+1}: Train Spearman: {train_spearman:.4f}")

        # Compute Spearman for test set
        test_spearman, _ = spearmanr(all_targets, all_preds)
        print(f"Fold {fold+1}: Test Spearman: {test_spearman:.4f}")

        fold_metrics.append({
            "Test MSE": mse,
            "R2": r2,
            "Train Spearman": train_spearman,
            "Test Spearman": test_spearman
        })

    metrics = {
        "fold_metrics": fold_metrics,
        "mean_test_mse": np.mean([m["Test MSE"] for m in fold_metrics]),
        "mean_r2": np.mean([m["R2"] for m in fold_metrics]),
        "mean_train_spearman": np.mean([m["Train Spearman"] for m in fold_metrics]),
        "mean_test_spearman": np.mean([m["Test Spearman"] for m in fold_metrics])
    }
    print("\n===== CV SUMMARY =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # ---------- Evaluation ----------
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")
    metrics["Test MSE"] = mse
    metrics["R2"] = r2

    # Compute Spearman correlation for train set
    all_train_preds, all_train_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            preds = model(batch_x)
            all_train_preds.append(preds.cpu())
            all_train_targets.append(batch_y.cpu())
    all_train_preds = torch.cat(all_train_preds).numpy()
    all_train_targets = torch.cat(all_train_targets).numpy()
    train_spearman, _ = spearmanr(all_train_targets, all_train_preds)
    print(f"Train Spearman: {train_spearman:.4f}")

    # Compute Spearman correlation for test set
    test_spearman, _ = spearmanr(all_targets, all_preds)
    print(f"Test Spearman: {test_spearman:.4f}")

    metrics["Train Spearman"] = train_spearman
    metrics["Test Spearman"] = test_spearman

    # Save metrics
    with open(f"{wt_name}_surrogate_metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    # Save model
    torch.save(model.state_dict(), DMS_path.replace(".csv","_surrogate.state_dict"))
    

