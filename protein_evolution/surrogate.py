from fitness_functions import esm_pseudo_log_likelihood, FastEmbedder
import torch, os
from torch import nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, r2_score

DMS_path = 'aav_dms.csv'
wt_path = 'aav_wt.txt'
wt_name = 'aav'

with open(wt_path, 'r') as file:
    wt = file.readline().strip()

class Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(320, 320),   # esm embedding is length 320
            nn.ReLU(),
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, embeddings):
        # embeddings = 16 x 320 or similar batch size x esm embedding size
        return self.linear_relu_stack(embeddings).squeeze(-1)   # shape [16]

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

        save_path = f'{wt_name}_embeddings.pkl'

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
    n_epochs = 10
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        i=0
        for batch_x, batch_y in train_loader:
            if i % 10 == 0:
                print(i)
            i+=1
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= n_train
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_loss:.4f}")
        metrics[f"Epoch {epoch+1} train loss"] = epoch_loss

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

    # Save metrics
    with open(f"{wt_name}_surrogate_metrics.pkl", 'wb') as file:
        pickle.dump(metrics, file)

    # Save model
    torch.save(model.state_dict(), DMS_path.replace(".csv","_surrogate.state_dict"))
    

