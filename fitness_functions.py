import torch, esm, pickle, os
import numpy as np
import torch.nn as nn

surrogate_path = 'aav_dms_surrogate.state_dict'

# Load the pretrained ESM2 8M model
esm8_model, esm8_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm8_model.eval()

batch_converter = esm8_alphabet.get_batch_converter()
mask_idx = esm8_alphabet.mask_idx

def fitness_ESM(wt, mut, *args):
    ''' Calculate fitness solely based on ESM likelihoods. 
        We can use as a baseline, and to pre-train a policy to yield biologically feasible mutations (?)
     '''
    return esm_pseudo_log_likelihood(wt, mut), 'ESM'

def fitness_ESM_DMS(wt, mut, DMS):
    ''' Calculate fitness based on ESM as well as querying DMS dataset. 
        Assumes DMS has columns `mutated_sequence` and `DMS_score`
    '''
    ESM = esm_pseudo_log_likelihood(wt, mut)
    if len(DMS.loc[DMS.mutated_sequence == mut].DMS_score) > 0:   # exists in dataset
        DMS_score = DMS.loc[DMS.mutated_sequence == mut].DMS_score.item()
        dataset_used = 'DMS'
    else:   # surrogate!
        DMS_score = surrogate(torch.tensor(embed(mut))).item()
        dataset_used = 'surrogate'
    # for now, combine just by addition. can choose other ways, to weigh one over the other.
    return ESM + DMS_score, dataset_used

#### HELPERS
def embed(seq):
    """ Compute ESM embedding (mean over all amino acids) of a protein """
    data = [("protein_id", seq)] 
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm8_model(batch_tokens, repr_layers=[6], return_contacts=False)

    token_embeddings = results["representations"][6][:, 1:-1, :][0].cpu().numpy()  # [L, 320]
    # remove start/end tokens with [:, 1:-1, :]

    sequence_embedding = token_embeddings.mean(axis=0)  # [320]
    return sequence_embedding # [320]

class FastEmbedder:
    def __init__(self, wt_seq, wt_name):
        self.wt_seq = wt_seq
        self.wt_name = wt_name
        self.wt_embedding = None
        self.position_embeddings = {}  # Cache per-position embeddings
    
    def set_wildtype(self):
        """Cache wildtype embedding and per-position contributions"""
        save_path = f'{self.wt_name}_wt_token_embeddings.pkl'
        if os.path.exists(save_path):
            print(f"Found {save_path}. Loading token-level embeddings from pickle...")
            with open(save_path, 'rb') as f:
                self.wt_token_embeddings = pickle.load(f)
                self.wt_embedding = self.wt_token_embeddings.mean(axis=0)  # [320]
                return 
        else:
            print('Within FastEmbedder set_wildtype: no token-level embeddings found. Doing it the long way. ')

        # Get full WT embedding
        data = [("wt", self.wt_seq)]
        _, _, batch_tokens = batch_converter(data)
        
        print('Within FastEmbedder set_wildtype: about to embed wt seq.')
        with torch.no_grad():
            results = esm8_model(batch_tokens, repr_layers=[6], return_contacts=False)
        
        token_embeddings = results["representations"][6][:, 1:-1, :]  # [1, L, 320]
        self.wt_token_embeddings = token_embeddings[0].cpu().numpy()  # [L, 320]
        self.wt_embedding = self.wt_token_embeddings.mean(axis=0)  # [320]

        print('Within FastEmbedder set_wildtype: saving token-level embeddings.')
        with open(save_path, 'wb') as file:
            pickle.dump(self.wt_token_embeddings, file)
    
    def embed_mutant(self, mut_seq):
        """
        Fast approximate embedding for mutant.
        Only re-embeds mutated positions, reuses WT for others.
        """
        return embed(mut_seq)

def esm_pseudo_log_likelihood(wt_seq, mut_seq):
    """Optimized version - reuse masked tensors"""
    assert isinstance(wt_seq, str) and isinstance(mut_seq, str)
    
    mut_positions = [i for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
    if len(mut_positions) == 0:
        return 0.0
    
    with torch.no_grad():
        _, _, wt_tokens = batch_converter([("wt", wt_seq)])
        _, _, mut_tokens = batch_converter([("mut", mut_seq)])
        wt_tokens = wt_tokens[0]
        mut_tokens = mut_tokens[0]
        
        device = next(esm8_model.parameters()).device
        wt_tokens = wt_tokens.to(device)
        mut_tokens = mut_tokens.to(device)
        
        # create masked versions ONCE, reuse them
        wt_masked = wt_tokens.clone()
        mut_masked = mut_tokens.clone()
        
        total_log_ratio = 0.0
        
        for pos_idx in mut_positions:
            token_pos = pos_idx + 1
            
            # save original values
            wt_orig = wt_masked[token_pos].item()
            mut_orig = mut_masked[token_pos].item()
            
            # mask in place
            wt_masked[token_pos] = mask_idx
            mut_masked[token_pos] = mask_idx
            
            # get logits
            wt_logits = esm8_model(wt_masked.unsqueeze(0))["logits"][0, token_pos]
            mut_logits = esm8_model(mut_masked.unsqueeze(0))["logits"][0, token_pos]
            
            # compute log probs
            wt_log_probs = torch.log_softmax(wt_logits, dim=-1)
            mut_log_probs = torch.log_softmax(mut_logits, dim=-1)
            
            # get scores
            wt_aa_log_prob = wt_log_probs[wt_orig].item()
            mut_aa_log_prob = mut_log_probs[mut_orig].item()
            
            total_log_ratio += (mut_aa_log_prob - wt_aa_log_prob)
            
            # restore original values for next iteration
            wt_masked[token_pos] = wt_orig
            mut_masked[token_pos] = mut_orig
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return total_log_ratio

class Surrogate(nn.Module):  # cannot import circularly from surrogate.py lol so copied here
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

surrogate = Surrogate()
if torch.cuda.is_available():
    surrogate.load_state_dict(torch.load(surrogate_path, weights_only=True))
else:
    surrogate.load_state_dict(torch.load(surrogate_path, weights_only=True, map_location=torch.device('cpu')))
surrogate.eval()
