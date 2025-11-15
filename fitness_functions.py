import torch, esm
import numpy as np

# Load the pretrained ESM2 150M model
esm150_model, esm150_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
esm150_model.eval()

batch_converter = esm150_alphabet.get_batch_converter()
mask_idx = esm150_alphabet.mask_idx

def fitness_ESM(wt, mut, *args):
    ''' Calculate fitness solely based on ESM likelihoods. 
        We can use as a baseline, and to pre-train a policy to yield biologically feasible mutations (?)
     '''
    return esm_pseudo_log_likelihood(wt, mut)

def fitness_ESM_DMS(wt, mut, DMS):
    ''' Calculate fitness based on ESM as well as querying DMS dataset. 
        Assumes DMS has columns `mutated_sequence` and `DMS_score`
    '''
    ESM = esm_pseudo_log_likelihood(wt, mut)
    DMS_score = DMS.loc[DMS.mutated_sequence == mut].DMS_score.item()

    # for now, combine just by addition. can choose other ways, to weigh one over the other.
    return ESM + DMS_score

#### HELPERS
def esm_pseudo_log_likelihood(wt_seq, mut_seq):
    """
    Score only the mutated positions instead of entire protein.
    
    Args:
        wt_seq: wild-type sequence
        mut_seq: mutant sequence
        mut_positions: list of positions that differ (if None, auto-detect)
    
    Returns:
        float: log likelihood ratio (mutant - wild-type)
    """
    assert isinstance(wt_seq, str), 'WT is not a string'
    assert isinstance(mut_seq, str), 'Mutant is not a string'
    # find mutated positions if not provided
    mut_positions = [i for i in range(len(wt_seq)) if wt_seq[i] != mut_seq[i]]
    
    if len(mut_positions) == 0:
        return 0.0  # no mutations
    
    # tokenize both sequences
    _, _, wt_tokens = batch_converter([("wt", wt_seq)])
    _, _, mut_tokens = batch_converter([("mut", mut_seq)])
    wt_tokens = wt_tokens[0]
    mut_tokens = mut_tokens[0]
    
    device = next(esm150_model.parameters()).device
    wt_tokens = wt_tokens.to(device)
    mut_tokens = mut_tokens.to(device)
    
    total_log_ratio = 0.0
    
    # score each mutated position
    for pos_idx in mut_positions:
        # account for BOS token (position 0 is BOS, so add 1)
        token_pos = pos_idx + 1
        
        # mask wild-type at this position
        wt_masked = wt_tokens.clone()
        wt_masked[token_pos] = mask_idx
        
        # mask mutant at this position
        mut_masked = mut_tokens.clone()
        mut_masked[token_pos] = mask_idx
        
        with torch.no_grad():
            # get logits for both
            wt_logits = esm150_model(wt_masked.unsqueeze(0))["logits"][0, token_pos]
            mut_logits = esm150_model(mut_masked.unsqueeze(0))["logits"][0, token_pos]
            
            # compute log probs
            wt_log_probs = torch.log_softmax(wt_logits, dim=-1)
            mut_log_probs = torch.log_softmax(mut_logits, dim=-1)
            
            # get log prob of true amino acids
            wt_aa_log_prob = wt_log_probs[wt_tokens[token_pos]].item()
            mut_aa_log_prob = mut_log_probs[mut_tokens[token_pos]].item()
            
            # log likelihood ratio
            total_log_ratio += (mut_aa_log_prob - wt_aa_log_prob)
    
    return total_log_ratio


def esm_pseudo_log_likelihood_BAD(seq, batch_size=32):
    """
    Computes the pseudo log-likelihood of an amino-acid sequence using ESM2 (masked LM). 
    Returns a float.
    """
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens[0]  # shape [L]
    L = tokens.size(0)

    # generate all masked sequences (L-2 internal positions)
    masked_tokens = tokens.repeat(L-2, 1)
    positions = torch.arange(1, L-1)
    masked_tokens[torch.arange(L-2), positions] = mask_idx

    all_log_probs = []
    
    # process in batches to avoid OOM
    for batch_start in range(0, L-2, batch_size):
        batch_end = min(batch_start + batch_size, L-2)
        batch_masked = masked_tokens[batch_start:batch_end]
        batch_positions = positions[batch_start:batch_end]
        
        with torch.no_grad():
            logits = esm150_model(batch_masked)["logits"]  # [batch, L, vocab]
        
        # vectorized gathering of log probs at masked positions
        batch_idx = torch.arange(batch_end - batch_start)
        true_tokens = tokens[batch_positions]
        
        # get logits only at masked positions
        masked_logits = logits[batch_idx, batch_positions]  # [batch, vocab]
        log_probs_batch = torch.log_softmax(masked_logits, dim=-1)  # [batch, vocab]
        
        # gather true token log probs
        true_log_probs = log_probs_batch[batch_idx, true_tokens]  # [batch]
        all_log_probs.append(true_log_probs)
    
    return float(torch.cat(all_log_probs).sum().item())

