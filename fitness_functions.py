import torch, esm
import numpy as np

# Load the pretrained ESM2 150M model
esm150_model, esm150_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
esm150_model.eval()

batch_converter = esm150_alphabet.get_batch_converter()
mask_idx = esm150_alphabet.mask_idx

def fitness_ESM(seq, *args):
    ''' Calculate fitness solely based on ESM likelihoods. 
        We can use as a baseline, and to pre-train a policy to yield biologically feasible mutations (?)
     '''
    return esm_pseudo_log_likelihood(seq)

def fitness_ESM_DMS(seq, DMS):
    ''' Calculate fitness based on ESM as well as querying DMS dataset. 
        Assumes DMS has columns `mutated_sequence` and `DMS_score`
    '''
    ESM = esm_pseudo_log_likelihood(seq)
    DMS_score = DMS.loc[DMS.mutated_sequence == seq].DMS_score.item()

    # for now, combine just by addition. can choose other ways, to weigh one over the other.
    return ESM + DMS_score

#### HELPERS
def esm_pseudo_log_likelihood(seq):
    """
    Computes the pseudo log-likelihood of an amino-acid sequence using ESM2 (masked LM). 
    Returns a float.
    """
    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)     # shape [1, L]
    tokens = tokens[0]                       # shape [L]
    L = tokens.size(0)

    # Generate all masked sequences (L-2 internal positions)
    masked_tokens = tokens.repeat(L-2, 1)
    positions = torch.arange(1, L-1)
    masked_tokens[torch.arange(L-2), positions] = mask_idx  # mask each pos

    # Add batch dimension
    masked_tokens = masked_tokens.unsqueeze(1)  # [L-2, 1, L]

    with torch.no_grad():
        logits = esm150_model(masked_tokens)["logits"]  # [L-2, 1, L, vocab]

    log_probs = []
    for i, pos in enumerate(positions):
        true_token = tokens[pos]
        log_prob_i = torch.log_softmax(logits[i, 0, pos], dim=-1)[true_token]
        log_probs.append(log_prob_i)

    return float(torch.stack(log_probs).sum().item())

