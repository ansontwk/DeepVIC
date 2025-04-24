import numpy as np
import random
def checklist(nparray):
    unique, counts = np.unique(nparray, return_counts=True)
    print(dict(zip(unique, counts)))
    
def p_to_star(pval):
    if pval < 0.0001:
        return "****"
    elif pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval <= 0.05:
        return "*"
    else:
        return "NS"

#perturbation test
def allX(sequence, ROILIST, _):
    from utils.biological import AMINOLIST
    perturbations = []
    num_domains = len(ROILIST)
    
    for i in range(num_domains):
        len_domain = ROILIST[i][1] - ROILIST[i][0]
        perturbed = []
        for j in range(len_domain):
            perturbed.append(random.choice(AMINOLIST))
        perturbations.append(''.join(perturbed))
    
    allx = sequence
    for n, coord in enumerate(ROILIST):
        starting = coord[0]
        ending = coord[1]
        allx = allx[0:starting] + 'X'*(len(perturbations[n])) + allx[ending:]
    return allx
def X_domain(sequence, ROILIST, header):
    all_perturbed_seqs = []
    
    for domain in ROILIST:
        preturbedseq = replace_seq(sequence, domain[0], domain[1])
        all_perturbed_seqs.append(preturbedseq)
    
    return all_perturbed_seqs

def replace_seq(original_seq, roi_idx_start, roi_idx_roi_end):
    roi_idx_start = roi_idx_start - 1
    len_roi = roi_idx_roi_end - roi_idx_start
    
    preturbed = "X"*len_roi
    
    outseq = original_seq[0:roi_idx_start] + preturbed + original_seq[roi_idx_roi_end:]
    
    return outseq
    
