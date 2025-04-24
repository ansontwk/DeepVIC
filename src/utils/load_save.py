import h5py as h5
from Bio import SeqIO
import numpy as np
def loadfasta_binlabel(file):
    from utils.formatting import formatseq
    test_sequences = []
    test_headers = []
    test_label = []
    with open(file, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            seq = formatseq(str(record.seq))
            test_sequences.append(seq)
            test_headers.append(record.id)
            if record.id.startswith("DLDB"):
                test_label.append(1)
            else:
                test_label.append(0)
    return test_sequences, test_headers, test_label
def load_PLM_embeddings(PLMPATH = "./data/DLDB_CDHIT70_4db_33456_embeddings.h5"):
    from utils.formatting import decode, format_labelsVFC
    
    with h5.File(PLMPATH, 'r') as hf:
        tensor_array = hf['embeddings'][:]
        vf = hf['labels'][:]
        cls = hf['classes'][:]    
    cls = decode(cls)
    cls = format_labelsVFC(cls)
    return tensor_array, vf, cls

def load_all_PSSM(PSSMPATH = "./data/pssm_embeddings_13_v2.h5"):
    from utils.biological import FEATURES
    
    pssm_vec = []
    with h5.File(PSSMPATH, 'r') as hf:
        for feature in FEATURES:
            vec = hf[feature][:]
            vec = np.nan_to_num(vec, posinf=0, neginf=0)
            pssm_vec.append(vec)
    pssm_vec = np.concatenate(pssm_vec, axis=1)
    return pssm_vec

def load_opt_PSSM(PSSMPATH = "./data/pssm_embeddings_13_v2.h5"):
    from utils.biological import OPT_PSSM as opt
    
    pssm_vec = []
    with h5.File(PSSMPATH, 'r') as hf:
        for key, _ in hf.items():
            if key in opt:
                vec = hf[key][:]
                vec = np.nan_to_num(vec, posinf=0, neginf=0)
                pssm_vec.append(vec)
    pssm_vec = np.concatenate(pssm_vec, axis=1)
    return pssm_vec

def load_fasta_faa_user(infile):
    from utils.formatting import formatseq
    all_seqs = []
    all_headers = []
    with open(infile, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            seq = formatseq(str(record.seq))
            header = record.id
            all_seqs.append(seq)
            all_headers.append(header)
    return all_seqs, all_headers

def save_training_embed(embeddings, classes, subclass, labels, filepath):
    with h5.File(filepath, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings)
        hf.create_dataset("classes", data=classes)
        hf.create_dataset("subclass", data=subclass)
        hf.create_dataset("labels", data=labels)

def save_emebed_only(embeddings, filepath):
    with h5.File(filepath, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings)
