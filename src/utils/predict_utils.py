import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from Bio import SeqIO
import h5py as h5

ORDERED_OPT_PSSM = ["aac_pssm", "d_fpssm", "edp", "k_separated_bigrams_pssm", "pssm_composition", "rpm_pssm"]
PSSM_TYPE = {"aac_pssm": {"default": 20,
                             "embeddings" : []},
                "ab_pssm" : {"default" : 400,
                              "embeddings" : []}, 
                "d_fpssm" : {"default": 20,
                              "embeddings" : []},
                "dpc_pssm" : {"default" : 400,
                               "embeddings" : []}, 
                "edp" : {"default" : 20,
                          "embeddings" : []}, 
                "eedp" : {"default" : 400,
                          "embeddings" : []}, 
                "k_separated_bigrams_pssm" : {"default" : 400,
                                             "embeddings" : []},
                "pse_pssm" : {"default" : 40,
                              "embeddings" : []},
                "pssm_composition" : {"default" : 400,
                                      "embeddings" : []}, 
                "rpm_pssm" : {"default" : 400,
                               "embeddings" : []}, 
                "rpssm" : {"default" : 110,
                            "embeddings" : []},
                "s_fpssm" : {"default" : 400,
                              "embeddings" : []},
                "tpc" : {"default" : 400,
                         "embeddings" : []}
                }
idx_to_VF = {
    0: "Adherence",
    1: "Invasion",
    2: "Effector Delivery System",
    3: "Motility",
    4: "Exotoxin",
    5: "Exoenzyme",
    6: "Immune modulation",
    7: "Biofilm",
    8: "Nutritional/metabolic factor",
    9: "Stress survival",
    10: "Regulation",
    11: "Post-translational modification",
    12: "Antimicrobial activity/competitive advantage",
    13: "Others"
}
idx_to_BIN = { 0: "Non-VF",
              1: "VF"}
def centroid_vector(input_array):
    #calculates the centroid vector of the embedded sequence, returns a vector of size 1024
    output = np.mean(input_array, axis=0)
    output = np.array(output)
    return output

def load_data_seqonly(infile):
    #from utils.formatting import formatseq_BFD as formatseq
    all_seqs = []
    all_headers = []
    with open(infile, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            seq = formatseq_BFD(str(record.seq))  
            header = record.id
            all_seqs.append(seq)
            all_headers.append(header)         
    return all_seqs, all_headers   

def formatseq_BFD(seq):
    seq = formatseq(seq)
    seq = " ".join(seq)
    return seq

def formatseq(seq):
    seq = seq.replace("B", "X").replace("Z", "X").replace("J", "X").replace("U", "X").replace("O", "X")
    return seq

def get_time(batchsize, timeused, total_num, startindex):
    timeleft = timeused * (total_num - startindex) / batchsize
    timeleft = round(timeleft, 3)
    return timeleft

def save_file(embeddings, outfile_path):
    with h5.File(outfile_path, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings)

#embed some data
def embed_PLM(seqs, ProtBertBFD_path = "/PATH/TO/ProtBertBFD"):
    from transformers.utils import logging
    logging.set_verbosity(40)
    from transformers import TFBertModel, BertTokenizer
    from src.utils.paths import ProtBertBFD_path
    tokenizer = BertTokenizer.from_pretrained(ProtBertBFD_path, do_lower_case=False)
    model = TFBertModel.from_pretrained(ProtBertBFD_path, from_pt=True)
    
    embeddings = []
    batchsize = 20

    for i in range(0, len(seqs), batchsize):
        seq = seqs[i:i+batchsize]
        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding='max_length', return_tensors="tf", max_length = 2000, truncation = True)

        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']

        embedding = model(input_ids)[0]
        embedding = np.asarray(embedding)
        attention_mask = np.asarray(attention_mask)

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][1:seq_len-1]
            seq_emd = centroid_vector(seq_emd)
            embeddings.append(seq_emd)
    
    embeddings = np.vstack(embeddings)

    return embeddings

def get_PSSM_embeddings(seqheaders, features, featuredict, feature_dir = "./tmp/features"):
    pssm_embeddings = []
    for header in seqheaders:
        embedding = []
        for key in features:
            file = f"{feature_dir}/{key}/{header}.csv"
            try: 
                if os.stat(file).st_size == 0:
                    raise FileNotFoundError   
                with open(file, "r") as readfile:
                    line = readfile.readline()
                    line = line.strip("\n")
                    data = line.split(",")
                    try:
                        dat2 = [float(i) for i in data]
                    except:
                        exit()
                    for vec in dat2:
                        embedding.append(vec)
            except FileNotFoundError:
                vecsize = featuredict[key]["default"]
                embedding.append([0] * vecsize)
        pssm_embeddings.append(embedding)
    return pssm_embeddings

def bin2VF(preds, cutoff = 0.537, dictionary = idx_to_BIN):
    bin_pred2 = [1 if pred >= cutoff else 0 for pred in preds]
    pred_bin_idx = [dictionary[pred] for pred in bin_pred2]

    return pred_bin_idx

def mult2VF(preds, dictionary = idx_to_VF):
    pred_mult_idx = [dictionary[np.argmax(pred)] for pred in preds]
    return pred_mult_idx

def write_file(path, headers, predictions, verbose):
    with open(path, "w") as writefile:
        for head, pred in zip(headers, predictions):
            if not verbose:
                print(head, pred)
            writefile.write(f"{head}\t{pred}\n")