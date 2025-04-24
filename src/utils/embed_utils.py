import pandas as pd
import numpy as np
from Bio import SeqIO
import h5py as h5


def embed_PLM(seqs, ProtBertBFD_path = "/mnt/A_16TB/Anson/programs/prot_bert_bfd"):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from transformers.utils import logging
    logging.set_verbosity(40)
    from transformers import TFBertModel, BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(ProtBertBFD_path, do_lower_case=False)
    model = TFBertModel.from_pretrained(ProtBertBFD_path, from_pt=True)
    
    embeddings = []
    batchsize = 20

    for i in range(0, len(seqs), batchsize):
        seq = seqs[i:i+batchsize]
        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, 
                                          padding='max_length', return_tensors="tf", 
                                          max_length = 2000, truncation = True)

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
def load_dict(file):
    dictionary = {}
    data = pd.read_csv(file, sep="\t")
    data = data.fillna("NA")
    for i, row in data.iterrows():
        header = row['DLDB header/Internal Reference']
        VFclass = row['VF class']
        VFsubclass = row['VF subclass']
        dictionary[header] = [VFclass, VFsubclass]        
    return dictionary

def centroid_vector(input_array):
    #calculates the centroid vector of the embedded sequence, returns a vector of size 1024
    output = np.mean(input_array, axis=0)
    output = np.array(output)
    return output

def get_data(record, dictionary):
    header = record.id
    if header in dictionary:
        VFclass, VFsubclass = dictionary[header]
    return VFclass, VFsubclass

def load_data(infile, dictionary):
    from utils.formatting import formatseq_BFD as formatseq
    all_seqs = []
    all_class = []
    all_subclass = []
    all_label = []
    
    with open(infile, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            seq = formatseq(str(record.seq))     
            if record.id.startswith("DLDB"):
                VFclass, VFsubclass = get_data(record, dictionary)
                VF = 1
            else:
                VFclass, VFsubclass = "NC", "NC"
                VF = 0
            
            all_seqs.append(seq)
            all_class.append(VFclass)
            all_subclass.append(VFsubclass)
            all_label.append(VF)
            
    return all_seqs, all_class, all_subclass, all_label
def load_data_seqonly(infile):
    from utils.formatting import formatseq_BFD as formatseq
    all_seqs = []
    all_headers = []
    with open(infile, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            seq = formatseq(str(record.seq))  
            header = record.id
            all_seqs.append(seq)
            all_headers.append(header)         
    return all_seqs, all_headers   

def get_time(batchsize, timeused, total_num, startindex):
    timeleft = timeused * (total_num - startindex) / batchsize
    timeleft = round(timeleft, 3)
    return timeleft

def save_file(embeddings, classes, subclass, labels, outfile_path):
    with h5.File(outfile_path, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings)
        hf.create_dataset("classes", data=classes)
        hf.create_dataset("subclass", data=subclass)
        hf.create_dataset("labels", data=labels)