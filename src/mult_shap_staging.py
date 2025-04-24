import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers.utils import logging
logging.set_verbosity(40)
from transformers import TFBertModel, BertTokenizer

import numpy as np
import h5py as h5
import time
from tqdm import tqdm

from utils.biological import OPT_PSSM as opt, PSSM_TYPE as feature_type
from utils.embed_utils import centroid_vector, load_data_seqonly as load_data, get_time
from utils.load_save import load_opt_PSSM
from utils.paths import ProtBertBFD_path
start_time = time.time()

INPUTFILE = "./customdb/2mucbp.faa"
INTERMEDIATE = "./customdb/2mucbp_multimodal.h5"
OUTFILE = "./customdb/final_multimodal_embeddings.h5"
tokenizer = BertTokenizer.from_pretrained(ProtBertBFD_path, do_lower_case=False)
model = TFBertModel.from_pretrained(ProtBertBFD_path, from_pt=True)
feature_dir = "./tmp/customdb/features"

def main():
    print("Loading Data")
    seqs, seqsheaders = load_data(INPUTFILE)
    embeddings = []
    batchsize = 20

    for i in range(0, len(seqs), batchsize):
        batch_start_time = time.time()
                
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
        
        #Get time
        batch_end_time = time.time()
        timeused = batch_end_time - batch_start_time
        timeleft = get_time(batchsize, timeused, len(seqs), i)
        
        if i % 1000 == 0:
            print(f"Embeddings for batch:\t{i}\tretrieved")
            print(f"Index Numbers:\t{i}\t{i+batchsize}.")
            print(f"Time used: {timeused} seconds, time left: {timeleft} seconds")
    tensor_array = np.vstack(embeddings)

    #PSSM
    for n, header in tqdm(zip(range(len(seqsheaders)),seqsheaders), total=len(seqsheaders)):
        for key, value in feature_type.items():            
            if not header.startswith("DLDB"):
                try:
                    header = header.split("|")[1]
                except:
                    header = header

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
                    assert len(dat2) == value["default"]
                    value["embeddings"].append(dat2)
                         
            except FileNotFoundError:
                vecsize = value["default"]
                defaultvec = [0] * vecsize
                value['embeddings'].append(defaultvec)
                
            if n!= len(feature_type[key]["embeddings"]) - 1 :
                print("error")
                break
        
    #print("Finished Reading.")
    with h5.File(INTERMEDIATE, "w") as hf:
        for key, value in feature_type.items():
            embedding = value["embeddings"]
            embedding = np.array(embedding)
            hf.create_dataset(key, data=embedding)
    
    pssm_vec = load_opt_PSSM(INTERMEDIATE)
    tensor_concat = np.concatenate((tensor_array, pssm_vec), axis=1)

    with h5.File(OUTFILE, "w") as hf:
        hf.create_dataset("embeddings", data=tensor_concat)
main()
end_time = time.time()
print(f"Total time elapsed: {round(end_time - start_time, 3)} seconds")