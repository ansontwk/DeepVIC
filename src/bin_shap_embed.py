import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers.utils import logging
logging.set_verbosity(40)
from transformers import TFBertModel, BertTokenizer
import numpy as np
import time
start_time = time.time()
from utils.embed_utils import centroid_vector, get_time, load_data_seqonly as load_data
from utils.load_save import save_emebed_only as save_file
from utils.paths import ProtBertBFD_path
#================================
tokenizer = BertTokenizer.from_pretrained(ProtBertBFD_path, do_lower_case=False)
model = TFBertModel.from_pretrained(ProtBertBFD_path, from_pt=True)

INPUTFILE = "./customdb/2mucbp.perturbX.faa"
OUTFILE = "./customdb/mucbp_perturb_embeddings_X.h5"

def main():
    seqs, _ = load_data(INPUTFILE)
    embeddings = []
    #print("Getting Embeddings")
    batchsize = 20

    for i in range(0, len(seqs), batchsize):
        batch_start_time = time.time()
                
        seq = seqs[i:i+batchsize]
        ids = tokenizer.batch_encode_plus(seq, add_special_tokens=True, padding=True, return_tensors="tf", truncation = True)

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
    
    embeddings = np.vstack(embeddings)
    print(embeddings.shape)
    
    print("Saving")
    save_file(embeddings, OUTFILE)

main()
end_time = time.time()
print(f"Total time elapsed: {end_time - start_time} seconds")