import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K 
import gc 
import numpy as np 
from sklearn.model_selection import train_test_split, StratifiedKFold
import h5py as h5
from utils.models import baseline_oh, baseline_plm, run_model_fit, get_performance
from utils.load_save import loadfasta_binlabel as loadfasta
from utils.formatting import pad, one_hot
from utils.load_save import load_PLM_embeddings
from utils.hyper import BASE_BATCHSIZE as BATCHSIZE, BASE_LR as LR, BIN_BASE_EPOCH as EPOCH
SEED = 179180
INFILE = "./data/DLDB_33456.faa"
OUTFILE = "./output/binbaseline.tsv"
def main():
    tensor_train, label_train, _ = load_PLM_embeddings()
    plm_tensor, _ , plm_label, _ = train_test_split(tensor_train, label_train, test_size=0.2, shuffle=True, stratify=label_train, random_state=SEED)


    test_seq, _, oh_label = loadfasta(INFILE)
    padded_sequences = []
    for seq in test_seq:
        padded_seq = pad(seq)
        encoded_seq = one_hot(padded_seq)
        padded_sequences.append(encoded_seq)
    oh_tensor = np.array(padded_sequences)
    x, _, y, _ = train_test_split(oh_tensor, oh_label, test_size=0.2, shuffle=True, stratify=oh_label, random_state=SEED)
    oh_train_x, _, oh_train_y, _ = train_test_split(x, y, test_size=0.25, shuffle=True, stratify=y, random_state=SEED)
    plm_train_x, _, plm_train_y, _ = train_test_split(plm_tensor, plm_label, test_size=0.25, shuffle=True, stratify=plm_label, random_state=SEED)
    oh_train_y = np.array(oh_train_y)
    
    result = {
        'OH': [],
        'PLM': []
    }
    #===========================================================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for i, (train_index, test_index) in enumerate(skf.split(oh_train_x, oh_train_y)):
        X_train, X_test = oh_train_x[train_index], oh_train_x[test_index]
        y_train, y_test = oh_train_y[train_index], oh_train_y[test_index]
        
        model = baseline_oh()
        model = run_model_fit(model, X_train, y_train, BATCHSIZE, LR, EPOCH)
        auc, _ = get_performance(model, X_test, y_test)
        result['OH'].append(auc)

        K.clear_session()
        gc.collect()
        
    for i, (train_index, test_index) in enumerate(skf.split(plm_train_x, plm_train_y)):
        X_train, X_test = plm_train_x[train_index], plm_train_x[test_index]
        y_train, y_test = plm_train_y[train_index], plm_train_y[test_index]
        
        model = baseline_plm()
        model = run_model_fit(model, X_train, y_train, BATCHSIZE, LR, EPOCH)
        auc, _ = get_performance(model, X_test, y_test)
        result['PLM'].append(auc)
        
        K.clear_session()
        gc.collect()
    #===========================================================
    
    mean_OH = np.mean(result['OH'])
    mean_PLM = np.mean(result['PLM'])
    with open(OUTFILE, 'w') as f:
        for i, j in zip(result['OH'], result['PLM']):
            f.write(f"{i}\t{j}\n")
        f.write(f"{mean_OH}\t{mean_PLM}\n")
main()
