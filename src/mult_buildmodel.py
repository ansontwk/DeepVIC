
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import h5py as h5
from utils.models import mult_DNN, build_mult, eval_mult
from utils.hyper import MULT_BATCH as BATCHSIZE, MULT_LR as LEARNRATE, MULT_PATIENCE as PATIENCE

OUTFILE = "./models/multiclass.keras"

def main():

    with h5.File('./data/PSSM_smote_opt.h5', 'r') as hf:
        train_x = hf['train_x_smote'][:]
        train_y = hf['train_y_smote_2'][:]
        val_x = hf['val_x'][:]
        val_y = hf['val_y_2'][:]

    train_y= np.eye(14)[train_y]
    val_y_oh = np.eye(14)[val_y]
    
    model = mult_DNN(train_x.shape[1])
    model = build_mult(train_x, train_y, val_x, val_y_oh, BATCHSIZE, LEARNRATE, PATIENCE)
    model.save(OUTFILE) 
    
    report_df = eval_mult(model, val_x, val_y)
    report_df.to_csv(f"./tmp/opt_BFD_6.tsv", sep='\t', index=True, header=True)
main()
