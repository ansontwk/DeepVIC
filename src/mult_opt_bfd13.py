import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from utils.hyper import MULT_BATCH as BATCHSIZE, MULT_LR as LEARNRATE, MULT_PATIENCE as PATIENCE
from utils.load_save import load_PLM_embeddings, load_all_PSSM
from utils.formatting import SMOTE_data
from utils.models import mult_DNN, build_mult, eval_mult
def main():
    plm_tensor, vf, cls = load_PLM_embeddings()
    pssm_vec = load_all_PSSM()
    tensor_concat = np.concatenate((plm_tensor, pssm_vec), axis=1)
    train_x, train_y, val_x, val_y, val_label = SMOTE_data(tensor_concat, vf, cls)
    
    model = mult_DNN(train_x.shape[1])
    model = build_mult(train_x, train_y, val_x, val_y, BATCHSIZE, LEARNRATE, PATIENCE)

    report_df = eval_mult(model, val_x, val_label)
    report_df.to_csv("./tmp/opt_BFD_13.tsv", sep='\t', index=True, header=True)
main()