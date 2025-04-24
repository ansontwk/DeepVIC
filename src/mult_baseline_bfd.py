import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.hyper import BASE_BATCHSIZE as BATCHSIZE, BASE_LR as LEARNRATE, MULT_BASE_PATIENCE as PATIENCE
from utils.load_save import load_PLM_embeddings
from utils.formatting import SMOTE_data
from utils.models import mult_DNN, build_mult, eval_mult
def main():
    plm_tensor, vf, cls = load_PLM_embeddings()
    train_x, train_y, val_x, val_y, val_label = SMOTE_data(plm_tensor, vf, cls)
    
    model = mult_DNN(train_x.shape[1])
    model = build_mult(train_x, train_y, val_x, val_y, BATCHSIZE, LEARNRATE, PATIENCE)
    
    report_df = eval_mult(model, val_x, val_label)
    report_df.to_csv("./tmp/baseline_BFD.tsv", sep='\t', index=True, header=True)
main()