import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
import h5py as h5
import time
from utils.hyper import BIN_LR as LR, BIN_BATCH as BATCHSIZE, BIN_EPOCH as EPOCH, BIN_CUTOFF as CUTOFF
from utils.models import baseline_plm, run_model_fit
from utils.load_save import load_PLM_embeddings
start_time = time.time()
SEED = 179180
OUTMODEL = "./models/binary.keras"
def main():
    tensor_train, label_train, _ = load_PLM_embeddings()
    plm_tensor, _ , plm_label, _ = train_test_split(tensor_train, label_train, 
                                                              test_size=0.2, shuffle=True, 
                                                              stratify=label_train, 
                                                              random_state=SEED)
    
    model = baseline_plm()
    model = run_model_fit(model, plm_tensor, plm_label, BATCHSIZE, LR, EPOCH)
    model.save(OUTMODEL)
main()