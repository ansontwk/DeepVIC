import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
from tensorflow import keras
import h5py as h5
from src.utils.predict_utils import load_data_seqonly, get_PSSM_embeddings, ORDERED_OPT_PSSM as features, PSSM_TYPE as pssm_dict, bin2VF, mult2VF, write_file
from src.utils.paths import model_binary, model_multiclass
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cutoff', type = float)
parser.add_argument('-m', '--mode', type = str, required= True, default = 'b')
parser.add_argument('-i', '--input', type = str, required= True)
parser.add_argument('-s', '--silent', type = bool)
parser.add_argument("-o", "--output", type = str, required = True)
parser.add_argument("--pssmpath", type = str)
args = parser.parse_args()
CUTOFF = args.cutoff
mode = args.mode.lower()
seqfile = str(args.input)
verbose = args.silent
outpath = args.output
pssmpath = args.pssmpath
def main():
    try:
        _, headers = load_data_seqonly(seqfile)
    except:
        raise SystemExit(f'{seqfile} is missing or problematic. Please check your input file and try again.')
    
    with h5.File("./tmp/tmp.h5") as hf:
        embeddings = hf["embeddings"][:]
    
    if mode == "m":
        multiclass_model = keras.models.load_model(model_multiclass)
        pssm = get_PSSM_embeddings(headers, features, pssm_dict, pssmpath)
        pssm_embeddings = np.asarray(pssm)
        tensor_concat = np.concatenate((embeddings, pssm_embeddings), axis=1)
        #print(tensor_concat.shape)
        mult_pred = multiclass_model.predict(tensor_concat, verbose = 0)
        pred_mult_idx = mult2VF(mult_pred)
        write_file(outpath, headers, pred_mult_idx, verbose)

    elif mode == "b":
        model = keras.models.load_model(model_binary)
        bin_pred = model.predict(embeddings, verbose = 0)
        pred_bin_idx = bin2VF(bin_pred)
        write_file(outpath, headers, pred_bin_idx, verbose)
main()
