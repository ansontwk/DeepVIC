import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import h5py as h5
import shap 
import tensorflow as tf

from utils.hyper import SEED
from utils.plotting import mult_confu_lab as classlab, plot_mult_shap as plot_combined
from utils.embed_utils import load_data_seqonly as load_data
from utils.others import p_to_star
from utils.stats import kwtest, dunn
np.random.seed(SEED)
np.set_printoptions(suppress=True)

testseq = "./customdb/2mucbp.faa"
modeldir = "./models/multiclass.keras"
embedfile = "./customdb/final_multimodal_embeddings.h5"

def main():
    
    _, testheader = load_data(testseq)
    model = tf.keras.models.load_model(modeldir)

    with h5.File("./data/PSSM_smote_opt.h5", 'r') as hf:
        train_x = hf['train_x_smote'][:]

    with h5.File(embedfile, 'r') as hf:
        tensor = hf['embeddings'][:]

    pred_working = model.predict(tensor, verbose = 0)

    background_train = train_x[np.random.choice(train_x.shape[0], 1000, replace=False)]
    shap_model = shap.DeepExplainer(model, background_train)
    shap_values = shap_model.shap_values(tensor)
    testframe = pd.DataFrame()
    for shapval, header, pred_softmax in zip(shap_values, testheader, pred_working):
        data = pd.DataFrame(shapval, columns = classlab)

        testframe[f"{header}_BFD"] = data.iloc[:1025].sum(axis=0).to_list()
        testframe[f"{header}_PSSM"] = data.iloc[1026:].sum(axis=0).to_list()

        plot_combined(data, classlab, f"./plot/mult_shap_{header}.pdf", pred_softmax)
        
        kwh, kwpval, kwfreedom = kwtest(data)
        print(f"Sample: {header}, H-statistic: {kwh}, pval: {kwpval} ({p_to_star(kwpval)}), freedom: {kwfreedom}")
        if kwpval < 0.05:
            dunps = dunn(data, classlab)
            print(dunps.map(p_to_star))
                    
    #save shap contribution values        
    testframe.to_csv("./output/mult_shap_contribution.csv", index = False)
main()