import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import h5py as h5
import tensorflow as tf
import shap 
from sklearn.model_selection import train_test_split

from utils.hyper import SEED
from utils.load_save import load_PLM_embeddings
from utils.plotting import plot_bin_shap

np.random.seed(SEED)
np.set_printoptions(suppress=True)

modeldir = "./models/binary.keras"
embedfile = "./customdb/mucbp_perturb_embeddings_X.h5"

def main():
    tensor_train, label_train, _ = load_PLM_embeddings()
    train_x, _ , _ , _ = train_test_split(tensor_train, label_train, test_size=0.2, shuffle=True, stratify=label_train, random_state=SEED)

    model = tf.keras.models.load_model(modeldir)
    
    with h5.File(embedfile, 'r') as hf:
        tensor_source = hf['embeddings'][:]

    background_train = train_x[np.random.choice(train_x.shape[0], 1000, replace=False)]
    shap_model_source = shap.DeepExplainer(model, background_train)
    shap_val = shap_model_source.shap_values(tensor_source)
    shap_val = shap_val.squeeze()    
    
    preds = model.predict(tensor_source, verbose = 0)
    
    columns = ['Baseline', 'mucbp domain 1 masked', 'mucbp domain 2 masked', 'mucbp domain 3 masked', 'mucbp domain 4 masked','all domain masked']
    columns2 = ['Baseline', 'mucbp domain 1 masked', 'mucbp domain 2 masked', 'mucbp domain 3 masked', 'all domain masked']
    df = pd.DataFrame(shap_val.T, columns = columns + columns2)
    dfs = np.split(df, [6], axis=1)
    
    df1_melt = dfs[0].melt(var_name='Treatment', value_name='SHAP Value')
    df2_melt = dfs[1].melt(var_name='Treatment', value_name='SHAP Value')
    
    all_scores = []
    for n, i in enumerate(shap_val):
        all_scores.append(np.sum(i))
    #===========
    
    with open("./output/shap_bin_scores.txt", "w") as writefile:
        counter = 0
        for pred, score in zip(preds, all_scores):
            if counter < 6:
                header = "WP_014025352"
            else:
                header = "WP_014024830"
            
            writefile.write(f"{header}\t{pred[0]}\t{score}\n")
            counter += 1

    #plot
    plot_bin_shap(df1_melt, "./plot/SHAP_bin_WP_014025352.pdf")
    plot_bin_shap(df2_melt, "./plot/SHAP_bin_WP_014024830.pdf")
    
main()