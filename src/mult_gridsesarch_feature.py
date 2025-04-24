import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import gc 
import tensorflow as tf
from keras import backend as K 
import numpy as np
import h5py as h5
import pandas as pd
from tqdm import tqdm
import itertools
import sklearn.metrics as skl_metrics
from utils.formatting import SMOTE_data
from utils.biological import FEATURES
from utils.models import mult_DNN as buildDNN
from utils.load_save import load_PLM_embeddings
from utils.hyper import BASE_BATCHSIZE as BATCHSIZE, BASE_LR as LEARNRATE

def main():
    tensor_array, vf, cls = load_PLM_embeddings()
    
    for combination in tqdm(itertools.product([True, False], repeat=len(FEATURES))):
        combination_str = [FEATURES[i] for i, include in enumerate(combination) if include]
        if combination_str:
            outstring = "-".join(combination_str)
            if os.path.exists(f"./tmp/gridsearch_features/{outstring}.tsv"):
                #print("Skipping, file already exist", outstring)
                continue
            
            pssm_vec = []
            with h5.File("./data/pssm_embeddings_13_v2.h5", 'r') as hf:
                for feature in combination_str:
                    vec = hf[feature][:]
                    vec = np.nan_to_num(vec, posinf=0, neginf=0)
                    pssm_vec.append(vec)
            pssm_vec = np.concatenate(pssm_vec, axis=1)
            tensor_concat = np.concatenate((tensor_array, pssm_vec), axis=1)
            
            train_smote_tensor, train_label_smote2_oh, val_tensor, val_label2_oh, val_label2 = SMOTE_data(tensor_concat, vf, cls)
            try:
                model = buildDNN(tensor_concat.shape[1])
                opt = tf.keras.optimizers.Adam(learning_rate =LEARNRATE)
                model.compile(optimizer = opt, loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy', 'AUC'])
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, start_from_epoch=5)
                
                model.fit(train_smote_tensor, train_label_smote2_oh,
                        epochs = 50, 
                        validation_data=(val_tensor, val_label2_oh), 
                        callbacks= [callback], 
                        batch_size = BATCHSIZE,
                        verbose = 0)
                
                pred = model.predict(val_tensor, verbose = 0)
                pred_idx = [np.argmax(val) for val in pred]
                report = skl_metrics.classification_report(val_label2, pred_idx, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(f"./tmp/gridsearch_features/{outstring}.tsv", sep='\t', index=True, header=True)
            except:
                with open("./tmp/grid_fail.txt", "a") as writefile:
                    writefile.write(f"{combination_str}\n") 
        K.clear_session()
        del model
        gc.collect()
main()