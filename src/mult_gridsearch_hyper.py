import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import gc 
import tensorflow as tf
from keras import backend as K 
from tqdm import tqdm

from utils.hyper import MULT_LRS as LRS, MULT_PATIENCES as PATIENCES, BATCHSIZES
from utils.formatting import multimodal, SMOTE_data
from utils.models import mult_DNN as buildDNN, eval_mult

def main():
    
    tensor_concat, vf, cls = multimodal()
    train_smote_tensor, train_label_smote2_oh, val_tensor, val_label2_oh, val_label2 = SMOTE_data(tensor_concat, vf, cls)
    
    for BATCHSIZE in tqdm(BATCHSIZES):
        for LEARNRATE in tqdm(LRS, leave = False):
            for PATIENCE in tqdm(PATIENCES, leave = False):
                try:
                    combination_str = f"{BATCHSIZE}-{LEARNRATE}-{PATIENCE}"
                    model = buildDNN(tensor_concat.shape[1])
                    opt = tf.keras.optimizers.Adam(learning_rate = LEARNRATE)
                    model.compile(
                    optimizer = opt,
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = ['accuracy', 'AUC'])
                    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, start_from_epoch=5)
                    model.fit(train_smote_tensor, train_label_smote2_oh,
                            epochs = 50, 
                            validation_data=(val_tensor, val_label2_oh), 
                            callbacks= [callback], 
                            batch_size = BATCHSIZE,
                            verbose = 0)
                        
                    report_df = eval_mult(model, val_tensor, val_label2)
                    report_df.to_csv(f"./tmp/lr_gridsearch/{combination_str}.tsv", sep='\t', index=True, header=True)
                except:
                    with open("./tmp/lr_fail.txt", "a") as writefile:
                        writefile.write(f"{combination_str}\n")
                        
                K.clear_session()
                del model
                gc.collect()
main()
