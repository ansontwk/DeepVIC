import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K 
import gc 
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils.hyper import BATCHSIZES, LRS, EPOCHS
from utils.models import baseline_plm, run_model_fit, get_performance
from utils.load_save import load_PLM_embeddings
start_time = time.time()
SEED = 179180
def main():
    tensor_train, label_train, _ = load_PLM_embeddings()
    plm_tensor, _ , plm_label, _ = train_test_split(tensor_train, label_train, test_size=0.2, shuffle=True, stratify=label_train, random_state=SEED)
    
    eval = {}
    #print("Start nested CV")
    #===========================================================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for i, (train_index, test_index) in enumerate(skf.split(plm_tensor, plm_label)):
        print("running Fold ", i)
        hyperparam_results = {}
        out_X_train, out_y_train = plm_tensor[train_index], plm_label[train_index]
        out_X_test, out_y_test = plm_tensor[test_index], plm_label[test_index]

        for j, (train_index, test_index) in enumerate(skf2.split(out_X_train, out_y_train)):
            #print("Running inner fold")
            X_train, X_val = out_X_train[train_index], out_X_train[test_index]
            y_train, y_val = out_y_train[train_index], out_y_train[test_index]
            #print("looping through hyperparameters")
            for BATCHSIZE in BATCHSIZES:
                for LR in LRS:
                    for EPOCH in EPOCHS:
                        dictkey = f"{BATCHSIZE}_{LR}_{EPOCH}"
                        if dictkey not in hyperparam_results:
                            hyperparam_results[dictkey] = []
                        
                        model = baseline_plm()
                        model = run_model_fit(model, X_train, y_train, BATCHSIZE, LR, EPOCH)
                        auc, youden = get_performance(model, X_val, y_val)
                        hyperparam_results[dictkey].append([auc, youden])
                        K.clear_session()
                        gc.collect()
        #get best model
        avg_hyperparam_results = {}
        for dictkey, values in hyperparam_results.items():
            avg_auc = np.mean([value[0] for value in values])
            avg_youden = np.mean([value[1] for value in values])
            print(f"AUC for {dictkey}: {avg_auc}, YoudenJ for {dictkey}: {avg_youden}")
            avg_hyperparam_results[dictkey] = [avg_auc, avg_youden]
        
        # Get best hyperparameter based on average AUC
        best_combo = max(avg_hyperparam_results, key=lambda x: avg_hyperparam_results[x][0])        
        # Train model with best hyperparameter on outer fold
        best_batch = int(best_combo.split("_")[0])
        best_lr = float(best_combo.split("_")[1])
        best_epoch = int(best_combo.split("_")[2])
        
        outer_model = baseline_plm()
        outer_model = run_model_fit(outer_model, out_X_train, out_y_train, best_batch, best_lr, best_epoch)
        outer_auc, outer_youden = get_performance(outer_model, out_X_test, out_y_test)
        eval[i] = [outer_auc, outer_youden]
        #print(f"optimal hyper: AUC for outer fold {i}: {outer_auc}, YoudenJ for outer fold {i}: {outer_youden}")
        
        with open("./output/bin_55_CV_eval.tsv", 'a') as f:
            f.write(f"{i}\t{outer_auc}\t{outer_youden}\n")
        K.clear_session()
        gc.collect()
    
    with open("./output/bin_55_CV_eval.json", 'w') as f:
        json.dump(eval, f)
        
    #===========================================================    
    #cross validation to final hyperparam
    #print("start CV for final model")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    final = {}
    for i, (train_index, test_index) in enumerate(skf.split(plm_tensor, plm_label)):
        #print("running Fold ", i)
        X_train, y_train = plm_tensor[train_index], plm_label[train_index]
        X_test, y_test = plm_tensor[test_index], plm_label[test_index]
        for BATCHSIZE in BATCHSIZES:
                for LR in LRS:
                    for EPOCH in EPOCHS:
                        dictkey = f"{BATCHSIZE}_{LR}_{EPOCH}"
                        if dictkey not in final:
                            final[dictkey] = []
                        
                        model = baseline_plm()
                        model = run_model_fit(model, X_train, y_train, BATCHSIZE, LR, EPOCH)
                        auc, youden = get_performance(model, X_test, y_test)
                        final[dictkey].append([auc, youden])
                        K.clear_session()
                        gc.collect()
    
    #get the final hyperparmeter
    final_avg = {}
    for dictkey, values in final.items():
        avg_auc = np.mean([value[0] for value in values])
        avg_youden = np.mean([value[1] for value in values])
        final_avg[dictkey] = [avg_auc, avg_youden]
                        
    best_combo_final = max(final_avg, key=lambda x: final_avg[x][0])
    
    
    best_batch_final = best_combo_final.split("_")[0]
    best_lr_final = best_combo_final.split("_")[1]
    best_epoch_final = best_combo_final.split("_")[2]
    #print(f"Best hyperparameter combination for all: {best_batch_final}\t{best_lr_final}\t{best_epoch_final}")
    #print(f"Best perfomring avg AUROC = {final_avg[best_combo_final][0]}, avg Youden J = {final_avg[best_combo_final][1]}")
    
    with open("./output/CV_bestcombo.tsv", 'w') as f:
            f.write("Batchsize\tLR\tEpoch\tAUROC\tYoudenJ\n")
            f.write(f"{best_batch_final}\t{best_lr_final}\t{best_epoch_final}\t{final_avg[best_combo_final][0]}\t{final_avg[best_combo_final][1]}\n")
        
main()
end_time = time.time()
print(f"Total time elapsed = {end_time - start_time} seconds")