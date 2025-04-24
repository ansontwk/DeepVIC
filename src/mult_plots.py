import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, classification_report, confusion_matrix
from utils.plotting import VF_DICT as vf_all, plot_prcurve_mult
import pandas as pd

from utils.plotting import plot_confu_mult, mult_confu_lab as labels
MODELFILE = "./models/multiclass.keras"
def main():
    with h5.File('./data/PSSM_smote_opt.h5', 'r') as hf:        
        indp_x = hf['indp_x'][:]
        indp_y = hf['indp_y_2'][:]
        
    model = tf.keras.models.load_model(MODELFILE)  
    predictions = model.predict(indp_x, verbose = 0)
    predictions_indx = [np.argmax(i) for i in predictions]
    
    #Confusion mtx
    obfu_matrix = confusion_matrix(indp_y, predictions_indx)
    obfu_matrix_normalized = obfu_matrix.astype('float') / obfu_matrix.sum(axis=1)[:, np.newaxis]
    
    df_cm = pd.DataFrame(obfu_matrix_normalized, index = labels, columns = labels)
    df_cm_2 = pd.DataFrame(obfu_matrix, index = labels, columns = labels)
    
    plot_confu_mult(df_cm, './plot/Norm_mult_confu.pdf') 
    plot_confu_mult(df_cm_2, './plot/mult_confu.pdf', formatting = '.3g')
    
    #report
    report = classification_report(indp_y, predictions_indx, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('./output/mult_metrics.tsv', sep = '\t')
    
    #AUPRC
    plot_dict = {}
    for cls_label in range(len(vf_all)):
        binlab = (indp_y == cls_label).astype(int)
        precisions, recalls, _ = precision_recall_curve(binlab, predictions[:,cls_label])
        avg_pre = average_precision_score(binlab, predictions[:,cls_label])
        auprc = auc(recalls, precisions)
        plot_dict[cls_label] = (precisions, recalls, auprc)
        #print(f"AUPRC for class {cls_label}: {auprc}")
    
    #print(len(plot_dict))
    precision = dict()
    recall = dict()
    average_precision = dict()     
    auprcs = dict()
    
    for i in range(len(vf_all)):
        precision[i], recall[i], _ = precision_recall_curve(np.array(indp_y == i), predictions[:, i])
        auprcs[i] = auc(recall[i], precision[i])
        average_precision[i] = average_precision_score(np.array(indp_y == i), predictions[:, i])
        
    precision["micro"], recall["micro"], _ = precision_recall_curve(np.eye(len(vf_all))[indp_y].ravel(), predictions.ravel())
    auprcs["micro"] = auc(recall["micro"], precision["micro"])
    average_precision["micro"] = average_precision_score(np.eye(len(vf_all))[indp_y].ravel(), predictions.ravel())
    #print("average_precision\tAUPRC")
    #print(average_precision["micro"], auprcs["micro"])

    micro_pre = precision['micro']
    micr_rec = recall['micro']
    micr_auprc = auprcs["micro"]
    plot_prcurve_mult(micr_rec, micro_pre, micr_auprc, './plot/mult_PR.pdf', plot_dict=plot_dict, label_dict=vf_all)
main()