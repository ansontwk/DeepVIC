import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
from tensorflow import keras 
import h5py as h5
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix, average_precision_score
from utils.others import sorting
from utils.hyper import SEED
from utils.formatting import format_labelsVFC, decode
plt.rcParams['font.family'] = 'serif'
CUTOFF = 0.537
COLOURS = ['#DC0000FF', '#00A087FF', '#3C5488FF', '#4DBBD5FF', '#7E6148FF', '#F39B7FFF', "#91D1C2FF"]
MARKERS = ['*', 'o', 's', 'p', '^', 'D', 'v']
#=======================
def scores2lab(scores, cutoff):
    # scores to labels
    predictions = [1 if x > cutoff else 0 for x in scores]
    return predictions
#=======================
#IO functions
#these functions are for reading the outputs of different methods
def ingest_dtvf(rawfile, h5file = None):
    predscore = pd.read_csv(rawfile, sep = ",", header = None).iloc[0].tolist()
    preds = []
    for i in predscore:
        if i > 0.5:
            preds.append(1)
        else:
            preds.append(0)
    
    labs = []
    if h5file is not None:
        with h5.File(h5file, 'r') as hf:
            for key, _ in hf.items():
                labs.append(key)
    truelabs = []
    for lab in labs:
        if lab.startswith("DLDB"):
            truelabs.append(1)
        else:
            truelabs.append(0)
    return predscore, preds, truelabs, labs
def ingest_virpred(filepath):
    data = pd.read_csv(filepath, sep = ',')
    Seq_ID = data['Seq_ID'].to_list()
    labs = data['class'].to_list()
    truelabs = []
    predlabs = []
    for seq, label in zip(Seq_ID, labs):
        if seq.strip(">").startswith("DLDB"):
            truelabs.append(1)
        else:
            truelabs.append(0)
        
        if label == "Non-virulent":
            predlabs.append(0)
        else:
            predlabs.append(1)
            
    return predlabs, truelabs

def ingest_mp4(resultfile):
    result = pd.read_csv(resultfile, sep = '\t')
    scores = result['Probability'].to_list()
    headers = result['Input'].to_list()
    predlabs = []
    truelabs = []
    for header, score in zip(headers, scores):
        if score > 0.5:
            predlabs.append(1)
        else:
            predlabs.append(0)
        truelab = int(header.split("lab")[1])
        truelabs.append(int(truelab))
    return scores, predlabs, truelabs

def ingest_virhunter(filepath):
    data = pd.read_csv(filepath, sep = ',')
    ids = data['id'].to_list()
    scores = data['vf_prob'].to_list()
    truelabs = []
    for id in ids:
        if id.startswith("DLDB"):
            truelabs.append(1)
        else:
            truelabs.append(0)
    return scores, truelabs, ids

def ingest_fungene(resultfile):
    with open(resultfile, 'r') as f:
        truelabs = []
        predlabs = []
        f.readline()
        for line in f:
            header = line.strip().split('\t')[0]
            prediction = line.strip().split('\t')[1]
            if header.startswith("DLDB"):
                truelabs.append(1)
            else:
                truelabs.append(0)
            
            if not prediction.startswith("Non-VFs"):
                predlabs.append(1)
            else:
                predlabs.append(0)
    return predlabs, truelabs

def load_PLM_embeddings(PLMPATH = "./data/DLDB_CDHIT70_4db_33456_embeddings.h5"):
    with h5.File(PLMPATH, 'r') as hf:
        tensor_array = hf['embeddings'][:]
        vf = hf['labels'][:]
        cls = hf['classes'][:]    
    cls = decode(cls)
    cls = format_labelsVFC(cls)
    return tensor_array, vf, cls

def ingest_compare(filepath):
    data = pd.read_csv(filepath, sep = '\t')
    data = data.drop(columns = ['TP', 'FP', 'FN', 'TN'])
    return data

def ingest_deepvf(filepath):
    data = pd.read_csv(filepath, sep = '\t')
    id = data['ID'].tolist()
    scores = data['Scores'].tolist()
    preds = [1 if x > 0.5 else 0 for x in scores]
    
    trues = []
    for i in id:
        if i.startswith(">DLDB"):
            trues.append(1)
        else:
            trues.append(0)
    return id, trues, scores, preds

#=======================
#custom plotting functions
def plot_roc_compare_curve(y_true, y_preds_list, model_labels, FILENAME, colours, deepVF_true):
    print("Plotting ROC curve")
    from sklearn.metrics import roc_curve
    plt.figure(figsize=(10,10))
    for y_pred, label, colour in zip(y_preds_list, model_labels, colours):
        if label == "DeepVF":
            fpr, tpr, _ = roc_curve(deepVF_true, y_pred)
            plt.plot(fpr, tpr, color=colour, label=label + ' AUROC = {0:.3f}'.format(roc_auc_score(deepVF_true, y_pred)), alpha = 0.9)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            plt.plot(fpr, tpr, color=colour, label=label + ' AUROC = {0:.3f}'.format(roc_auc_score(y_true, y_pred)), alpha = 0.9)
    
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random Classifier', alpha = 0.7)        
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 18, loc = 4, frameon=False)
    plt.savefig(FILENAME, dpi = 600, format = 'pdf', bbox_inches='tight')
    
def plot_prcurve_compare(y_true, y_preds_list, model_labels, FILENAME, colours, deepVF_true):
    print("Plotting PR curve")
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(10,10))
    plt.ylim((-0.05, 1.05))
    for y_pred, label, colour in zip(y_preds_list, model_labels, colours):
        if label == "DeepVF":
            precision, recall, _ = precision_recall_curve(deepVF_true, y_pred)
            plt.plot(recall, precision, color=colour, label=label + ' AUPRC = {0:.3f}'.format(average_precision_score(deepVF_true, y_pred)), alpha = 0.9)
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            plt.plot(recall, precision, color=colour, label=label + ' AUPRC = {0:.3f}'.format(average_precision_score(y_true, y_pred)), alpha = 0.9)
    plt.axhline(y = 0.5, color = 'grey', linestyle = '--', label = 'Random Classifier', alpha = 0.7) 
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 18, loc = 4, frameon=False)
    plt.savefig(FILENAME, dpi = 600, format = 'pdf', bbox_inches='tight')

def get_metrics_compare(y_true, y_preds, filepath, labels, deepvf_true):
    with open(filepath, 'w') as writefile:
        writefile.write(f'Tool\tMCC\tF1\tPrecision\tRecall\tSpecificity\tAccuracy\tTN\tFP\tFN\tTP\n')
        for predictions, label in zip(y_preds, labels):
            if label == "DeepVF":
                #deepvf_true
                mcc = matthews_corrcoef(deepvf_true, predictions)
                F1 = f1_score(deepvf_true, predictions)
                precision = precision_score(deepvf_true, predictions)
                recall = recall_score(deepvf_true, predictions)
                tn, fp, fn, tp = confusion_matrix(deepvf_true, predictions).ravel()
                specificity = tn / (tn + fp)
                acc = (tp + tn )/ (tp + tn + fp + fn)
            else: 
                mcc = matthews_corrcoef(y_true, predictions)
                F1 = f1_score(y_true, predictions)
                precision = precision_score(y_true, predictions)
                recall = recall_score(y_true, predictions)
                tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
                specificity = tn / (tn + fp)
                acc = (tp + tn )/ (tp + tn + fp + fn)
           
            writefile.write(f'{label}\t{mcc}\t{F1}\t{precision}\t{recall}\t{specificity}\t{acc}\t{tn}\t{fp}\t{fn}\t{tp}\n') 

def radar(data, metrics, filepath):
    print("Plotting radar plot")
    n = len(metrics)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = np.concatenate((theta, [theta[0]]))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.spines['polar'].set_zorder(1)
    ax.spines['polar'].set_color('white')

    ax.grid(color='lightgrey', linestyle='--', linewidth = 2, alpha = 0.5)
    for idx, (i, row) in enumerate(data.iterrows()):
        values = row[metrics].values.flatten().tolist()
        values = values + [values[0]]
        ax.plot(theta, values, linewidth=2.75, 
                linestyle='solid', 
                label=row['Tool'],
                #markerfacecolor='none', 
                marker=MARKERS[idx % len(MARKERS)], 
                markersize=14,
                alpha = 0.7, 
                color=COLOURS[idx % len(COLOURS)])
    
    plt.ylim(0, 1.05)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0", "0.2", "0.4", "0.6", "0.8", "1"], color="black", size=18)
    plt.xticks(theta, metrics + [metrics[0]], color='black', size=22, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.8, 0.75), fontsize = 20, frameon=False)
    plt.savefig(filepath, format = 'pdf', dpi = 600, bbox_inches='tight')
    
def main():
    #DeepVIC
    tensor_train, label_train, _ = load_PLM_embeddings()
    _, indp_x , _, indp_y = train_test_split(tensor_train, label_train, test_size=0.2, shuffle=True, stratify=label_train, random_state=SEED)
    model = keras.models.load_model("./models/binary.keras")
    deepvic_scores = model.predict(indp_x, verbose=0)
    
    mp4_scores, _, truelabs = ingest_mp4("./tmp/bin_indp_mp4.txt")
    virulenthunter_scores, vir_truelabs, vir_ids = ingest_virhunter("./tmp/bin_indp_virhunter.csv")
    dt_scores, _, dt_truelabs, dt_ids = ingest_dtvf("./tmp/bin_indp_dtvf.raw", "./tmp/bin_indp_dtvf.h5")    
    dt_scores_sorted, _, dt_truelabs_sorted = sorting(dt_scores, dt_truelabs, dt_ids, vir_ids) #resorts ids to be the same

    _, deepvf_truelabs, deepvf_scores, _ = ingest_deepvf("./tmp/bin_indp_deepvf.txt")
    
    #these tools only have binary outputs, no scores available
    fungene_labs, fg_truelabs = ingest_fungene("./tmp/bin_indp_fungene.txt")
    virpred2_labs, virpred2_truelabs = ingest_virpred("./tmp/bin_indp_virpred2.csv")
    
    #sanity check
    np.testing.assert_equal(indp_y, truelabs)
    np.testing.assert_equal(indp_y, vir_truelabs)
    np.testing.assert_equal(indp_y, dt_truelabs_sorted)
    np.testing.assert_equal(indp_y, fg_truelabs)
    np.testing.assert_equal(indp_y, virpred2_truelabs)
    
    testlist = [deepvic_scores, mp4_scores, virulenthunter_scores, dt_scores_sorted, deepvf_scores]
    testlab = ["DeepVIC", "MP4", "VirulentHunter", "DTVF", "DeepVF"]
    colours = ['#DC0000FF', '#00A087FF', '#3C5488FF', '#4DBBD5FF', "#7E6148FF"]
    testlist2 = [virpred2_labs, fungene_labs]
    testlab2 = ["VirulentPred 2.0", "FunGeneTyper"]
    
    full_list = []
    full_labs = []
       
    for i, label in zip(testlist, testlab):
        if label == "DeepVIC":
            full_list.append(scores2lab(i, CUTOFF))
        elif label == "DeepVF": 
            full_list.append(scores2lab(i, 0.85))
        else:
            full_list.append(scores2lab(i, 0.5))
        full_labs.append(label)
    
    #without_scores
    for i, label in zip(testlist2, testlab2):
        full_list.append(i)
        full_labs.append(label)
    
    
    #=========================================
    #real plotting goes here
    
    plot_roc_compare_curve(indp_y, testlist, testlab, "./plot/bin_VF_predictor_compare_ROC.pdf", colours, deepvf_truelabs)
    plot_prcurve_compare(indp_y, testlist, testlab, "./plot/bin_VF_predictor_compare_PR.pdf", colours, deepvf_truelabs)
    get_metrics_compare(indp_y, full_list, "./output/bin_VF_predictor_compare.txt", full_labs, deepvf_truelabs)
    
    #=========================================s
    comparison_df = ingest_compare("./output/bin_VF_predictor_compare.txt")
    metrics = comparison_df.columns[1:].tolist()
    radar(comparison_df, metrics, "./plot/bin_VF_predictor_compare_radar.pdf")

main()



