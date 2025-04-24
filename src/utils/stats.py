import numpy as np
from scipy.stats import shapiro, t, sem, kruskal
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score, confusion_matrix, average_precision_score
from scikit_posthocs import posthoc_dunn
def shapiro_wilk(data_list):
    shapiro_res = shapiro(data_list, nan_policy = 'omit')
    return shapiro_res.pvalue

def correction_student_t(data1, data2, n_train = 42823, n_test = 10706):
    rho = n_train / n_train  
    n = len(data1)
    df = n - 1
    diff = [(data1[i]-data2[i]) for i in range(n)]
    sig = np.std(diff)
    t_stat = (1 / n * sum(diff)) / (np.sqrt(1 / n + rho) * sig)
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    return t_stat, p

def mean_95(data_list, confidence=0.95):
    a = np.array(data_list)
    n = len(a)
    mean, std_err = np.mean(a), sem(a)
    h = std_err * t.ppf((1 + confidence) / 2., n-1)
    return mean, mean-h, mean+h

'''def get_performance_final(model, test_x, test_y, cutoff):
    y_pred = model.predict(test_x, verbose = 0)
    auc = roc_auc_score(test_y, y_pred)
    truelist = test_y.tolist()
    predlist = y_pred.tolist()
    pred_idx = [1 if x[0] > cutoff else 0 for x in predlist]
  
    mcc = matthews_corrcoef(truelist, pred_idx)
    F1 = f1_score(truelist, pred_idx)
    precision = precision_score(truelist, pred_idx)
    recall = recall_score(truelist, pred_idx)
    tn, fp, fn, tp = confusion_matrix(truelist, pred_idx).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    acc = (tp + tn )/ (tp + tn + fp + fn)
        
    return auc, mcc, F1, precision, recall, specificity, sensitivity, acc'''


def get_metrics(y_true, y_pred, filepath, cutoff):
    predictions = [1 if x > cutoff else 0 for x in y_pred]
    
    mcc = matthews_corrcoef(y_true, predictions)
    F1 = f1_score(y_true, predictions)
    precision = precision_score(y_true, predictions)
    recall = recall_score(y_true, predictions)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    specificity = tn / (tn + fp)
    acc = (tp + tn )/ (tp + tn + fp + fn)
    
    auc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    #print(f'Youden: {youden}, MCC: {mcc}, F1: {F1}, precision: {precision}, recall: {recall}, specificity: {specificity}, sensitivity: {sensitivity}, accuracy: {acc}, AUC: {auc}')
    
    with open(filepath, 'w') as writefile:
        writefile.write(f'AUC\tAUPRC\tMCC\tF1\tPrecision\tRecall\tSpecificity\tAccuracy\tTN\tFP\tFN\tTP\n')
        writefile.write(f'{auc}\t{auprc}\t{mcc}\t{F1}\t{precision}\t{recall}\t{specificity}\t{acc}\t{tn}\t{fp}\t{fn}\t{tp}\n') 
        
def kwtest(df):
    samples = [df[col].values for col in df.columns]
    h, pval = kruskal(*samples)
    freedom = len(samples) - 1
    return h, pval, freedom
    
def dunn(df, labels):

    samples = [df[col].values for col in df.columns]
    ps = posthoc_dunn(samples, p_adjust= "holm")
    return ps    