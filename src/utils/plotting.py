import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
plt.rcParams['font.family'] = 'serif'
color_dict = {
    999: '#D3D3D3',
    86: '#a2d2e7',
    1: '#67a8cd',
    258: '#ffc17f',
    235: '#b3e19b',
    272: '#6fb3a8',
    204: '#cf9f88',
    271: '#ff9d9f',
    251: '#50aa4b',
    346: '#000000',
    282: '#3581b7',
    301: '#cdb6da',
    83: '#f36569',
    315: '#704ba3',
    325: '#e43030',
}

c_999 = mpatches.Patch(color='#D3D3D3', label='Unclassified')
c_86 = mpatches.Patch(color='#a2d2e7', label='Effector Delivery System')
c_1 = mpatches.Patch(color='#67a8cd', label='Adherence')
c_258 = mpatches.Patch(color='#ffc17f', label='Immune modulation')
c_235 = mpatches.Patch(color='#b3e19b', label='Exotoxin')
c_272 = mpatches.Patch(color='#6fb3a8', label='Nutritional/metabolic factor')
c_204 = mpatches.Patch(color='#cf9f88', label='Motility')
c_271 = mpatches.Patch(color='#ff9d9f', label='Biofilm')
c_251 = mpatches.Patch(color='#50aa4b', label='Exoenzyme')
c_346 = mpatches.Patch(color='#000000', label='Others')
c_282 = mpatches.Patch(color='#3581b7', label='Stress Survival')
c_301 = mpatches.Patch(color='#cdb6da', label='Regulation')
c_83 = mpatches.Patch(color='#f36569', label='Invasion')
c_315 = mpatches.Patch(color='#704ba3', label='Post-translational modification')
c_325 = mpatches.Patch(color='#e43030', label='Antimicrobial activity/competitive advantage')

colour_handle = [c_999, c_86, c_1, c_258, c_235, c_272, c_204, c_271, c_251, c_346, c_282, c_301, c_83, c_315, c_325]
colour_handle_nouncls = [c_86, c_1, c_258, c_235, c_272, c_204, c_271, c_251, c_346, c_282, c_301, c_83, c_315, c_325]

mult_confu_lab = ["Adherence", "Invasion", "Effector Delivery\nSystem", "Motility", "Exotoxin",
        "Exoenzyme", "Immune modulation", "Biofilm", "Nutritional/metabolic\nfactor", "Stress survival",
        "Regulation", "Post-translational\nmodification", "Antimicrobial activity/\ncompetitive advantage", "Others"]

VF_DICT = {
    0:"Adherence",
    1:"Invasion",
    2:"Effector delivery system",
    3:"Motility",
    4:"Exotoxin",
    5:"Exoenzyme",
    6:"Immune modulation",
    7:"Biofilm",
    8:"Nutritional/metabolic factor",
    9:"Stress survival",
    10:"Regulation",
    11:"Post-translational modification",
    12:"Antimicrobial activity/\nCompetitive advantage",
    13:"Others"
}
def filter_unclassified(tensor_arr, vf, cls):
    pos_tensor = []
    pos_cls = []
    for tensor, vf_bin, vf_cls in zip(tensor_arr, vf, cls):
        if vf_bin == 1 and vf_cls != 999:
            pos_tensor.append(tensor)
            pos_cls.append(vf_cls)
    pos_tensor = np.array(pos_tensor)
    pos_cls = np.array(pos_cls)
    return pos_tensor, pos_cls

def filter_VF(tensor_arr, vf, cls):
    
    pos_tensor = []
    pos_cls = []
    for tensor, vf_bin, vf_cls in zip(tensor_arr, vf, cls):
        if vf_bin == 1:
            pos_tensor.append(tensor)
            pos_cls.append(vf_cls)
    pos_tensor = np.array(pos_tensor)
    pos_cls = np.array(pos_cls)
    return pos_tensor, pos_cls

def plot_umap(embeddings, colours, handle, filename, plot_size = (15, 8), col_alpha = 0.5):
    plt.figure(figsize=plot_size)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c = colours, s=2.5, alpha = col_alpha, edgecolors='none')
    #0.3 for facet, 0.5 for nouncls full, 0.8 for all full
    plt.legend(handles=handle,loc="upper center", bbox_to_anchor=(0.5, -0.1), markerscale = 2, fontsize = 10, ncol = 5)
    plt.xlabel("UMAP 2", fontsize = 16)
    plt.ylabel("UMAP 1", fontsize = 16)
    ax = plt.gca()
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    
    ax.set(xticks=[], yticks=[])
    plt.savefig(f"./{filename}", dpi = 300, bbox_inches = 'tight')

def plot_roc_curve(y_true, y_pred ,FILENAME):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='black', label='DeepVIC binary classifier')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label = 'Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 20, loc = 4)
    plt.savefig(FILENAME, dpi = 300, format = 'pdf', bbox_inches='tight')

def plot_prcurve(y_true, y_pred, FILENAME):
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(10,10))
    plt.ylim((0, 1))
    plt.plot(recall, precision, color='black', label='DeepVIC Binary Classifier')
    plt.axhline(y = 0.5, color = 'grey', linestyle = '--', label = 'Random Classifier') 
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.tick_params(labelsize = 18)
    plt.legend(fontsize = 20, loc = 4)
    plt.savefig(FILENAME, dpi = 300, format = 'pdf', bbox_inches='tight')
    
def plot_confu(y_true, y_pred, FILENAME, cutoff):
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    
    plt.figure(figsize=(10,10))
    predictions = [1 if x[0] > cutoff else 0 for x in y_pred]
    
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    array = [[tn, fp], [fn, tp]]
    df_cm = pd.DataFrame(array, index = ['Not VF', 'VF'], columns = ['Not VF', 'VF'])
    ax = sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', annot_kws={"size": 24})       #Fmt G suppress scientific notation
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.tick_params(labelsize = 18)
    plt.savefig(FILENAME, format = 'pdf', dpi = 300, bbox_inches='tight') 

def plot_confu_mult(df, filepath, formatting = '.2f'):
    plt.figure(figsize=(20,17))
    ax = sns.heatmap(df, annot=True, fmt=formatting, cmap='Blues', annot_kws={"size": 24}, square=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    plt.xlabel("Predicted", fontsize=22)
    plt.ylabel("Actual", fontsize=22)
    plt.tick_params(labelsize = 20)
    plt.savefig(filepath, format = 'pdf', dpi = 300, bbox_inches='tight')

def plot_prcurve_mult(micro_rec, micro_pre, micro_auprc, filepath, plot_dict, label_dict):
    plt.figure(figsize=(10,10))
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.tick_params(labelsize = 18)
    plt.plot(micro_rec, micro_pre, color='black', label=f'Micro-average \n(AUPRC = {micro_auprc :.3f})', alpha = 0.8, lw = 1)
    
    #baseline
    rand_chance = 1/len(plot_dict)
    plt.plot([0, 1], [rand_chance, rand_chance], color='grey', linestyle='--', label = f'Baseline AP ={rand_chance :.3f}', alpha = 0.7)
    #multiclass
    for i in range(len(plot_dict)):
        precisions, recalls = plot_dict[i][0], plot_dict[i][1]
        plt.plot(recalls, precisions, lw=1, alpha=0.5, label = f"{label_dict[i]} \n(AUPRC = {plot_dict[i][2] :.3f})")
    
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize = 14)
    plt.savefig(filepath, dpi = 300, format = 'pdf', bbox_inches='tight')
    
def getcosine(listofvectors):
    cosines = cosine_similarity(listofvectors)
    cosines = np.array(cosines)
    return cosines[np.triu_indices_from(cosines, k=1)]

def get_interclass_cosine(tensor_concat, cls, setcls):
    interclass_cosines = {}
    for i in sorted(setcls):
        for j in sorted(setcls):
            if i != j:  # skip intra-class comparison
                vftensor_i = [tensor for tensor, vfcls in zip(tensor_concat, cls) if vfcls == i]
                vftensor_j = [tensor for tensor, vfcls in zip(tensor_concat, cls) if vfcls == j]
                vftensor_i = np.array(vftensor_i)
                vftensor_j = np.array(vftensor_j)
                cosines = cosine_similarity(vftensor_i, vftensor_j)
                cosines = np.array(cosines)
                interclass_cosines[(i, j)] = cosines.flatten().tolist()
    return interclass_cosines

def plot_intraclass(alldistances, labels, filepath, colour = "#4DBBD5FF"):
    plt.figure(figsize=(15, 10))
    g = sns.catplot(alldistances, kind = 'violin', height = 8, aspect = 2, cut = 0, color = colour)
    g.set_xticklabels(labels, rotation = 60, fontsize = 18)
    plt.gca().xaxis.set_tick_params(pad=15)
    plt.xlabel('VF class', fontsize = 20, labelpad= 50)
    plt.yticks(fontsize = 18)
    plt.ylabel('Cosine Similarity', fontsize = 20)
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight', format = 'pdf')

def plot_interclass(matrix, labels, mask, filepath, colour = "coolwarm"):
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(matrix, annot=True, cmap=colour, square=True, fmt='.2f', 
            xticklabels=[], yticklabels=labels, cbar_kws={'shrink': 0.75}, annot_kws= {"size": 18}, mask = mask)
    ax.collections[0].colorbar.ax.tick_params(labelsize=20)
    plt.tick_params(labelsize = 20)
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight', format = 'pdf')
    
def plot_data_cosine_correlation(x, y, data, filepath, xvar, yvar, xname, yname, ylim_start = 0, ylim_end = 1, text_x = 2000, text_y = 0.6, setlim = False, num_test = 6):
    from scipy.stats import pearsonr, spearmanr
    from utils.others import p_to_star
    
    rho = '$\\rho$'
    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)
    
    f = sns.lmplot(x = xname, y = yname, data = data,  scatter_kws={"alpha": 0.8})
    
    f.set(ylim = (ylim_start, ylim_end))
    if setlim:
        f.set(xlim = (0, 1))
        
    f.set_axis_labels(x_var=xvar, y_var=yvar, fontsize = 14)
    f.tick_params(labelsize = 14)
    
    spear_p_cor = spear_p * num_test
    spear_p_cor = 1 if spear_p_cor > 1 else spear_p_cor
    
    pear_p_cor = pear_p * num_test
    pear_p_cor = 1 if pear_p_cor > 1 else pear_p_cor
    
    plt.text(text_x, text_y,
             s = rho + f" : {spear_r:.3f}, p = {spear_p_cor:.3f}, {p_to_star(spear_p_cor)}" + 
             "\n" + f"r : {pear_r:.3f}, p = {pear_p_cor:.3f}, {p_to_star(pear_p_cor)}", fontsize = 12)
    
    plt.savefig(filepath, dpi = 300, bbox_inches = 'tight', format = 'pdf')

def plot_VF(df_melted, filepath, colours = ["#4DBBD5FF", "#8491B4FF", "#3C5488FF", "#E64B35FF", "#BC3C29FF", "#DC0000FF"] , classlab = mult_confu_lab, numcol = 2):
    #plt.rcParams["font.size"] = 12
    plt.figure(figsize=(15, 10))
    g = sns.barplot(data=df_melted, x="VF Class", y="F1", hue="Model", palette=colours)
    g.set_xticklabels(classlab, rotation=45, ha = "right", rotation_mode='anchor')
    plt.ylim(0, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("VF Class", labelpad= 60, fontsize = 16)
    plt.ylabel("Weighted F1 Score", fontsize = 16)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", fancybox = True, ncol = numcol, mode = "expand")
    plt.savefig(filepath, dpi = 300, bbox_inches = "tight")

def plot_overall(df_melted, filepath, colours = ["#4DBBD5FF", "#8491B4FF", "#3C5488FF", "#E64B35FF", "#BC3C29FF", "#DC0000FF"], numcol = 2):
    #plt.rcParams["font.size"] = 12
    plt.figure(figsize=(15, 10))
    g = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette=colours)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 1)
    plt.ylabel("F1 score", fontsize = 16)
    plt.xlabel("Metric", fontsize = 16)
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", fancybox = True, ncol = numcol, mode = "expand")
    plt.savefig(filepath, dpi = 300, bbox_inches = "tight")

def plot_combined(df_melted_14, df_melted_overall, filepath, colours = ["#4DBBD5FF", "#8491B4FF", "#3C5488FF", "#E64B35FF", "#BC3C29FF", "#DC0000FF"], classlab = mult_confu_lab):
    #combine plot_VF and plot_overall
    fig, axs = plt.subplots(2, 1, figsize=(20, 16))

    # 14 vf classes
    sns.barplot(data=df_melted_14, x="VF Class", y="F1", hue="Model", palette=colours, ax=axs[0])
    axs[0].set_xticklabels(classlab, rotation=45, ha="right", rotation_mode='anchor')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel("VF Class", labelpad=60, fontsize=16)
    axs[0].set_ylabel("Weighted F1 Score", fontsize=16)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].get_legend().remove()
    
    # overall
    sns.barplot(data=df_melted_overall, x="Metric", y="Score", hue="Model", palette=colours, ax=axs[1])
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("F1 score", fontsize=16)
    axs[1].set_xlabel("Metric", fontsize=16)
    axs[1].tick_params(axis='both', labelsize=16)
    axs[1].get_legend().remove()
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=20, fancybox=True, labelspacing = 1)
    fig.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    
def plot_bin_shap(df, filepath, xlab = "Treatment", ylab = "SHAP Value"):
    plt.figure(figsize=(10, 6))
    sns.catplot(x=xlab, y=ylab, data=df, kind='violin', height=6, aspect=2)
    plt.xlabel("")
    plt.tick_params(labelsize = 18)
    plt.xticks(fontsize = 18, rotation = 45)
    plt.ylabel("SHAP Value", fontsize = 20)
    plt.savefig(filepath, format = 'pdf', dpi = 300, bbox_inches = 'tight')

def plot_mult_shap(df, labels, filepath, prob):
    plt.rcParams.update({'figure.autolayout': True})
    shapmax = max(abs(np.min(df)), abs(np.max(df)))
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 1]})
    sns.stripplot(df, orient = 'h', alpha = 0.1, ax=axs[0], size=6, jitter = True, color= "#3C5488FF")
    if shapmax < 0.5:
        axs[0].set(xlim=(-0.5, 0.5), xticks=np.arange(-0.5, 0.6, 0.1))
    elif shapmax >= 0.5:
        axs[0].set(xlim=(-1.5, 1.5), xticks=np.arange(-1.5, 1.6, 0.25))
        
    axs[0].set_xlabel("SHAP Values", fontsize = 20)
    axs[0].set_ylabel("VF Class", fontsize = 20)
    axs[0].tick_params(labelsize = 17)
    axs[0].axvline(x=0, color='black', linestyle='--')
    
    # Bar chart
    axs[1].barh(labels, prob, alpha = 0.8, color = '#3C5488FF')
    axs[1].set_xlim(0, 1)
    axs[1].set_xlabel('Predicted Probability', fontsize = 20)
    axs[1].invert_yaxis()
    axs[1].set_yticks([])
    axs[1].set_ylabel('')
    axs[1].xaxis.set_ticks([0, 0.25, 0.5, 0.75, 1])
    axs[1].tick_params(labelsize = 20)
    for index, value in enumerate(prob):
        axs[1].text(value + 0.05, index, str(round(value, 3)), va = 'center', ha = 'left', fontweight = 'bold', fontsize = 16)
    axs[0].set_ylim(axs[1].get_ylim())
    plt.tight_layout()
    plt.savefig(filepath, dpi = 300, format = 'pdf')
    #plt.savefig(f"./plot/{name}_combined.svg", dpi = 300, format = 'svg')