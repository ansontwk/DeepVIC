import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
plt.rcParams['font.family'] = 'serif'
from utils.biological import PSSM_TYPE as feature_type
#plot PSSM vector size vs F1 scores
baseline_w_f1 = 0.695
alltheway_f1 = 0.752
bfd_vec = 1024
allthewayvec = 3410 + 1024 

filedir = "./tmp/gridsearch_features"
filelist = os.listdir(filedir)

allsize = []
f1s = []
cols = []

rd = "#DC0000FF"
ble = "#3C5488FF" 

def main():
    for file in filelist:
        filename = file.split(".")[0]
        features = filename.split("-")
        vec_size = 0
        
        for feature in features:
            vec_size += feature_type[feature]["default"]
        
        totalvec = bfd_vec + vec_size
        allsize.append(totalvec)
        
        data = pd.read_csv(f"{filedir}/{file}", sep ="\t")
        f1 = data["f1-score"][16]
        f1s.append(f1)
        cols.append(ble)

    z = np.polyfit(allsize, f1s, 1)
    p = np.poly1d(z)
    r2 = r2_score(f1s, p(allsize))
    
    #plot
    plt.figure(figsize = (7, 5))
    plt.scatter(allsize, f1s, s = 3, alpha = 0.5, c = cols)
    plt.scatter([bfd_vec, allthewayvec], [baseline_w_f1, alltheway_f1], s = 8, alpha = 1, c = [rd, rd])

    plt.ylabel("Weighted-F1 score", fontsize = 14)
    plt.xlabel("Vector size", fontsize = 14)
    plt.plot(allsize, p(allsize), "--", c = "black")
    plt.text(4100, 0.7, f"$R^2$ = {r2:.3f}", fontsize = 8)

    plt.savefig("./plot/PSSM_feature_gridsearch.pdf", dpi = 300, bbox_inches = 'tight')
main()