import os
import pandas as pd

def main():
    filedir ="./tmp/gridsearch_features"
    filelist = os.listdir(filedir)

    max_f1 = 0
    min_f1 = 1
    min_feat = ""
    feat = ""
    
    for file in filelist:
        filename = file.split(".")[0]
        features = filename.split("-")
        data = pd.read_csv(f"{filedir}/{file}", sep ="\t")
        f1 = data["f1-score"][16] #follow classification report format
        if f1 > max_f1:
            max_f1 = f1
            feat = features
        if f1 < min_f1:
            min_f1 = f1
            min_feat = features
            
    print(feat, max_f1)
    print(min_feat, min_f1)

main()