import os
import pandas as pd

def main():    
    filedir = "./tmp/lr_gridsearch"
    filelist = os.listdir(filedir)

    max_f1 = 0
    feat = ""

    for file in filelist:
        filename = ".".join(file.split(".")[:-1])
        features = filename.split("-")
        #batch = features[0]
        #lr = features[1]
        #patience = features[2]

        data = pd.read_csv(f"{filedir}/{file}", sep ="\t")
        f1 = data["f1-score"][16]
        if f1 > max_f1:
            max_f1 = f1
            feat = features

    print(feat)
    print(max_f1)

main()
