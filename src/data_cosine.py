import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import time

from utils.plotting import mult_confu_lab as labels
from utils.formatting import multimodal
from utils.plotting import getcosine, get_interclass_cosine, plot_intraclass, plot_interclass
starttime = time.time()

def main():    
    tensor_concat, _, cls = multimodal()
    cls = np.array(cls)

    alldistances = {}
    vfcls = list(sorted(set(cls)))
    del vfcls[0]
    del vfcls[-1]

    for i in vfcls:
        vftensor = [tensor for tensor, vfcls in zip(tensor_concat, cls) if vfcls == i]
        vftensor = np.array(vftensor)
        alldistances[str(i)] = getcosine(vftensor).tolist()
    
    interclass_cosines = get_interclass_cosine(tensor_concat, cls, vfcls)
    
    #for key, value in alldistances.items():
    #    print(key, len(value))
    
    mean_interclass_cosines = {}
    for (i, j), value in interclass_cosines.items():
        mean_interclass_cosines[(i, j)] = np.mean(value)
    classes = vfcls
    matrix = np.zeros((len(classes), len(classes)))
    for i, class_i in enumerate(classes):
        for j, class_j in enumerate(classes):
            if i != j:
                matrix[i, j] = mean_interclass_cosines[(class_i, class_j)]
            else:
                matrix[i, j] = np.nan
                
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            matrix[i, j] = np.nan
    
    #==============================
    #saving stuff
    df = pd.DataFrame(matrix)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    df.to_csv("./tmp/interclass_cosine.csv", index=False, header=False)

    #with open("./tmp/cosinev2.json", "w") as writefile:
    #    json.dump(alldistances, writefile)
    
    with open("./tmp/intraclass_mean_median.txt", "w") as writefile:
        writefile.write("key\tmean\tmedian\n")
        for key, value in alldistances.items():
            value = np.array(value)
            writefile.write(f"{key}\t{np.mean(value)}\t{np.median(value)}\n")
    #==============================
    #Plotting
    #print("Plotting Intraclass")
    plot_intraclass(alldistances, labels, "./plot/intra_cosine.pdf")
    
    #print("Plotting Interclass")
    plot_interclass(matrix, labels, mask, "./plot/inter_cosine.pdf")

main()
endtime = time.time()
print(f"Total time elapsed: {endtime - starttime} seconds")