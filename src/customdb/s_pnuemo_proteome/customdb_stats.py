import numpy as  np
import pandas as pd
import scipy.stats
filename = "./gpsc24.tsv"
data = pd.read_csv(filename, sep = "\t", header = None)

def get_mean_95ci(data_list):
    data_list = np.array(data_list)
    mean = np.mean(data_list)
    std_err = scipy.stats.sem(data_list)
    h = std_err * scipy.stats.t.ppf((1 + 0.95) / 2., len(data_list)-1)
    return mean, mean-h, mean+h

def get_percent_pos(data_pos, data_all):
    ppos = []
    for pos, all in zip(data_pos, data_all):
        ppos.append(pos / all)
    ppos = np.array(ppos)
    return ppos

with open("stats.txt", "w") as f:
    f.write("GPSC24\n")
    f.write(str(get_mean_95ci(data[3])) + "\n")
    f.write(str(get_mean_95ci(get_percent_pos(data[1], data[6]))) + "\n")