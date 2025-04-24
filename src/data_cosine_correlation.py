
import pandas as pd
from utils.plotting import plot_data_cosine_correlation
INFILE = "./tmp/data_cosine_relationship.csv"

def main():
    data = pd.read_csv(INFILE, sep = ",")
    seqnum = data["No. of seq"].to_numpy()
    meancosine = data["mean cosine"].to_numpy()
    f1 = data["F1"].to_numpy()
    #precision = data["precision"].to_numpy()
    #recall = data["recall"].to_numpy()
    
    plot_data_cosine_correlation(seqnum, f1, data, "./f1_seq_spear.pdf",  "Number of Trainable Sequences", "F1", "No. of seq", "F1")
    plot_data_cosine_correlation(seqnum, meancosine, data, "./cosine_seq_spear.pdf", " Number of Trainable Sequences", "Mean Cosine Similarity", "No. of seq", "mean cosine", text_y = 0.4)
    plot_data_cosine_correlation(meancosine, f1, data, "./f1_cosine_spear.pdf", "Mean Cosine Similarity", "F1", "mean cosine", "F1", text_x=0.1, text_y=0.7, setlim = True)
main()