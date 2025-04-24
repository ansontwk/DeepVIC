import pandas as pd
from utils.stats import shapiro_wilk, correction_student_t, mean_95
#Ensure scipy >= 1.13.0, otherwise error will be thrown due to nan_policy flag

INFILE = "./output/binbaseline.tsv"

def main():
    data = pd.read_csv(INFILE, sep = '\t', header = None)
    oh = data[0].to_list()[:-1]
    plm = data[1].to_list()[:-1]
    print("Shapiro-wilk", shapiro_wilk(oh), shapiro_wilk(plm))
    #assert shapiro_wilk(oh) > 0.05 and shapiro_wilk(plm) > 0.05
    print("means", mean_95(oh), mean_95(plm))
    t, p = correction_student_t(oh, plm) 
    print("Student T", t, p)
main()