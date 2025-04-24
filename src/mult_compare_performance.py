import pandas as pd
from utils.plotting import plot_VF, plot_overall, plot_combined
mult_val_VF = "./tmp/mult_val_VF.txt"
mult_val_overall = "./tmp/mult_val_overall.txt"
def main():
    mult_val_VF_wide = pd.melt(pd.read_csv(mult_val_VF, sep="\t"), id_vars=["VF Class"], var_name="Model", value_name="F1")
    mult_val_overall_wide = pd.melt(pd.read_csv(mult_val_overall, sep="\t"), id_vars=["Metric"], var_name = "Model", value_name = "Score")
    #plot_VF(mult_val_VF_wide, "./plot/mult_val_VF14.pdf")
    #plot_overall(mult_val_overall_wide, "./plot/mult_val_overall.pdf")
    plot_combined(mult_val_VF_wide, mult_val_overall_wide, "./plot/mult_val_combined.pdf")
main()

