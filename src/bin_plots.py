import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras 
import h5py as h5
from sklearn.model_selection import train_test_split
from utils.hyper import BIN_CUTOFF as CUTOFF
from utils.stats import get_metrics
from utils.plotting import plot_roc_curve, plot_confu, plot_prcurve
from utils.load_save import load_PLM_embeddings
SEED = 179180
MODELFILE = "./models/binary.keras"

def main():
    tensor_train, label_train, _ = load_PLM_embeddings()
    _, indp_x , _, indp_y = train_test_split(tensor_train, label_train, test_size=0.2, shuffle=True, stratify=label_train, random_state=SEED)
    model = keras.models.load_model(MODELFILE)
    scores = model.predict(indp_x, verbose=0)
        
    #plot ROC
    plot_roc_curve(indp_y, scores, "./plot/bin_ROC.pdf")
    
    #plot Confu
    plot_confu(indp_y, scores, "./plot/bin_confu.pdf", CUTOFF)
    
    #Generate output file for metrics 
    get_metrics(indp_y, scores, "./output/bin_metrics.tsv", CUTOFF)

    #plot PR curve
    plot_prcurve(indp_y, scores, "./plot/bin_PR.pdf")
    

main()