import numpy as np
import h5py as h5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
#from utils.others import checklist
from utils.formatting import multimodal
from utils.plotting import filter_unclassified as filter_uncls

SEED = 179180
OUTFILE = "./data/PSSM_smote_opt.h5"

def main():
    tensor_concat, vf, cls = multimodal()
    train_x, indp_x, train_y, indp_y, train_cls, indp_cls = train_test_split(tensor_concat, vf, cls, test_size = 0.2, shuffle=True, stratify=cls, random_state = SEED)
    train_x_true, val_x, train_y_true, val_y, train_cls_true, val_cls = train_test_split(train_x, train_y, train_cls, test_size = 0.25 , shuffle=True, stratify=train_cls, random_state = SEED)
    
    train_tensor, train_label = filter_uncls(train_x_true, train_y_true, train_cls_true)
    indp_tensor, indp_label = filter_uncls(indp_x, indp_y, indp_cls)
    val_tensor, val_label = filter_uncls(val_x, val_y, val_cls)    

    train_tensor = np.array(train_tensor)
    indp_tensor = np.array(indp_tensor)
    val_tensor = np.array(val_tensor)
    
    train_label = np.array(train_label)
    indp_label = np.array(indp_label)
    val_label = np.array(val_label)
    
    oversample = SMOTE(random_state=SEED)
    train_smote_tensor, train_label_smote = oversample.fit_resample(train_tensor, train_label)
    
    le = LabelEncoder()
    train_label_smote2 = le.fit_transform(train_label_smote)
    train_label2 = le.fit_transform(train_label)
    indp_label2= le.fit_transform(indp_label)
    val_label2 = le.fit_transform(val_label)

    print(train_smote_tensor.shape, train_label_smote.shape, indp_tensor.shape, indp_label.shape, val_tensor.shape, val_label.shape, train_tensor.shape, train_label.shape)
    print(train_label_smote2.shape, train_label2.shape, indp_label2.shape, val_label2.shape)
    
    with h5.File(OUTFILE, 'w') as hf:
        hf.create_dataset('train_x_smote', data = train_smote_tensor)
        hf.create_dataset('train_y_smote', data = train_label_smote)
        hf.create_dataset('train_y_smote_2', data = train_label_smote2)
        
        hf.create_dataset('indp_x', data = indp_tensor)
        hf.create_dataset('indp_y', data = indp_label)
        hf.create_dataset('indp_y_2', data = indp_label2)
        
        hf.create_dataset('val_x', data = val_tensor)
        hf.create_dataset('val_y', data = val_label)
        hf.create_dataset('val_y_2', data = val_label2)

        hf.create_dataset('train_x', data = train_tensor)
        hf.create_dataset('train_y', data = train_label)
        hf.create_dataset('train_y_2', data = train_label2)
main()