import numpy as np
from utils.hyper import SEED
def formatseq(seq):
    seq = seq.replace("B", "X").replace("Z", "X").replace("J", "X").replace("U", "X").replace("O", "X")
    return seq

def formatseq_BFD(seq):
    seq = formatseq(seq)
    seq = " ".join(seq)
    return seq

def pad(sequence, max_length = 2000):
    if len(sequence) < max_length:
        sequence += "*" * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

def one_hot(sequence):
    from utils.biological import AMINO
    seq_index = [AMINO.get(b, b) for b in sequence]
    seq_index = list(map(int, seq_index))
    matrix = np.eye(int(len(AMINO)), dtype = np.uint8)[seq_index]
    return matrix

def decode(inlist):
    outlist = []
    for i in inlist:
        outstring = i.decode('utf-8')
        outlist.append(outstring)
    return outlist

def format_labelsVFC(inlist):
    outlist = []
    for i  in inlist:
        outstring = i.split("(")[-1].split(")")[0].split("VFC")[-1]
        if outstring == "NA":
            outstring = 999
        elif outstring == "NC":
            outstring = 0
        outstring = int(outstring)
        
        outlist.append(outstring)
    return outlist

def multimodal():
    from utils.load_save import load_PLM_embeddings, load_opt_PSSM
    tensor_array, vf, cls = load_PLM_embeddings()
    pssm_vec = load_opt_PSSM()
    tensor_concat = np.concatenate((tensor_array, pssm_vec), axis=1)
    return tensor_concat, vf, cls

def SMOTE_data(tensor_concat, vf, cls):
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from utils.plotting import filter_unclassified as filter_uncls
    from sklearn.preprocessing import LabelEncoder
    
    train_x, _, train_y, _, train_cls, _ = train_test_split(tensor_concat, vf, cls, test_size = 0.2, shuffle=True, stratify=cls, random_state = SEED)
    train_x_true, val_x, train_y_true, val_y, train_cls_true, val_cls = train_test_split(train_x, train_y, train_cls, test_size = 0.25 , shuffle=True, stratify=train_cls, random_state = SEED)
    train_tensor, train_label = filter_uncls(train_x_true, train_y_true, train_cls_true)
    val_tensor, val_label = filter_uncls(val_x, val_y, val_cls)
    train_tensor = np.array(train_tensor)
    val_tensor = np.array(val_tensor) #output
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    oversample = SMOTE(random_state=SEED)
    train_smote_tensor, train_label_smote = oversample.fit_resample(train_tensor, train_label)
    le = LabelEncoder()
    train_label_smote2 = le.fit_transform(train_label_smote)
    train_label_smote2_oh = np.eye(14)[train_label_smote2]
    val_label2 = le.fit_transform(val_label) #output
    val_label2_oh = np.eye(14)[val_label2]

    return train_smote_tensor, train_label_smote2_oh, val_tensor, val_label2_oh, val_label2

    