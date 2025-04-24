import os
from tqdm import tqdm
import h5py as h5
import numpy as np
from utils.biological import PSSM_TYPE as feature_type
from utils.load_save import load_fasta_faa_user as load_data
from utils.paths import PSSM_path as feature_dir

def main(): 
    _, seqsheaders = load_data("./data/DLDB_33456.faa")
    assert len(seqsheaders) == 33456*2

    for n, header in tqdm(zip(range(len(seqsheaders)),seqsheaders), total=len(seqsheaders)):
        for key, value in feature_type.items():            
            if not header.startswith("DLDB"):
                try:
                    header = header.split("|")[1]
                except:
                    header = header
                    
            file = f"{feature_dir}/{key}/{header}.csv"

            try: 
                if os.stat(file).st_size == 0:
                    raise FileNotFoundError   
                with open(file, "r") as readfile:
                    line = readfile.readline()
                    line = line.strip("\n")
                    data = line.split(",")
                    try:
                        dat2 = [float(i) for i in data]
                    except:
                        exit()
                    assert len(dat2) == value["default"]
                    value["embeddings"].append(dat2)
                    
            except FileNotFoundError:
                vecsize = value["default"]
                defaultvec = [0] * vecsize
                value['embeddings'].append(defaultvec)
                
            if n!= len(feature_type[key]["embeddings"]) - 1 :
                print("error")
                break

    for feature, _ in feature_type.items():
        print(feature, len(feature_type[feature]["embeddings"]))
        assert len(feature_type[feature]["embeddings"]) == len(seqsheaders)
    
    with h5.File("./data/pssm_embeddings_13_v2.h5", "w") as hf:
        for key, value in feature_type.items():
            embedding = np.array(value["embeddings"])
            hf.create_dataset(key, data=embedding)
main()