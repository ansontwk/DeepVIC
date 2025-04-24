import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers.utils import logging
logging.set_verbosity(40)
from src.utils.predict_utils import load_data_seqonly, embed_PLM, save_file
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type = str, required= True)
args = parser.parse_args()
seqfile = str(args.input)

def main():
    try:
        seqs, _ = load_data_seqonly(seqfile)
    except:
        raise SystemExit(f'{seqfile} is missing or problematic. Please check your input file and try again.')
    #print(seqs, headers)
    embeddings = embed_PLM(seqs)
    subprocess.run(["mkdir", "-p", "./tmp"])
    save_file(embeddings=embeddings, outfile_path= "./tmp/tmp.h5")
main()
