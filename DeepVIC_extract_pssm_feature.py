#This is the module for getting PSSM features into the correct format for DeepVIC from a fasta file

#1. make pssm
#2. get extract pssm_features
import subprocess
from Bio import SeqIO
from os import listdir, makedirs, getcwd, chdir
from src.utils.paths import UNIREF50
import src.utils.headers as cfg
from src.utils.biological import ORDERED_OPT_PSSM as features
import argparse
parser = argparse.ArgumentParser(description=f"{cfg.program_whatdoesitdo} Version {cfg.version}", 
                                 epilog = f"{cfg.flavour_text}")
parser.add_argument('-i', '--input', type = str, required= True)
parser.add_argument('-d', '--db', type = str, default = UNIREF50, help = "path to UniRef50 database, defaults to UNIREF50 as defined in ./src/utils/paths.py")
parser.add_argument('-t', '--threads', type = int, default = 8, help = "number of threads for running PSI-BLAST, defaults to 8")
parser.add_argument("--pssmpath", default = "./tmp/features", type = str, help = "path to pssm feature files, defaults to tmp/features")
args = parser.parse_args()
dbpath = args.db
seqfile = str(args.input)
pssmpath = args.pssmpath
threads = args.threads

def formatseq(seq):
    seq = seq.replace("B", "X").replace("Z", "X").replace("J", "X").replace("U", "X").replace("O", "X")
    return seq

def main():
    subprocess.run(["mkdir", "-p", "./tmp/pssmfiles", "./tmp/pssm"]) #makes staging grounds
    with open(seqfile, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            with open(f"./tmp/tmp_pssm.fa", "w") as writefile:
                writefile.write(f">{record.id}\n{record.seq}\n")
            outpssm = f"./tmp/pssmfiles/{record.id}.pssm"
            print("Now running psiblast on {} with {} threads".format(record.id, threads))
            subprocess.run(["psiblast", "-query", f"./tmp/tmp_pssm.fa", "-db", dbpath, "-num_iterations", "3", "-num_threads", f"{threads}", "-outfmt", "6", "-out_ascii_pssm", outpssm])
            cleanseq = formatseq(record.seq)

            subprocess.run(["cp", outpssm, f"./tmp/pssm/tmp.pssm"])
        
            for feature in features:
                deepvicdir = getcwd()
                makedirs(f"./tmp/features/{feature}", exist_ok=True)
                chdir("./src/dependencies/POSSUM_Standalone_Toolkit")
                try:
                    subprocess.run(["conda", "run", "-n", "py27", "perl", "possum_standalone.pl", "-i", f"../../../tmp/tmp_pssm.fa", "-o", f"../../../tmp/features/{feature}/{record.id}.csv", "-t", f"{feature}", "-p", f"../../../tmp/pssm", "-h", "F"] )
                except:
                    pass #DeepVIC can handle missing features
                
                chdir(deepvicdir)

main()

