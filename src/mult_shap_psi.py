import subprocess
from Bio import SeqIO
from utils.paths import UNIREF50
#psiblast on mucbp2

def main():
    with open("./customdb/2mucbp.faa", "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            with open(f"./tmp/tmp_pssm.fa", "w") as writefile:
                writefile.write(f">{record.id}\n{record.seq}\n")
            subprocess.run(["psiblast", "-query", f"./tmp/tmp_pssm.fa", "-db", UNIREF50, "-num_iterations", "3", "-num_threads", "20", "-outfmt", "6", "-out_ascii_pssm", f"./tmp/pssmfiles/{record.id}.pssm"])
main()