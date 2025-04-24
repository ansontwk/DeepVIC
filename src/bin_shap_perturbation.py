from Bio import SeqIO
from utils.customdb_dat import domains
from utils.others import allX, X_domain
#perturbation mucbp file

INPUTFILE = "./customdb/2mucbp.faa"
OUTFILE = "./customdb/2mucbp.perturbX.faa"

def main():
    
    headers = []
    seqs = []
    with open(INPUTFILE) as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            headers.append(str(record.id))
            seqs.append(str(record.seq))
            
    with open(OUTFILE, "w") as writefile:
        for header, seq in zip(headers, seqs):
            
            writefile.write(f">{header}\n")
            writefile.write(f"{seq}\n")

            all_seq_preturbed = X_domain(seq, domains[header], header)
            
            for n, seq_preturbed in enumerate(all_seq_preturbed):
                writefile.write(f">{header}_{n}_preturbed\n")
                writefile.write(f"{seq_preturbed}\n")
                
            x_seq = allX(seq, domains[header], header)
            writefile.write(f">{header}_allX\n")
            writefile.write(f"{x_seq}\n")    
main()