import torch

from Bio import SeqIO
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from tqdm import tqdm
import os
import time
#================================
startime = time.time()
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
torch.backends.cuda.matmul.allow_tf32 = True
model = model.cuda()
model.esm = model.esm.half()
#model.trunk.set_chunk_size(64)
PDB_DIR = "./customdb/pdb"
TRAINDAT = "./customdb/2mucbp.faa"
#================================
def loadfasta(file):
    #loads fasta file for prediction
    test_sequences = []
    test_headers = []
    
    with open(file, "r") as readfile:
        for record in SeqIO.parse(readfile, "fasta"):
            #seq = str(record.seq).replace("B", "X").replace("Z", "X").replace("J", "X").replace("U", "X").replace("O", "X")
            seq = str(record.seq)
            if len(seq) > 1024:
                seq = seq[:1024]
            test_sequences.append(seq)
            test_headers.append(record.id)
    return test_sequences, test_headers

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def main():
    seqs, headers = loadfasta(TRAINDAT)
    
    for seq, header in tqdm(zip(seqs, headers), total=len(seqs)):
        if f"{header}.pdb" in os.listdir(PDB_DIR):
            continue
        else:
            try:
                print(header, len(seq))
                tokenized_input = tokenizer([seq], return_tensors="pt", 
                                            add_special_tokens=False, 
                                            padding="max_length", truncation=True, 
                                            max_length=len(seq))['input_ids']
                tokenized_input = tokenized_input.cuda()
                with torch.no_grad():
                    output = model(tokenized_input)
                pdbs = convert_outputs_to_pdb(output)
                for pdb in pdbs:
                    with open(f"{PDB_DIR}/{header}.pdb", "w") as f:
                        f.write("".join(pdb))
            except: #catch failed sequences
                with open("./failed.txt", "a") as writefile:
                    writefile.write(f"{header}\n")
main()