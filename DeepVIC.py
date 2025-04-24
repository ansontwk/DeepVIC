import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import subprocess
import argparse
import src.utils.headers as cfg 
import time
start_time = time.time()
parser = argparse.ArgumentParser(description=f"{cfg.program_whatdoesitdo} Version {cfg.version}",
                                 epilog = f"{cfg.flavour_text}")
parser.add_argument('-c', '--cutoff', type = float, default=0.537)
parser.add_argument('-m', '--mode', type = str, default = 'b')
parser.add_argument('-i', '--input', type = str, required= True)
parser.add_argument('-s', '--silent',  default = False, action = 'store_true', help = "Silences the stdout printing, defaults to False")
parser.add_argument("-o", "--output", type = str, required = True, help = "filepath to your output file")
parser.add_argument("--clean", default= False, action = 'store_true', help = "clean up intermediate files, defaults to False")
parser.add_argument("--pssmpath", default = "./tmp/features", type = str, help = "path to pssm feature files, defaults to tmp/features")
args = parser.parse_args()
cutoff = args.cutoff
mode = args.mode.lower()
seqfile = str(args.input)
verbose = args.silent
outpath = args.output
cleanup = args.clean
pssmpath = args.pssmpath
def main():
    if not verbose:
        cfg.print_header()
    if not mode == "b" and not mode == "m":
        raise SystemExit(f'{mode} is not a correct mode. Please specific binary "b" or multiclass "m" in the --mode flag')
    subprocess.run(["python", "embed.py", "-i", str(seqfile)])
    subprocess.run(["python", "predict.py", "-c", str(cutoff), "-m", str(mode), "-i", str(seqfile), "-s", str(verbose), "-o", str(outpath), "--pssmpath", str(pssmpath)])
    if cleanup:
        subprocess.run(["rm", "./tmp/tmp.h5"])
main()
if not verbose:
    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.3f} seconds")
    cfg.print_bye()