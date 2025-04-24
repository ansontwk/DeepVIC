import subprocess
from os import listdir, makedirs
from os.path import isfile, join
from utils.biological import FEATURES as features

deepvic = "/PATH/TO/YOUR/DEEPVIC/INSTALL"
input = f"{deepvic}/src/customdb/2mucbp.faa"
tmpdir = f"{deepvic}/src/tmp"

#run this script in conda py27 environment in the path where you installed POSSUM, invoke this script like this
#python3 /PATH/TO/DeepVIC/src/extract_feature_predict.py

def main():
    allfiles = [f for f in listdir(f"{tmpdir}/pssmfiles") if isfile(join(f"{tmpdir}/pssmfiles", f))]
    #print(allfiles)
    for file in allfiles:
        regex = ".".join(file.split(".")[:-1])
        
        #Staging temporary files
        with open(f"{tmpdir}/extract.txt", "w") as writefile:
            writefile.write(f"{regex}\n")
        with open(f"{tmpdir}/tmp.fa", "w") as writefile:
            subprocess.run(["seqkit", "grep", "-f", f"{tmpdir}/extract.txt", f"{input}", "-w0", "--quiet"], stdout=writefile)
        subprocess.run(["cp", f"{tmpdir}/pssmfiles/{file}", f"{tmpdir}/pssm/tmp.pssm"])
            
        #Feature Extraction
        for feature in features:
            makedirs(f"{tmpdir}/customdb/features/{feature}", exist_ok=True)
            subprocess.run(["perl", "possum_standalone.pl", "-i", f"{tmpdir}/tmp.fa", "-o", f"{tmpdir}/customdb/features/{feature}/{regex}.csv", "-t", f"{feature}", "-p", f"{tmpdir}/pssm", "-h", "F"], stdout = subprocess.DEVNULL )
main()