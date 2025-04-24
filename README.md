# DeepVIC - Deep learning Virulence Factor Identifier and Classifier :mag::microbe:

![Static Badge](https://img.shields.io/badge/Version-1.0.0-yellow)
![Static Badge](https://img.shields.io/badge/Linux-Ubuntu-orange?style=flat&logo=ubuntu&logoColor=%23E95420)
![Static Badge](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=%233776AB)
![Static Badge](https://img.shields.io/badge/Tensorflow-v2.14-orange?style=flat&logo=tensorflow&logoColor=%23FF6F00)
[![Static Badge](https://img.shields.io/badge/License-MIT-brightgreen?style=flat)](./LICENSE.md)


DeepVIC enables the prediction and classification of bacterial virulence factors from protein sequences by using ProtBert BFD PLM model embeddings and evolutionary features from position-specific scoring matrices. 

## Setup and Installation
0. Clone this repository
    ```bash
    git clone https://github.com/ansontwk/DeepVIC.git
    ```
1. Install conda environment and dependencies
    ```bash 
    conda env create -f DeepVIC.yml
    ```

2. Activate conda environment
    ```bash
    conda activate DeepVIC
    ```

3. Under `src/utils/paths.py` modify the paths to your local installation of ProtBert BFD.

4. Verify that the `DeepVIC.py` can be executed.
    ```bash
    python DeepVIC.py -h
    ```
    (Optional)

    You may wish to run a dummy sample using test.fa in the `./example` directory.

    ```bash
    python DeepVIC.py -m b -i ./example/test.fa -o ./example/output_binary.tsv
    python DeepVIC.py -m m -i ./example/test.fa --pssmpath ./example/features -o ./example/output_multiclass.tsv
    ```

## Usage

### Basic Usage

#### Binary Classification
In the binary mode, DeepVIC only requires the protein sequence in fasta format for a prediction. DeepVIC, by default, runs in binary mode.

```bash
python DeepVIC.py -m b -i myseq.faa -o /PATH/TO/OUTPUT.TSV
```

#### Multiclass Classification

For VF classification, a path pointing to a directory of pssm features can be provided with the `--pssmpath` flag, defaulting to ./tmp/features. In cases where PSI-BLAST failed to yield hits or if PSSM features are not available, the model can still predict VF classes, but may yield inaccurate/strange results.

```bash
python DeepVIC.py -m m -i myseq.faa --pssmpath /PATH/TO/PSSM/FEATURES -o /PATH/TO/OUTPUT.TSV
```

In the path parsed by the `--pssmpath` flag, it should contain subdirectories with names of `["aac_pssm", "d_fpssm", "edp", "k_separated_bigrams_pssm", "pssm_composition", "rpm_pssm"]`. In each subdirectory, `csv` files corresponding to the fasta header is expected.

For example, your fasta sequences are as 
    ```
    >seq1
    AAAA
    >seq2
    AAAA
    ```
and the `--pssmpath` flag is set to ./featurefile/, the directory structure should be as follows:

    ```
    featurefile/
        aac_pssm/
            seq1.csv
            seq2.csv
        ...
    ```
#### Expected outputs

In both modes, DeepVIC produces a tab-separated file with the predictions as specified by the `-o` flag. Using the same example above, the output file will be as follows:

    ```
    seq1    VF
    seq2    Non-VF
    ```
    
for binary classification, and 

    ```
    seq1    Adherence
    ```
for multiclass classification.

### Additional flags
Add the `-s`/`--silent` flag to suppress the standard output.

Add `--clean` to remove any intermediate files.

## OS and hardware requirements
- unix/linux system (tested on Ubunutu 20.04)
- CUDA-compatible GPU (tested on NVIDIA a6000 ada GPU and NVIDIA RTX 4090 systems)

## External dependencies
- [POSSUM](http://possum.erc.monash.edu/) Version `1.0.0` and related dependencies 
- [ProtBert BFD](https://huggingface.co/Rostlab/prot_bert_bfd) and related dependencies

## Requisites and dependencies
The following packages and versions are used in the project: 

- `python == 3.10.13`
- `bio==1.7.1`
- `pandas == 2.2.2`
- `numpy=1.26.4`
- `scipy == 1.14.0`
- `scikit-learn==1.5.1`
- `seaborn==0.13.2`
- `matplotlib==3.9.2`
- `tensorflow == 2.14.0`
- `pytorch == 2.3.0`
- `umap-learn == 0.5.6`
- `transformers==4.41.2`
- `xgboost==2.1.1`
- `shap==0.46.0`
- `imbalanced-learn==0.12.4`
- `tqdm==4.66.4`

## FAQ and Notes

* Import Error
    
    If you see error such as

    ```
    ImportError: /PATH/TO/DeepVIC/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommRegister
    ```

    A suggested solution is to reinstall torch by running

    ```bash
    pip3 uninstall -y torch torchvision torchaudio  
    pip3 cache purge  
    pip3 install --pre torch torchvision torchaudio
    ```

* DeepVIC GPU requirements

    The DeepVIC package is built with GPU support, CPU-only systems is not explicitly supported. Please ensure you have a CUDA-compatible system (i.e. NVIDIA GPU) before using DeepVIC.

    Discrepancies on results may occur between different systems due to the differences in floating point operations on different GPUs. DeepVIC was created on a system with an NVIDIA a6000 ada GPU. Additional testing was done on an independent system with a NVIDIA RTX 4090 GPU. 

* Alternatives to POSSUM

    As of Aug 2024, [POSSUM](http://possum.erc.monash.edu/) is no longer available. Suggested alternatives to POSSUM is the R-based tool [PSSMCOOL](https://github.com/BioCool-Lab/PSSMCOOL) or the CLI-tool [ProtFeat](https://github.com/gozsari/ProtFeat).

    Kindly cite those tools if you use them for feature extraction.

## Citation

Please cite this repo if you use DeepVIC in your work

Citation details will be updated.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE.md) file for more details. 