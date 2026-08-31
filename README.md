# DeepVIC - Deep learning Virulence Factor Identifier and Classifier :mag::microbe:

![Static Badge](https://img.shields.io/badge/Version-1.1.1-yellow)
![Static Badge](https://img.shields.io/badge/Linux-Ubuntu-orange?style=flat&logo=ubuntu&logoColor=%23E95420)
![Static Badge](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=%233776AB)
![Static Badge](https://img.shields.io/badge/Tensorflow-v2.14-orange?style=flat&logo=tensorflow&logoColor=%23FF6F00)
[![Static Badge](https://img.shields.io/badge/License-MIT-brightgreen?style=flat)](./LICENSE.md)


DeepVIC enables the prediction and classification of bacterial virulence factors from protein sequences by using ProtBert BFD PLM model embeddings and evolutionary features from position-specific scoring matrices. 

## Setup and Installation
0. Clone this repository and enter the directory
    ```bash
    git clone https://github.com/ansontwk/DeepVIC.git
    cd DeepVIC
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
    
    Alternatively, you can specify the path to ProtBert BFD using the `--protbert_path` flag.


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

    If you did not set a path to ProtBert BFD in step 3

    ```bash
    python DeepVIC.py -m b -i ./example/test.fa -o ./example/output_binary.tsv --protbert_path /PATH/TO/PROTBERTBFD
    python DeepVIC.py -m m -i ./example/test.fa --pssmpath ./example/features -o ./example/output_multiclass.tsv --protbert_path /PATH/TO/PROTBERTBFD
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
and the `--pssmpath` flag is set to `./featurefile`, the directory structure should be as follows:

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

## Extracting PSSM features

As of version 1.1.1, DeepVIC provides a CLI script `DeepVIC_extract_pssm_feature.py` for extraction of PSSM features, in the format that DeepVIC expects. 

Default output directory is `./tmp/features`. Invoke this directory using the `--pssmpath` flag when running DeepVIC in multiclass mode.

We have also provided the corresponding conda environment dependencies in `py27.yml`. To install:

```bash
conda env create -f py27.yml
```

To run psi-blast and extract PSSM features:

```bash
python DeepVIC_extract_pssm_feature.py -i YOUR_FASTA_FILE -t NUM_OF_THREADS -d /PATH/TO/UNIREFBLASTDB
```

Num of threads defaults to 8, and UniRef blastdb defaults to the UNIREF50 defined under `src/utils/paths.py`. 

**Note**: If you want to use a different UniRef blastdb, you can specify the path to it using the `-d` flag.
Also note that DeepVIC uses sequence headers as extracted from ids from `Biopython SeqRecord` to name the output files. Please take note when handling headers that may break the script.

## GUI version

As of version 1.1.2, DeepVIC supports a GUI version based on the Gradio frontend. However, this version is early in development. Bugs and unintended behavior may occur. Please use the CLI version in case you are unsure.

1. Navigate to the DeepVIC directory.

2. Run `python DeepVIC_gui.py`

3. On your browser, navigate to `http://127.0.0.1:7860`

Alternatively, run `python DeepVIC_gui.py --share` to create a sharable link to the GUI.

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
- `gradio==5.39.0`

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

* PSSM information

    As of Aug 2024, [POSSUM](http://possum.erc.monash.edu/) is no longer accessible. Although DeepVIC has a bundled version of POSSUM, you may also consider suggested alternatives to POSSUM such as the R-based tool [PSSMCOOL](https://github.com/BioCool-Lab/PSSMCOOL) or the CLI-tool [ProtFeat](https://github.com/gozsari/ProtFeat).

    Furthermore, note that the UniRef50 database version that DeepVIC uses for feature extraction is from **Feb 2024**. Later versions *may* cause discrepancies in results.

    
## Citation

17 Aug 2026 - DeepVIC has been published in Bioinformatics Advances. Please cite the corresponding article if you use DeepVIC in your work.
```
@article{10.1093/bioadv/vbag237,
    author = {Tsui, Wai-Kai and Chan, You-Xiang and Chow, Kin-Hung and Ho, Pak-Leung and Cao, Huiluo},
    title = {DeepVIC: Modular Prediction and Classification of Bacterial Virulence Factors using Protein Language Model Embeddings},
    journal = {Bioinformatics Advances},
    pages = {vbag237},
    year = {2026},
    month = {08},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbag237},
    url = {https://doi.org/10.1093/bioadv/vbag237},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbag237/70673432/vbag237.pdf},
}
```

DeepVIC uses POSSUM and ProtBert BFD for feature extraction. Please kindly cite those tools as well.

Protbert-BFD/ProtTrans
```
@misc{elnaggar2021prottranscrackinglanguagelifes,
      title={ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing}, 
      author={Ahmed Elnaggar and Michael Heinzinger and Christian Dallago and Ghalia Rihawi and Yu Wang and Llion Jones and Tom Gibbs and Tamas Feher and Christoph Angerer and Martin Steinegger and Debsindhu Bhowmik and Burkhard Rost},
      year={2021},
      eprint={2007.06225},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2007.06225}, 
}
```

POSSUM
```
@article{10.1093/bioinformatics/btx302,
    author = {Wang, Jiawei and Yang, Bingjiao and Revote, Jerico and Leier, André and Marquez-Lago, Tatiana T and Webb, Geoffrey and Song, Jiangning and Chou, Kuo-Chen and Lithgow, Trevor},
    title = {POSSUM: a bioinformatics toolkit for generating numerical sequence feature descriptors based on PSSM profiles},
    journal = {Bioinformatics},
    volume = {33},
    number = {17},
    pages = {2756-2758},
    year = {2017},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btx302},
    url = {https://doi.org/10.1093/bioinformatics/btx302},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/17/2756/49040623/bioinformatics_33_17_2756.pdf},
}
```


## License

This project is licensed under the terms of the MIT license. See [LICENSE](./LICENSE.md) file for more details. 

## Changelog

* v1.0.0 Initial release
* v1.0.1 Added support to directly calling path to ProtBert BFD in `DeepVIC.py`. Cleaned up some formatting, increased some verbosity.
* v1.1.0 Added GUI support! See the [GUI tutorial](#gui-version) for more details.
* v1.1.1 Added utility script support for extracting PSSM features for multiclass predictions. See [how to extract features](#extracting-pssm-features) for more details. Updated minor typo and bug fixes.
* v1.1.2 Improved sharing support for DeepVIC GUI, cleaned up some redundant code.


