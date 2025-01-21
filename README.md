Hybridization of SMILES and chemical-environment-aware tokens to improve performance of molecular structure generation
===
This is molecular structure generation using SMI+AIS(100) tokens based on Pytorch framework.

## Requirements
 * Python==3.8.11
 * Pytorch==2.2.0
 * Pytorch-lightning==2.0.7
 * rdkit
 * sentencepiece
 * botorch

 * For the other packages, please refer to the `*.yml`. To resolve  `PackageNotFoundError`, please add the following channels before creating the environment. 
 ```bash
    conda config --add channels pytorch
    conda config --add channels rdkit
    conda config --add channels conda-forge
 ```
or just use 'env.yml' to create the conda environment.
```
conda env create --file cpu_env.yml
conda env create --file gpu_env.yml
```

## Download for Checkpoint File and Source Data
You can download the pretrained models used in paper. 
   - [ckpt_files<sub>](https://docs.google.com/uc?export=download&id=1FINro8crgOSS0XpSwWRGMCqHGR5fRVnA) 
   - [src_data<sub>](https://docs.google.com/uc?export=download&id=1yHiIs6K3ndoA5cznSgp0ycZnEbFKFKjQ)


or just use 'gdown' to install the files
```
[ckpt file] gdown https://drive.google.com/uc?id=1FINro8crgOSS0XpSwWRGMCqHGR5fRVnA
[src_data] gdown https://drive.google.com/uc?id=1yHiIs6K3ndoA5cznSgp0ycZnEbFKFKjQ
```
 
