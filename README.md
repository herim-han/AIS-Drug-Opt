Hybridization of SMILES and chemical-environment-aware tokens to improve performance of molecular structure generation
===
This is molecular structure generation using SMI+AIS(100) tokens based on Pytorch framework.

## Dependencies:
```pip-requirements
python==3.8.11
torch==2.2.0
pytorch-lightning==2.0.7
rdkit
sentencepiece
botorch
```

 * For the other packages, please refer to the `*.yml`. To resolve  `PackageNotFoundError`, please add the following channels before creating the environment. 
 ```bash
    conda config --add channels pytorch
    conda config --add channels rdkit
    conda config --add channels conda-forge
 ```
or just use 'env.yml' to create the conda environment.
```
conda env create --file env.yml
```

## Download for Checkpoint File and Source Data
Option 1. To install models and source data, git lfs is needed.
```
git lfs install
git lfs pull
```

Option 2. You can download the pretrained models used in paper. 
   - [ckpt_files<sub>](https://docs.google.com/uc?export=download&id=1FINro8crgOSS0XpSwWRGMCqHGR5fRVnA) 
   - [src_data<sub>](https://docs.google.com/uc?export=download&id=1yHiIs6K3ndoA5cznSgp0ycZnEbFKFKjQ)

Option 3. It will be downloaded to use 'gdown'.
```
gdown https://drive.google.com/uc?id=1FINro8crgOSS0XpSwWRGMCqHGR5fRVnA
gdown https://drive.google.com/uc?id=1yHiIs6K3ndoA5cznSgp0ycZnEbFKFKjQ
```
 
## Setting Environment for Docking Simulation
For docking simulation, define the `BASEDIR` path as follows:
```
export BASEDIR=path/to/AIS-Drug-Opt/simulation
```
**When you apply it to other targets:**
Prepare .pdbqt for the apo-protein and modify the binding pocket in `input.json`.

## Run Optimization
Running command:
```
python  run_optimize.py  --sp_model ./data/sp/ais_vocab_100.model \
                         --ckpt_path ./models/ais_100_cvae.ckpt \
                         --csv_path path/to/dir \
                         --n_repeat 10 \
                         --num_ask 100 \
                         --tokenize_method ais \
                         --opt_iter 5 \
                         --init_num 10 \
                         --target pdk4 \
                         --input_file ligand_smi/randn_pdk4.txt \
                         --accelerator cpu \
                         --strategy='ddp' \
                         --devices 1 \
                         --opt_bound 1.0
```
> `--csv_path`: Path to the directory for CSV output files.
> `--n_repeat`: Number of docking simulations.
> `--num_ask`: Number of queries to sampling compounds.
> `--tokenize_method`: Tokenization method to use (`smi`,`ais`, and `selfies`).
> `--opt_iter`: Number of optimization iterations.
> `--init_num`: Number of initial compounds.
> `--input_file`: Path to the input file containing ligand SMILES. (.txt or .csv files)
