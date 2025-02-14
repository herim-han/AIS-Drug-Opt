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
For the other packages, please refer to the `*.yml`. To resolve  `PackageNotFoundError`, please add the following channels before creating the environment. 

```bash
conda config --add channels pytorch
conda config --add channels rdkit
conda config --add channels conda-forge
```
or just use 'env.yml' to create the conda environment.
```
conda env create --file env.yml
```

## Download for All Checkpoint Files and Source Data
**Optional 1.** Using download link to install models and full source data:
   - [checkpoints<sub>](https://docs.google.com/uc?export=download&id=1FINro8crgOSS0XpSwWRGMCqHGR5fRVnA) 
   - [src_data<sub>](https://docs.google.com/uc?export=download&id=1yHiIs6K3ndoA5cznSgp0ycZnEbFKFKjQ)

**Optional 2.** Using gdown:
```
gdown https://drive.google.com/uc?id=1FINro8crgOSS0XpSwWRGMCqHGR5fRVnA
gdown https://drive.google.com/uc?id=1yHiIs6K3ndoA5cznSgp0ycZnEbFKFKjQ
```
 
## Setting Environment for Docking Simulation
#### Define the `BASEDIR` path as follows:
```
export BASEDIR=path/to/AIS-Drug-Opt/simulation
```
#### When you apply it to other targets:
Prepare simulation/qvina/input/*.pdbqt for the apo-protein and modify the binding pocket in `utils/input.json`.
You need to assign a specific target name (for example, tmp_target):
```
utils/input.json

{
   "tmp_target": {
       "targetfile": "tmp_target.pdbqt", #located in simulation/qvina/input/
       "pocket_param": [x_center, y_center, z_center, x_size, y_size, z_size]
       }
}
```

## Run
### Run optimize from pre-trained model:
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
* `--csv_path`: Path to the directory for CSV output files.
* `--n_repeat`: Number of docking simulations.
* `--num_ask`: Number of queries to sampling compounds.
* `--tokenize_method`: Tokenization method to use (`smi`,`ais`, and `selfies`).
* `--opt_iter`: Number of optimization iterations.
* `--init_num`: Number of initial compounds.
* `--input_file`: Path to the input file containing ligand SMILES. (.txt or .csv files)

### Fine-tuning using inital smiles:
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
                         --accelerator gpu \
                         --strategy='ddp' \                  
                         --devices 1 \
                         --opt_bound 1.0 \
                         --ft_service                     
``` 

The fine-tuned model will be saved at ./Fine-tuning/
 
### Pre-training model:
```
python train_gru.py  --sp_model data/sp/ais_vocab_100.model \
                     --max_epochs 300 \
                     --batch_size 512 \
                     --accelerator gpu \
                     --devices 1  \
                     --check_val_every_n_epoch 5 \
                     --strategy='ddp' \
                     --lr 5e-5 \
                     --masking_ratio 0 \
                     --filename path/to/file \
                     --tokenize_method ais \
                     --train_ratio 0.2
```
* `--max_epochs`: maximum epochs for training.
* `--accelerator`: gpu or cpu.
* `--check_val_every_n_epoch`: check val every n train epochs.
* `--tokenize_method`: Tokenization method to use (`smi`,`ais`, and `selfies`).
* `--filename`: smiles input.
* `--train_ratio`: split dataset for train dataset (default: 0.2)

The pre-trained model will be stored at ./lightning_logs/*/checkpoints/
