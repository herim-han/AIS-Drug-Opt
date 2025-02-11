import os 
import argparse 
import re 
import pickle
import sentencepiece as spm
from tqdm import trange, tqdm
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader 
import sys
torch.set_float32_matmul_precision('medium')
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data import pad_id, sos_id, eos_id, unk_id, CustomDataset
from gru_model import MyModel
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles, QED
from elem_ais import encode as to_ais
from elem_ais import decode as to_smi
from elem_ais import smiles_tokenizer 
import selfies as sf
        
def train(args):
    sp = spm.SentencePieceProcessor(model_file=f"{args.sp_model}")
    vocab_size = sp.GetPieceSize()
    print('vocab_size loaded from sp model: ', vocab_size) 
    list_smi = open(args.filename).readlines() 
    list_tokens=[]
    for line in tqdm(list_smi, desc='loading data'):#list
        tmp_line = ( to_ais(line, args.sp_model) if args.tokenize_method == 'ais' 
               else " ".join(smiles_tokenizer(sf.encoder(line))) if args.tokenize_method=='selfies' 
               else " ".join(smiles_tokenizer(line))  )
        list_tokens.append( [ sp.Encode(token.strip())[0] for token in re.split("\s+", tmp_line.strip()) ] )
    train_dataset, valid_dataset = torch.utils.data.random_split(list_tokens, [ int(args.train_ratio*len(list_tokens)), len(list_tokens)-int(args.train_ratio*len(list_tokens)) ] )

    train_dataset = CustomDataset(train_dataset, [list_smi[idx] for idx in train_dataset.indices], args.masking_ratio)
    train_loader  = DataLoader(train_dataset, batch_size = args.batch_size,
                               pin_memory=True, num_workers=30, shuffle=True
                              )

    valid_dataset = CustomDataset(valid_dataset, [list_smi[idx] for idx in valid_dataset.indices], args.masking_ratio)
    valid_loader  = DataLoader(valid_dataset, batch_size = args.batch_size,
                               pin_memory=True, num_workers=30, shuffle=False, 
                              )

    model = MyModel(vocab_size = vocab_size, 
                    seq_len = max(train_dataset.seq_len, valid_dataset.seq_len), 
                    d_model = args.d_model,
                    dropout=args.dropout, 
                    n_layer = args.n_layer,
                    lr = args.lr)

    model.to(torch.device('cuda') if torch.cuda.is_available()==True else torch.device('cpu'))

    if args.ckpt_path is not None:
        model = MyModel.load_from_checkpoint(args.ckpt_path, map_location=model.device)

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{valid_loss:.3f}', 
                                          verbose=True, 
                                          save_top_k=-1, 
                                          monitor='valid_loss', 
                                          mode='min',
                                          every_n_epochs=args.check_val_every_n_epoch)

    trainer = Trainer(accelerator=args.accelerator, 
                      devices    =args.devices, 
                      strategy   =args.strategy,
                      max_epochs =args.max_epochs,
                      check_val_every_n_epoch=args.check_val_every_n_epoch,
                      callbacks = [checkpoint_callback],
                     )

####training
    trainer.fit(model, train_loader, valid_loader, ckpt_path = args.ckpt_path)
####validation
    #trainer.validate(model, valid_loader, ckpt_path = args.ckpt_path)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', help="trained model path", type=str, default = None)    
    parser.add_argument('--filename', help="dataset path", type=str, default = None)    
    parser.add_argument( '--train_ratio', type=float, default=0.8 )
    parser.add_argument( '--masking_ratio', type=float, default=0.0 )
    parser.add_argument( '--sp_model', type=str )
    parser.add_argument( '--lr', type=float, default=5e-4 )
    parser.add_argument( '--batch_size', type=int, default=1024 )
    parser.add_argument( '--max_epochs', type=int, default=1 )
    parser.add_argument( '--dropout', type=float, default=0.1 )
    parser.add_argument( '--d_model', type=int, default=512 )
    parser.add_argument( '--n_layer', type=int, default=4 )
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default='auto')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1 )
    parser.add_argument( '--tokenize_method', type=str,  choices=['ais', 'smi', 'selfies'], default='ais' )

    args = parser.parse_args()  

    assert args.train_ratio <=1.0
    train(args)
