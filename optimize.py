import argparse
from subprocess import run, PIPE
import time  
from contextlib import contextmanager
import pickle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles, QED
import pandas as pd
import torch 
torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader, TensorDataset    
import sentencepiece as spm
from pytorch_lightning import Trainer
from data import CustomDataset, sos_id, eos_id, pad_id, calculate_prop
from get_property_par import get_property_qvina
from botorch_func import tell, ask
from elem_ais import encode as to_ais
from elem_ais import decode as to_smi
from elem_ais import smiles_tokenizer 
import selfies as sf
import torch.nn as nn

dict_timer = {}

@contextmanager
def timer(desc=""):
    st = time.time()
    yield 
    et = time.time()
    elapsed_time = et-st
    if desc in dict_timer.keys() :
        dict_timer[desc].append(elapsed_time)
    else:
        dict_timer[desc] = [elapsed_time]        
    print( f"Elapsed Time[{desc}]: ", elapsed_time )

def get_encoder_dataloader( tokens, smiles,  batch_size=1024 ):
    """ 
    """
    from torch.utils.data import dataloader, TensorDataset  
    from torch.nn.utils.rnn import pad_sequence
    tokens = pad_sequence([ torch.tensor(item, dtype=torch.long) for item in tokens] , batch_first=True, padding_value=pad_id) #set max_length
    seq_len = torch.sum( torch.any( tokens != pad_id, dim=0 ) )
    tokens= tokens[:, :seq_len] #torch.Tensor(n, l)
    e_mask = (tokens == pad_id )
    prop = torch.Tensor([args.prop]).repeat(tokens.size(0))
    
    return DataLoader( TensorDataset(tokens, e_mask, prop), batch_size=batch_size)

def optimize(args, initial_smi, obj_func = lambda docking, SA: -docking-0.5*SA*SA ):
    with timer('Load Model ') :
        print(args)
        sp = spm.SentencePieceProcessor(model_file= f"{args.sp_model}")
        vocab_size = sp.GetPieceSize()

        from gru_model import MyModel, PositionalEncoder

        model = MyModel(
                         vocab_size = vocab_size,
                         d_model = args.d_model,
                         n_layer = args.n_layer,
                         seq_len = args.seq_len,
                         lr = args.lr
                        )
        device = torch.device('cuda') if torch.cuda.is_available()==True else torch.device('cpu')

        trainer = Trainer(accelerator=args.accelerator, 
                          devices    =args.devices, 
                          strategy   =args.strategy,
                          max_epochs =args.max_epochs
                         )

    with timer('Initial Tokens') :
        initial_tokens =[
                          to_ais(s, args.sp_model) if args.tokenize_method == 'ais'
                          else " ".join(smiles_tokenizer(sf.encoder(s))) if args.tokenize_method=='selfies'
                          else " ".join(smiles_tokenizer(s)) for s in initial_smi ]

        print('encoding ais:\n', initial_tokens)
        initial_tokens = [ sp.encode_as_ids( s ) for s in initial_tokens ]
        print('length initial tokens', [len(tokens) for tokens in initial_tokens])
        print('encoding idx:',initial_tokens)
        
    with timer('Initial Encoder Pred') :
        model.encoder.positional_encoder = PositionalEncoder(args.seq_len, model.encoder.positional_encoder.d_model)
        initial_feature = trainer.predict(model.encoder, dataloaders=get_encoder_dataloader(initial_tokens, initial_smi,args.batch_size) ) #Encoder.predict_step
        initial_feature = torch.cat(initial_feature) #1024
        #print('Feature size (Compressed dim): ', initial_feature.size())

    device, dtype = initial_feature.device, initial_feature.dtype
    feature_size = initial_feature.size(-1)

    with timer('Initial Get Property') :
        valid_struct_id=0
        _, initial_docking, initial_SA, _, num_smiles, failed_smiles = get_property_qvina(initial_smi, n_repeat = args.n_repeat_docking, csv_path=f'{args.csv_path}/tmp', num_smiles = valid_struct_id, target=args.target)
        #print(valid_struct_id)
    if len(failed_smiles)>0:
        print('Error! get_property return error for initial smiles. please check initial smiles')
        return

    dict_invalid = {0: {
                        "invalid_docking": failed_smiles,
                        "invalid_decoding": None
                        }
                   }

    dict_output = {0: { "tokens": initial_tokens, 
                        "smi": initial_smi, 
                        "feature": initial_feature, 
                        "docking": initial_docking, 
                        "SA": initial_SA,
                        "obj_val": [ obj_func(docking,SA) for docking, SA in zip(initial_docking, initial_SA) ],
                       }
                  }
    print("initial obj:", dict_output[0]['obj_val' ] )

    all_feature = initial_feature #initialize feature tensor
    obj_val     = torch.tensor([ obj_func(docking,SA) for docking, SA in zip(initial_docking, initial_SA) ], device=device, dtype=dtype)
    for i_iter in range(1, args.opt_iter+1):
        print(f"{i_iter}  ---------------------- optimization input")
        if args.opt_method == 'botorch':
            with timer(f'[{i_iter}] Tell'):
                acqf, bounds = tell( all_feature, obj_val )
        
            with timer(f'[{i_iter}] Ask') :
                new_feature = ask(args.num_ask, acqf, bounds, 
                                  torch.Tensor([0, args.opt_bound], 
                                               device=device ).unsqueeze(-1).repeat(1, feature_size ) )
        
        with timer(f'[{i_iter}] Decoding') :
            # do decoding
            new_prop = torch.Tensor([args.prop]).repeat(new_feature.size(0))
            tokens = trainer.predict(model, dataloaders=DataLoader( TensorDataset(new_feature, new_prop) , batch_size=args.batch_size)  ) #model.predict_step = val_tokens (Recover Dim)
            tokens = torch.cat(tokens)

        with timer(f'[{i_iter}] Generate String') :
            # get string
            list_smi = [ sp.decode_ids(token.detach().cpu().numpy().tolist()) for token in tokens ]
            smi    = [ to_smi(token) for token in list_smi ] if args.tokenize_method == 'ais' else [ sf.decoder(token.replace(" ", "")) for token in list_smi] if args.tokenize_method == 'selfies' else [ token.replace(" ", "") for token in list_smi ]
        total_smi = len(smi)

        # validate the generated smi
        list_mol = [ Chem.MolFromSmiles(s.strip()) for s in smi ]
        valid_mol = torch.tensor( [i for i, mol in enumerate(list_mol) if (mol is not None and mol.GetNumAtoms()!=0) ] )#idx
        invalid_mol = torch.tensor( [i for i, mol in enumerate(list_mol) if ( mol is None or mol.GetNumAtoms()==0 ) ] )
        
        if(len(valid_mol)==0): 
            dict_output[i_iter] = { "tokens":None, "smi": None, "feature": None, "docking": None, "SA": None, "obj_val": None, "length":0 } 
            print("It doesn't exist new generated smi after filtering gen smi (by MolFromSmiles, GetNumAtoms)")
            dict_invalid[i_iter] = {
                            "invalid_docking": None,
                            "invalid_decoding": smi
                            }
            continue

        valid_smi     = [ smi[idx] for idx in valid_mol ] #type(smi) = list

        with timer(f'[{i_iter}] Get Property') :
            valid_smiles, list_docking, list_SA, success_indices, valid_struct_id, failed_smiles = get_property_qvina(valid_smi, n_repeat = args.n_repeat_docking, csv_path = f'{args.csv_path}/tmp', num_smiles = valid_struct_id , target=args.target)
            print('failed molecules during docking simulation', len(valid_mol) - len(success_indices) )

        if(len(success_indices)==0): 
            dict_output[i_iter] = { "tokens":None, "smi": None, "feature": None, "docking": None, "SA": None, "obj_val": None, "length":0 } 
            dict_invalid[i_iter] = {
                            "invalid_docking": failed_smiles,
                            "invalid_decoding": None
                            }
            pass

        dict_invalid[i_iter] = {
                            "invalid_docking": failed_smiles,
                            "invalid_decoding": [ smi[idx] for idx in invalid_mol ]
                            }

        success_indices = torch.tensor(success_indices, dtype=torch.long)#by docking

        with timer(f'[{i_iter}] Encoding') :
            # do encoding
            feature = trainer.predict(model.encoder, dataloaders=get_encoder_dataloader(tokens, args.batch_size) )
            feature = torch.cat(feature)

        list_score = [ obj_func(docking, SA) for docking, SA in zip(list_docking, list_SA) ]
        valid_idx = valid_mol[success_indices]
        
        list_obj_score = [list_score[valid_idx.tolist().index(i)] if i in valid_idx.tolist() else -100 for i in range(args.num_ask)]

        #update feature/obj_val for optimizer
        obj_val = torch.cat((obj_val, torch.tensor(list_obj_score) ), dim=0)
        all_feature = torch.cat((all_feature, new_feature), dim=0 )
#        print('invalid_idx (obj_val=-100/feature=False): ', invalid_mol)
#        print('obj_val ', obj_val)
#        print('compare feature in all_feature ', (all_feature[:,None] == new_feature[valid_mol][success_indices]).all(-1).any(-1))
        
        dict_output[i_iter] = { "tokens": tokens[success_indices], "smi": valid_smiles, "feature": new_feature, 
                                "docking":list_docking, "SA": list_SA, "obj_val": list_score, "length": len(success_indices) }
    
    with timer(f'save result') :   
        with open(f'{args.csv_path}/optimize_result.pkl', 'wb') as f:
            pickle.dump(dict_output, f)
        with open(f'{args.csv_path}/invalid_result.pkl', 'wb') as f:
            pickle.dump(dict_invalid, f) 

    print("========  timer  ========")            
    print( "\n".join( [ f"{key}: {sum(dict_timer[key])}" for key in dict_timer.keys() ] ) )

    print("======== End opt ========")

    with open(f'{args.csv_path}/optimize_result.pkl', 'rb') as f:
        data = pickle.load(f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', help="dataset path", type=str, default = None)
    parser.add_argument('--opt_iter', help="number of optimization iterations", type=int, default = 100)
    parser.add_argument('--init_smiles', type=int, default=10 )

    # sentencepiece
    parser.add_argument( '--sp_model', type=str )
    parser.add_argument( '--tokenize_method', type=str,  choices=['ais', 'smi', 'selfies'], default='ais' )

    # pytorch_lightning Trainer
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--strategy", default='auto')


    parser.add_argument('--seq_len', help="sequence length", type=int, default=256)

    # model parameters
    parser.add_argument( '--d_model', type=int, default=512 )
    parser.add_argument( '--n_layer', type=int, default=4 )
    parser.add_argument( '--comp_size', type=int, default=32 )
    parser.add_argument( '--check_every_n_epoch', type=int, default=5 )

    # optimize
    parser.add_argument( '--lr', type=float, default=5e-6 )
    parser.add_argument( '--batch_size', type=int, default=1024 )
    parser.add_argument( '--max_epochs', type=int, default=2000 )

    # optimization
    parser.add_argument('--n_repeat_docking', help="number of repeat for docking simulation", type=int, default=10)
    parser.add_argument('--target', type=str, default='pdk4')
    parser.add_argument('--search_const', type=float, default=1.5 )
    parser.add_argument('--opt_bound', type=float, default=1.5)
    parser.add_argument('--opt_method', type=str, default='botorch')
    parser.add_argument('--input_file', type=str, default='pdk4_5' )
    parser.add_argument('--csv_path', type=str, default='./tmp' )
    parser.add_argument('--init_num', type=int, default=5 )

    # Bayesian parameters
    parser.add_argument("--num_ask", type=int, default=100)
    parser.add_argument("--prop", type=int, default=400)

    args = parser.parse_args()  
    assert args.opt_bound>0

    import os
    if not os.path.isdir(args.csv_path):
        os.mkdir(args.csv_path)

    initial_smi = [ line.strip() for line in open(args.input_file).readlines() ][:args.init_num]
    result = optimize(args, initial_smi)
