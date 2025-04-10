from rdkit import Chem
import os
import os.path
#import pandas as pd
import subprocess
from subprocess import TimeoutExpired, run, PIPE
from rdkit import DataStructs, Chem
from multiprocessing import Pool
import sys
import re
from functools import partial 
import warnings
from rdkit.Chem import CanonSmiles
import json
import pathlig import Path

config = json.load(open('utils/input.json','r'))

def func(idx, smiles, csv_path, num_smiles, target):
    coord_dir = f'{csv_path}/xyz_coord'
    #convert ligand smiles to .pdbqt
    save_dir = f"{csv_path}/input-"+str(idx)
    tmp_dir = f'{csv_path}/input-'+str(idx)+'/tmp.smi'
    os.system(f'mkdir -p {save_dir}' )

    try:
        #print(os.path.isfile(targetfile) )
        data = subprocess.run(f'obabel -:"{smiles}" -ocan --gen3D -h', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        smiles = data.stdout.decode('utf-8').strip()
        targetname = config[f'{target}']['targetfile']
        targetfile = f'{os.environ["BASEDIR"]}/qvina/input/{targetname}'
        pocket_param = config[f'{target}']['pocket_param']
        os.system('echo "' + smiles + '" > ' + tmp_dir)
        subprocess.run('obabel -i smi ' + tmp_dir + ' -O ' + save_dir + '/tmp.pdbqt --gen3D -h', shell=True, timeout=10)
        results = subprocess.run(f'{os.environ["BASEDIR"]}/qvina/qvina2.1 --receptor {targetfile} --ligand ' + save_dir + f'/tmp.pdbqt --center_x {pocket_param[0]} --center_y {pocket_param[1]} --center_z {pocket_param[2]} --size_x {pocket_param[3]} --size_y {pocket_param[4]} --size_z {pocket_param[5]} --exhaustiveness 1 --cpu 1 --num_modes 10 --out ' + save_dir + '/tmp_out.pdbqt', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)

        filename = save_dir + '/tmp_out.pdbqt'
        error_message = results.stderr.decode('utf-8')

    except subprocess.TimeoutExpired:
        filename = save_dir + '/tmp_out.pdbqt'
        error_message='TimeoutExpired'
        warnings.warn('TimeoutExpired:')

    if (('Traceback' in error_message) or ('An internal error occurred in' in error_message) or (not os.path.exists(filename) ) ):
        print(f'Error occur when docking simulation runs!\nError message is on the file({smiles})!')

        with open('qvina_error.txt', 'w') as f:
            f.write(smiles+'\n')
            f.write(error_message)
        f.close()
        os.system(f'rm -rf {save_dir}')
        return None 

    else:
        #print(f'success: {smiles}')
        
        #read tmp_out.pdbqt
        if os.path.getsize(filename) == 0:
            print(f'failed docking simulation!\ntmp_out.pdbqt file size 0 ({smiles})!')
            os.system(f'rm -rf {save_dir}')
            return None

        else:
            lines = open(filename, 'r').readlines()
            score = re.split('\s+', lines[1].strip())[3]

            # generate .xyz file
            current_coord =[]
            for line in lines:
                current_coord.append(line.strip())
                if ('ENDMDL' in line):
                    break

            with open(f'{coord_dir}/{num_smiles}-{idx}.xyz', 'w') as f:
                f.write(str(sum (line[:4] == 'ATOM' for line in current_coord)))
                f.write('\n\n')
                for line in current_coord:
                    if 'ATOM' in line:
                        data = re.split('\s+', line.strip())
                        f.write(data[2]+"\t"+data[5]+"\t"+data[6]+"\t"+data[7]+'\n') 
#            os.system(f'rm -rf {save_dir}' )
            return float(score)

def get_property_qvina(list_smiles, verbose=True, n_repeat=10, csv_path='.', num_smiles=0, target='pdk4'):
    from multiprocessing import Pool
    import sascorer
    import utils.sascorer as sascorer

    coord_dir = f'{csv_path}/xyz_coord'
    Path(coord_dir).mkdir(parents=True,exist_ok=True)
    list_docking=[]
    list_SA=[]
    val_smiles=[]
    success_indice=[]
    failed_smiles=[]
    for idx, smiles in enumerate(list_smiles):
        with Pool() as p:
            store_score = p.map(partial(func, smiles=smiles, csv_path=csv_path, num_smiles=num_smiles, target=target), list(range(n_repeat)  ) )
        store_score = list( filter(lambda s: not(s is None), store_score) )

        try:
            s = sascorer.calculateScore(Chem.MolFromSmiles(smiles))
        except:
            print(f'failed sascore: {idx, smiles}')
            store_score = []

        if len(store_score)!=0:
            os.system(f'cp {coord_dir}/{num_smiles}-{store_score.index(min(store_score))}.xyz {coord_dir}/{num_smiles}-topscored.xyz')
            num_smiles+=1
            success_indice.append(idx)
            list_SA.append(round(s, 2))

            list_docking.append(min(store_score))
            val_smiles.append(CanonSmiles(smiles))
        else:
            failed_smiles.append(smiles)

    return val_smiles, list_docking, list_SA, success_indice, num_smiles, failed_smiles
        

if __name__=="__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    from utils.elem_ais import decode as to_smi
    import utils.sascorer as sascorer
    from rdkit import Chem
    import pickle
    import pandas as pd
    import os
    target = '5btr'
    csv_path='test'
    
    list_smi = [line.strip() for line in open(f'ligand_smi/can_sirt1.txt').readlines()]
#    list_smi = [
#'C[C@H]1CN(c2ccc(nc2N1C(=O)Nc3cccc(c3)c4cnco4)c5cccc(c5)C(F)(F)F)C',
#'[s]1c(nc(c1C(=O)Nc3c(cccc3)c4nc5[s]cc([n]5c4)CN6CCOCC6)C)c2cnccc2',
#'O=C(NC1=C(C=CC=C1)C2=CN3C(SC=C3CN4CCNCC4)=N2)C5=NC6=C(N=C5)C=CC=C6',
#'COc1cc(C(=O)Nc2ccccc2-c2cn3c(CN4CCNCC4)csc3n2)cc(OC)c1OC',
#'O=C(Nc1ccccc1-c1cn2c(CN3CC[C@@H](O)C3)csc2n1)c1ccc2ccccc2c1'
#    ]
    print(len(list_smi))
    list_smiles, list_docking, list_SA, sucess_indices, num_smiles, failed_smiles = get_property_qvina(list_smi, n_repeat=10, target=target, csv_path='test/')
    print(list_smi, list_docking, list_SA)
    df = pd.DataFrame({'smi':list_smiles, 'docking_score':list_docking, 'sa_score':list_SA})
    df.to_csv('5BTR_docking.csv',index=False)
