from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMolecule
import pandas as pd
import os
import json
import glob

with open(f'{os.environ["BASEDIR"]}/input/input.json') as f:
    dm = json.load(f)

if dm['ligand'] == 'csv':
    csv_list = glob.glob(f'{os.environ["BASEDIR"]}/input/*.csv')
    if len(csv_list) > 1:
        print("Upload only one csv file.")
        exit(1)

    try:
        smiles = pd.read_csv(csv_list[0],header=None)
        ligand_list = smiles[0].tolist()
        if Chem.MolFromSmiles(ligand_list[0]) == None:
            ligand_list = ligand_list[1:]
    except Exception as e:
        print("Invalid csv file.", e)
        exit(1)

    if os.path.isdir(f'{os.environ["BASEDIR"]}/input/ligand') == False:
        os.mkdir(f'{os.environ["BASEDIR"]}/input/ligand')

    for i, ligand in enumerate(ligand_list):
        print(ligand)
        mol = Chem.MolFromSmiles(ligand)
        if mol != None:
#            save_dir = f'{os.environ["BASEDIR"]}/input/ligand/ligand_' + str(i+1) + '.pdb'
            save_dir = f'{os.environ["BASEDIR"]}/input/ligand/ligand_' + str(i+1) + '.mol2'
            tmp_dir = f'{os.environ["BASEDIR"]}/input/tmp.smi'

#            mol = Chem.MolFromSmiles(ligand)
#            mol_ = Chem.AddHs(mol)
#            ligand = Chem.MolToSmiles(mol_)
           
            os.system('echo "' + ligand + '" > ' + tmp_dir)
            os.system('obabel -i smi ' + tmp_dir + ' -o pdb -O ' + save_dir + ' -m --gen3D -h')
            os.system('obabel -i smi ' + tmp_dir + ' -O ' + save_dir + ' -m --gen3D -h')
            os.system('rm -fr ' + tmp_dir)

#            #org_code using rdkit
#            mol_ = Chem.AddHs(mol)
#            EmbedMolecule(mol_)
#            Chem.MolToPDBFile(mol_, save_dir)

else:
    ligand_list = glob.glob(f'{os.environ["BASEDIR"]}/input/ligand/*')
    if len(ligand_list) == 1:
        file_name, ext = os.path.splitext(ligand_list[0])
        print(file_name, ext)
        os.system('obabel ' + ligand_list[0] + ' -O' +f'{os.environ["BASEDIR"]}/input/ligand/ligand_' + ext  + ' -m --gen3d -h')
        os.system('mv ' + ligand_list[0] + f'{os.environ["BASEDIR"]}/input/input_ligand')

input_receptor = glob.glob(f'{os.environ["BASEDIR"]}/input/receptor/*')
_, ext = os.path.splitext(input_receptor[0])
os.system('obabel ' + input_receptor[0] + ' -O'+f'{os.environ["BASEDIR"]}/input/receptor/receptor_' + ext + ' -m')
os.system('rm -f ' + input_receptor[0])

if dm['auto']:
    pass
else:
    f = open(f'{os.environ["BASEDIR"]}/input/user.log', 'w')
    f.close()
