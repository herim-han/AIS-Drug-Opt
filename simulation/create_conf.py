import os, json
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="config file path", type=str, default=None)
arg = parser.parse_args()

with open(f'{os.environ["BASEDIR"]}/input/input.json') as f:
    dm = json.load(f)

receptor_list = os.listdir(f'{os.environ["BASEDIR"]}/input/converted_receptor/')
ligand_list = os.listdir(f'{os.environ["BASEDIR"]}/input/converted_ligand/')

docking_list = open(f'{os.environ["BASEDIR"]}/input/ligandlist', 'w')

for receptor_file in receptor_list:
    for ligand_file in ligand_list:
        file_dir = receptor_file.split('.')[0] + '/' + ligand_file
        docking_list.write(file_dir)
        docking_list.write('\n')

        os.system('mkdir -p '+f'{os.environ["BASEDIR"]}/input/config/' + file_dir)
        os.system('mkdir -p '+f'{os.environ["BASEDIR"]}/output/' + file_dir)
        if dm['auto'] == "true":
            pocket_pdb = open(f'{os.environ["BASEDIR"]}/input/grid/' + receptor_file.split('.')[0]  + '/out.pocket')
            pocket = pocket_pdb.readlines()
            for line in pocket:
                if line.startswith('REMARK  cluster_grid_size'):
                    tmp_line = line.split()
                    size_x, size_y, size_z = float(tmp_line[2]), float(tmp_line[3]), float(tmp_line[4])
                elif line.startswith('REMARK  cluster_OrigPos'):
                    tmp_line = line.split()
                    center_x, center_y, center_z = float(tmp_line[2]), float(tmp_line[3]), float(tmp_line[4])
                    break;

        else:
            center_x = dm['center_x']
            center_y = dm['center_y']
            center_z = dm['center_z']

            size_x = dm['size_x']
            size_y = dm['size_y']
            size_z = dm['size_z']

        node = math.ceil(dm['core']/16)
        core = int(dm['core']/(node*2))
#core = 16

        energy_range = dm['energy_range']
        exhaustiveness = dm['exhaustiveness']
        num_modes = dm['num_modes']

        with open(f'{os.environ["BASEDIR"]}/input/config/' + file_dir + '/conf.txt', 'w') as f:
            f.write('receptor = '+f'{os.environ["BASEDIR"]}/input/converted_receptor/')
            f.write(receptor_file)
            f.write('\n\n')
            f.write('ligand = '+f'{os.environ["BASEDIR"]}/input/converted_ligand/')
            f.write(ligand_file)
            f.write('\n\n')

            x = 'center_x = ' + str(center_x) + '\n'
            f.write(x)
            y = 'center_y = ' + str(center_y) + '\n'
            f.write(y)
            z = 'center_z = ' + str(center_z) + '\n'
            f.write(z)
            f.write('\n')

            s_x = 'size_x = ' + str(size_x) + '\n'
            f.write(s_x)
            s_y = 'size_y = ' + str(size_y) + '\n'
            f.write(s_y)
            s_z = 'size_z = ' + str(size_z) + '\n'
            f.write(s_z)
            f.write('\n')

            core = 'cpu = ' + str(core) + ' \n'
            f.write(core)
            energy = 'energy_range = ' + str(energy_range) + '\n'
            f.write(energy)
            exhaust = 'exhaustiveness = ' + str(exhaustiveness) + '\n'
            f.write(exhaust)
            num = 'num_modes = ' + str(num_modes) + '\n'
            f.write(num)
docking_list.close()
