import os
# os.system('rm SOL_ADM_fina.npy')
# os.system('rm residuo.npy')
# os.system('rm SOL_TPFA.npy')

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')

os.chdir(parent_dir)
os.system('rm *.npy')
