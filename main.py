import os
import yaml

parent_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(parent_dir, 'input')
flying_dir = os.path.join(parent_dir, 'flying')
utils_dir = os.path.join(parent_dir, 'utils')
preprocessor_dir = os.path.join(parent_dir, 'preprocessor')
processor_dir = os.path.join(parent_dir, 'processor')

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

bifasico = data_loaded['bifasico']

import importlib.machinery
# loader = importlib.machinery.SourceFileLoader('definicoes_certa', preprocessor_dir + '/definicoes_certa.py')
# loader.load_module('definicoes_certa')
# loader = importlib.machinery.SourceFileLoader('definicoes_v03', preprocessor_dir + '/definicoes_v03.py')
# loader.load_module('definicoes_v03')
# loader = importlib.machinery.SourceFileLoader('definicoes', preprocessor_dir + '/definicoes.py')
# loader.load_module('definicoes')
# # loader = importlib.machinery.SourceFileLoader('ADM_20', processor_dir + '/ADM_20.py')
# # loader.load_module('ADM_20')
# loader = importlib.machinery.SourceFileLoader('malha_adm_v01', processor_dir + '/malha_adm_v01.py')
# loader.load_module('malha_adm_v01')

# loader = importlib.machinery.SourceFileLoader('definicoes_certa', preprocessor_dir + '/definicoes_certa.py')
# loader.load_module('definicoes_certa')
# loader = importlib.machinery.SourceFileLoader('malha_adm_v01', processor_dir + '/malha_adm_v01.py')
# loader.load_module('malha_adm_v01')

def definir():
    loader = importlib.machinery.SourceFileLoader('definicoes_certa', preprocessor_dir + '/definicoes_certa.py')
    loader.load_module('definicoes_certa')
    loader = importlib.machinery.SourceFileLoader('malha_adm_v01', processor_dir + '/malha_adm_v01.py')
    loader.load_module('malha_adm_v01')

# definir()

if bifasico == False:
    loader = importlib.machinery.SourceFileLoader('solucao_adm_mono_v01', processor_dir + '/solucao_adm_mono_v01.py')
    loader.load_module('solucao_adm_mono_v01')
    # loader = importlib.machinery.SourceFileLoader('solucao_adm_mono_v02', processor_dir + '/solucao_adm_mono_v02.py')
    # loader.load_module('solucao_adm_mono_v02')
else:
    loader = importlib.machinery.SourceFileLoader('run_bifasico', processor_dir + '/run_bifasico.py')
    loader.load_module('run_bifasico')
