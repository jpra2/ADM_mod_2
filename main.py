import os
import yaml
import shutil
import sys
import pdb

deletar = True # deletar os arquivos gerados
somente_deletar = False # deletar os arquivos e sair do script

parent_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(parent_dir, 'input')
flying_dir = os.path.join(parent_dir, 'flying')
utils_dir = os.path.join(parent_dir, 'utils')
preprocessor_dir = os.path.join(parent_dir, 'preprocessor')
processor_dir = os.path.join(parent_dir, 'processor')
output_dir = os.path.join(parent_dir, 'output')

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

ler_anterior = data_loaded['ler_anterior']

if deletar and (not ler_anterior):
    ### deletar arquivos no flying
    bifasico_dir = os.path.join(flying_dir, 'bifasico')
    sol_direta = os.path.join(bifasico_dir, 'sol_direta')
    sol_multi = os.path.join(bifasico_dir, 'sol_multiescala')
    try:
        shutil.rmtree(sol_multi)
    except:
        pass
    try:
        shutil.rmtree(sol_direta)
    except:
        pass
    os.makedirs(sol_direta)
    os.makedirs(sol_multi)
    os.chdir(sol_direta)
    with open('__init__.py', 'w') as f:
        pass
    os.chdir(sol_multi)
    with open('__init__.py', 'w') as f:
        pass

    ### deletar arquivos no output
    bifasico_dir = os.path.join(output_dir, 'bifasico')
    sol_direta = os.path.join(bifasico_dir, 'sol_direta')
    sol_multi = os.path.join(bifasico_dir, 'sol_multiescala')
    try:
        shutil.rmtree(sol_multi)
    except:
        pass
    try:
        shutil.rmtree(sol_direta)
    except:
        pass
    os.makedirs(sol_direta)
    os.makedirs(sol_multi)
    os.chdir(sol_direta)
    with open('__init__.py', 'w') as f:
        pass
    os.chdir(sol_multi)
    with open('__init__.py', 'w') as f:
        pass

    if somente_deletar:
        sys.exit(0)



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
#
# pdb.set_trace()

if bifasico == False:
    # loader = importlib.machinery.SourceFileLoader('solucao_adm_mono_v01', processor_dir + '/solucao_adm_mono_v01.py')
    # loader.load_module('solucao_adm_mono_v01')
    # loader = importlib.machinery.SourceFileLoader('solucao_adm_mono_v02', processor_dir + '/solucao_adm_mono_v02.py')
    # loader.load_module('solucao_adm_mono_v02')
    from processor import solucao_adm_mono_v01

else:
    # loader = importlib.machinery.SourceFileLoader('run_bifasico', processor_dir + '/run_bifasico.py')
    # loader.load_module('run_bifasico')
    os.chdir(parent_dir)

    n = 200
    verif = True
    cont = 0

    while verif:
        cont += 1
        os.system('python rodarbif.py')
        if cont >= n:
            cont = 0
            pdb.set_trace()

    # import processor.run_bifasico

    # from processor import run_bifasico
