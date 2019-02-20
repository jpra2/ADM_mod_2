import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(parent_dir, 'input')
flying_dir = os.path.join(parent_dir, 'flying')
utils_dir = os.path.join(parent_dir, 'utils')
preprocessor_dir = os.path.join(parent_dir, 'preprocessor')
processor_dir = os.path.join(parent_dir, 'processor')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('definicoes', preprocessor_dir + '/definicoes.py')
loader.load_module('definicoes')
loader = importlib.machinery.SourceFileLoader('ADM_20', processor_dir + '/ADM_20.py')
loader.load_module('ADM_20')
