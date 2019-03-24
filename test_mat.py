import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
import numpy as np
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(parent_dir, 'input')
flying_dir = os.path.join(parent_dir, 'flying')
utils_dir = os.path.join(parent_dir, 'utils')
preprocessor_dir = os.path.join(parent_dir, 'preprocessor')
processor_dir = os.path.join(parent_dir, 'processor')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
bifasico_sol_direta_dir = os.path.join(bifasico_dir, 'sol_direta')
bifasico_sol_multiescala_dir = os.path.join(bifasico_dir, 'sol_multiescala')

os.chdir(bifasico_sol_multiescala_dir)

def get_solution(T, b):
    T = T.tocsc()
    x = linalg.spsolve(T, b)
    return x

# Tf2 = sp.load_npz('Tf2.npz')
# OP1_AMS = sp.load_npz('OP1_AMS.npz')
# OR1_AMS = sp.load_npz('OR1_AMS.npz')
# OP2_AMS = sp.load_npz('OP2_AMS.npz')
# OR2_AMS = sp.load_npz('OR2_AMS.npz')
# OP1_ADM = sp.load_npz('OP1_ADM.npz')
# OR1_ADM = sp.load_npz('OR1_ADM.npz')
# OP2_ADM = sp.load_npz('OP2_ADM.npz')
# OR2_ADM = sp.load_npz('OR2_ADM.npz')
# b = np.load('b.npy')
# ids_volumes_d = np.load('ids_volumes_d.npy')
# values_d = np.load('values_d.npy')
# sol = get_solution(Tf2, b)
# historico = np.load('historico.npy')
# os.chdir(parent_dir)
# n = int(len(historico)/6)
# historico = historico.reshape([n, 6])
Tf = sp.load_npz('Tf.npz')
# import pdb; pdb.set_trace()

# T1_ADM = OR1_ADM.dot(Tf2)
# T1_ADM = T1_ADM.dot(OP1_ADM)
# b1_ADM = OR1_ADM.dot(b)
# T1_ADM = T1_ADM.tocsc()
# # PC1_ADM = get_solution(T1_ADM, b1_ADM)
#
# T2_ADM = OR2_ADM.dot(T1_ADM)
# T2_ADM = T2_ADM.dot(OP2_ADM)
# b2_ADM = OR2_ADM.dot(b1_ADM)
# T2_ADM = T2_ADM.tocsc()
# print(np.linalg.det(T2_ADM.todense()))
# import pdb; pdb.set_trace()

# gg = np.diag(T1_ADM.todense())
# print(gg)
# print(np.prod(gg))
# cont = 0
# for i in gg:
#     print(i)
#     cont+=1
#     if cont == 20:
#         import pdb; pdb.set_trace()
#         cont = 0
# import pdb; pdb.set_trace()
"""
cont = 0
mat = OR1_AMS
for i in range(mat.shape[0]):
    col = mat[i, :]
    print(col)
    if cont == 20:
        import pdb; pdb.set_trace()
        cont = 0
    cont += 1
"""
