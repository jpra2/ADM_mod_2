import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
import cython
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
import yaml


parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

input_file = data_loaded['input_file']
ext_h5m = input_file + '.h5m'

mb = core.Core()
root_set = mb.get_root_set()
mtu = topo_util.MeshTopoUtil(mb)
os.chdir(flying_dir)
mb.load_file(ext_h5m)
os.chdir(parent_dir)

#--------------Início dos parâmetros de entrada-------------------
all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)
list_names_tags = ['PERM', 'PHI', 'CENT', 'finos', 'P', 'Q', 'FACES_BOUNDARY', 'AREA',
                   'G_ID_tag', 'ID_reord_tag', 'FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC',
                   'PRIMAL_ID_1', 'PRIMAL_ID_2', 'd1', 'd2', 'K_EQ', 'S_GRAV', 'L2_MESHSET',
                   'intermediarios', 'R0', 'R1', 'GRAVITY', 'MPFA', 'BIFASICO', 'TZ', 'L_TOT']
tags_1 = utpy.get_all_tags_1(mb, list_names_tags)
dict_tags = dict(zip(list_names_tags, tags_1))

def get_tag(name):
    global list_names_tags
    global tags_1
    index = list_names_tags.index(name)
    return tags_1[index]

# n_levels = len(np.unique(mb.tag_get_data(L3_ID_tag, all_volumes, flat=True))) - 1
n_levels = 2
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'

for i in range(n_levels):
    name = name_tag_faces_boundary_meshsets + str(i+2)
    tag = mb.tag_get_handle(name)
    list_names_tags.append(name)
    tags_1.append(tag)
    dict_tags[name] = tag

r0 = mb.tag_get_data(dict_tags['R0'], 0, flat=True)[0]
r1 = mb.tag_get_data(dict_tags['R1'], 0, flat=True)[0]

print('INICIOU PROCESSAMENTO')
print('\n')

intermediarios = mb.get_entities_by_handle(mb.tag_get_data(dict_tags['intermediarios'], 0, flat=True)[0])
##########################################################################################
L1_ID_tag=mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
L2_ID_tag=mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
L3_ID_tag=mb.tag_get_handle("l3_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
list_names_tags.append('l1_ID')
list_names_tags.append('l2_ID')
list_names_tags.append('l3_ID')
tags_1.append(L1_ID_tag)
tags_1.append(L2_ID_tag)
tags_1.append(L3_ID_tag)
dict_tags['l1_ID'] = L1_ID_tag
dict_tags['l2_ID'] = L2_ID_tag
dict_tags['l3_ID'] = L3_ID_tag
##########################################################################################

#############################################

L2_meshset = mb.tag_get_data(get_tag('L2_MESHSET'), 0, flat=True)[0]
finos = mb.tag_get_data(get_tag('finos'), 0, flat=True)
finos = list(mb.get_entities_by_handle(finos))

######################################################################
# ni = ID do elemento no nível i
n1=0
n2=0
aux=0
meshset_by_L2 = mb.get_child_meshsets(L2_meshset)
print("  ")
print("INICIOU SOLUÇÃO ADM")
tempo0_ADM=time.time()
t0 = tempo0_ADM
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1= mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = mb.get_entities_by_handle(m1)
        for elem1 in elem_by_L1:
            if elem1 in finos:
                aux=1
                tem_poço_no_vizinho=True
            if elem1 in intermediarios:
                tem_poço_no_vizinho=True
        if aux==1:
            aux=0
            for elem in elem_by_L1:
                n1+=1
                n2+=1

                mb.tag_set_data(L1_ID_tag, elem, n1)
                mb.tag_set_data(L2_ID_tag, elem, n2)
                mb.tag_set_data(L3_ID_tag, elem, 1)
                elem_tags = mb.tag_get_tags_on_entity(elem)
                elem_Global_ID = mb.tag_get_data(elem_tags[0], elem, flat=True)
                finos.append(elem)

    if tem_poço_no_vizinho:
        for m1 in meshset_by_L1:
            elem_by_L1 = mb.get_entities_by_handle(m1)
            n1+=1
            n2+=1
            t=1
            for elem in elem_by_L1:
                if elem not in finos:
                    mb.tag_set_data(L1_ID_tag, elem, n1)
                    mb.tag_set_data(L2_ID_tag, elem, n2)
                    mb.tag_set_data(L3_ID_tag, elem, 2)
                    t=0
            n1-=t
            n2-=t
    else:
        n2+=1
        for m1 in meshset_by_L1:
            elem_by_L1 = mb.get_entities_by_handle(m1)
            n1+=1
            for elem2 in elem_by_L1:
                elem2_tags = mb.tag_get_tags_on_entity(elem)
                mb.tag_set_data(L2_ID_tag, elem2, n2)
                mb.tag_set_data(L1_ID_tag, elem2, n1)
                mb.tag_set_data(L3_ID_tag, elem2, 3)

# ------------------------------------------------------------------------------
print('Definição da malha ADM: ',time.time()-t0)
t0=time.time()

av=mb.create_meshset()
# for v in all_volumes:
#     mb.add_entities(av,[v])

mb.add_entities(av, all_volumes)

# fazendo os ids comecarem de 0 em todos os niveis
tags = [L1_ID_tag, L2_ID_tag]
for tag in tags:
    all_gids = mb.tag_get_data(tag, all_volumes, flat=True)
    minim = min(all_gids)
    all_gids -= minim
    mb.tag_set_data(tag, all_volumes, all_gids)

os.chdir(flying_dir)
ext_h5m_adm = input_file + '_malha_adm.h5m'
mb.write_file(ext_h5m_adm)
ext_vtk_adm = input_file + '_malha_adm.vtk'
av = mb.create_meshset()
mb.add_entities(av, all_volumes)
mb.write_file(ext_vtk_adm, [av])
np.save('list_names_tags',np.array(list_names_tags))
# names_tags_with_level = ['FACES_BOUNDARY_nv', 'DUAL_nv', 'PRIMAL_ID_nv']
# np.save('names_tags_with_level', np.array(names_tags_with_level))
os.chdir(parent_dir)
##########################################################

#####################################################
#################################
# #Aqui comeca o calculo do metodo adm
#################################
######################################################
