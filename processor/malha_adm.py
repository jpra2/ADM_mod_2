import numpy as np
from pymoab import core, types, rng, topo_util
import time
import os
import scipy as sp
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
import yaml


parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

# import importlib.machinery
# loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
# utpy = loader.load_module('pymoab_utils')
# loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
# oth = loader.load_module('others_utils').OtherUtils

class malha_adm:

    def __init__(self, mb, dict_tags, input_file_name):
        self.tags = {}
        self.tags['finos'] = mb.tag_get_handle('finos')
        self.tags['intermediarios'] = mb.tag_get_handle('intermediarios')
        self.tags['ID_reord_tag'] = mb.tag_get_handle('ID_reord_tag')
        self.tags['l1_ID'] = mb.tag_get_handle('l1_ID')
        self.tags['l2_ID'] = mb.tag_get_handle('l2_ID')
        self.tags['l3_ID'] = mb.tag_get_handle('l3_ID')
        self.L2_meshset = mb.tag_get_data(mb.tag_get_handle('L2_MESHSET'), 0, flat=True)[0]
        self.input_file_name = input_file_name
        self.intermediarios = mb.get_entities_by_handle(mb.tag_get_data(self.tags['intermediarios'], 0, flat=True)[0])

    def generate_adm_mesh(self, mb, all_volumes, loop=0):

        L2_meshset = self.L2_meshset
        finos = mb.tag_get_data(self.tags['finos'], 0, flat=True)[0]
        finos = mb.get_entities_by_handle(finos)
        intermediarios = self.intermediarios
        ######################################################################
        # ni = ID do elemento no nível i
        n1=0
        n2=0
        aux=0
        meshset_by_L2 = mb.get_child_meshsets(self.L2_meshset)
        print('\n')
        print("INICIOU GERAÇÃO DA MALHA ADM")
        print('\n')
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

                        mb.tag_set_data(self.tags['l1_ID'], elem, n1)
                        mb.tag_set_data(self.tags['l2_ID'], elem, n2)
                        mb.tag_set_data(self.tags['l3_ID'], elem, 1)
                        # elem_tags = self.mb.tag_get_tags_on_entity(elem)
                        # elem_Global_ID = self.mb.tag_get_data(elem_tags[0], elem, flat=True)
                        finos.append(elem)

            if tem_poço_no_vizinho:
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    n1+=1
                    n2+=1
                    t=1
                    for elem in elem_by_L1:
                        if elem not in finos:
                            mb.tag_set_data(self.tags['l1_ID'], elem, n1)
                            mb.tag_set_data(self.tags['l2_ID'], elem, n2)
                            mb.tag_set_data(self.tags['l3_ID'], elem, 2)
                            t=0
                    n1-=t
                    n2-=t
            else:
                n2+=1
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    n1+=1
                    for elem2 in elem_by_L1:
                        # elem2_tags = self.mb.tag_get_tags_on_entity(elem)
                        mb.tag_set_data(self.tags['l2_ID'], elem2, n2)
                        mb.tag_set_data(self.tags['l1_ID'], elem2, n1)
                        mb.tag_set_data(self.tags['l3_ID'], elem2, 3)

        # ------------------------------------------------------------------------------
        print('Definição da malha ADM: ',time.time()-t0)
        t0=time.time()

        # fazendo os ids comecarem de 0 em todos os niveis
        tags = [self.tags['l1_ID'], self.tags['l2_ID']]
        for tag in tags:
            all_gids = mb.tag_get_data(tag, all_volumes, flat=True)
            minim = all_gids.min()
            all_gids -= minim
            mb.tag_set_data(tag, all_volumes, all_gids)
