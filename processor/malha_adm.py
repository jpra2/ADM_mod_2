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

    def __init__(self, mb, dict_tags, input_file_name, mtu):
        self.tags = {}
        self.tags['finos'] = mb.tag_get_handle('finos')
        self.tags['intermediarios'] = mb.tag_get_handle('intermediarios')
        self.tags['ID_reord_tag '] = mb.tag_get_handle('ID_reord_tag')
        self.tags['l1_ID'] = mb.tag_get_handle('l1_ID')
        self.tags['l2_ID'] = mb.tag_get_handle('l2_ID')
        self.tags['l3_ID'] = mb.tag_get_handle('l3_ID')
        # self.tags['l3_ID'] = mb.tag_get_handle('NIVEL_ID')
        self.mtu = mtu
        self.L2_meshset = mb.tag_get_data(mb.tag_get_handle('L2_MESHSET'), 0, flat=True)[0]
        self.input_file_name = input_file_name
        self.intermediarios = mb.get_entities_by_handle(mb.tag_get_data(self.tags['intermediarios'], 0, flat=True)[0])
        self.meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([dict_tags['PRIMAL_ID_1']]), np.array([None]))
        self.meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([dict_tags['PRIMAL_ID_2']]), np.array([None]))
        self.vertices = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['d1']]), np.array([3]))

    def generate_adm_mesh_v0(self, mb, all_volumes, loop=0):

        nn = len(all_volumes)
        map_volumes = dict(zip(all_volumes, range(nn)))

        list_L1_ID = np.ones(nn, dtype = np.int32)
        list_L2_ID = list_L1_ID.copy()
        list_L3_ID = list_L1_ID.copy()

        L2_meshset = self.L2_meshset
        finos = mb.tag_get_data(self.tags['finos'], 0, flat=True)[0]
        finos = list(mb.get_entities_by_handle(finos))
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
                int_finos = set(elem_by_L1) & set(finos) # interseccao do elementos do meshset com os volumes do nivel1
                int_interm = set(elem_by_L1) & set(intermediarios) # interseccao do elementos do meshset com os volumes do nivel2
                # for elem1 in elem_by_L1:
                #     if elem1 in finos:
                #         aux=1
                #         tem_poço_no_vizinho=True
                #     if elem1 in intermediarios:
                #         tem_poço_no_vizinho=True
                if int_finos:
                    aux=1
                    tem_poço_no_vizinho=True
                if int_interm:
                    tem_poço_no_vizinho=True

                if aux==1:
                    aux=0
                    for elem in elem_by_L1:
                        n1+=1
                        n2+=1

                        # mb.tag_set_data(self.tags['l1_ID'], elem, n1)
                        # mb.tag_set_data(self.tags['l2_ID'], elem, n2)
                        # mb.tag_set_data(self.tags['l3_ID'], elem, 1)
                        # elem_tags = self.mb.tag_get_tags_on_entity(elem)
                        # elem_Global_ID = self.mb.tag_get_data(elem_tags[0], elem, flat=True)
                        finos.append(elem)
                        # finos = rng.unite(finos, rng.Range(elem))

                        level = 1
                        id_elem = map_volumes[elem]
                        list_L1_ID[id_elem] = n1
                        list_L2_ID[id_elem] = n2
                        list_L3_ID[id_elem] = level

                    # level = 1
                    # finos += list(elem_by_L1)
                    # ids_elem = list(map_volumes[elem] for elem in elem_by_L1)
                    # nn1 = len(elem_by_L1)
                    # ids_l1 = np.arange(n1, n1+nn1)
                    # ids_l2 = np.arange(n2, n2+nn1)
                    # list_L1_ID[ids_elem] = ids_l1
                    # list_L2_ID[ids_elem] = ids_l2
                    # list_L3_ID[ids_elem] = np.repeat(level, nn1)
                    # n1 += nn1
                    # n2 += nn1

            if tem_poço_no_vizinho:
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    n1+=1
                    n2+=1
                    t=1
                    for elem in elem_by_L1:
                        if elem not in finos:

                            # mb.tag_set_data(self.tags['l1_ID'], elem, n1)
                            # mb.tag_set_data(self.tags['l2_ID'], elem, n2)
                            # mb.tag_set_data(self.tags['l3_ID'], elem, 2)
                            t=0

                            level = 2
                            id_elem = map_volumes[elem]
                            list_L1_ID[id_elem] = n1
                            list_L2_ID[id_elem] = n2
                            list_L3_ID[id_elem] = level

                    n1-=t
                    n2-=t
            else:
                n2+=1
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    n1+=1
                    for elem2 in elem_by_L1:
                        # elem2_tags = self.mb.tag_get_tags_on_entity(elem)
                        # mb.tag_set_data(self.tags['l2_ID'], elem2, n2)
                        # mb.tag_set_data(self.tags['l1_ID'], elem2, n1)
                        # mb.tag_set_data(self.tags['l3_ID'], elem2, 3)

                        id_elem = map_volumes[elem2]
                        level = 3
                        list_L1_ID[id_elem] = n1
                        list_L2_ID[id_elem] = n2
                        list_L3_ID[id_elem] = level


        # ------------------------------------------------------------------------------
        print('Definição da malha ADM: ',time.time()-t0)
        t0=time.time()

        # fazendo os ids comecarem de 0 em todos os niveis
        # tags = [self.tags['l1_ID'], self.tags['l2_ID']]
        # for tag in tags:
        #     all_gids = mb.tag_get_data(tag, all_volumes, flat=True)
        #     minim = all_gids.min()
        #     all_gids -= minim
        #     mb.tag_set_data(tag, all_volumes, all_gids)

        list_L1_ID -= list_L1_ID.min()
        list_L2_ID -= list_L2_ID.min()

        mb.tag_set_data(self.tags['l1_ID'], all_volumes, list_L1_ID)
        mb.tag_set_data(self.tags['l2_ID'], all_volumes, list_L2_ID)
        mb.tag_set_data(self.tags['l3_ID'], all_volumes, list_L3_ID)

    def generate_adm_mesh_v2(self, mb, all_volumes, loop=0):

        nn = len(all_volumes)
        map_volumes = dict(zip(all_volumes, range(nn)))

        list_L1_ID = np.zeros(nn, dtype = np.int32)
        list_L2_ID = list_L1_ID.copy()
        list_L3_ID = list_L1_ID.copy()

        L2_meshset = self.L2_meshset
        finos = mb.tag_get_data(self.tags['finos'], 0, flat=True)[0]
        finos = mb.get_entities_by_handle(finos)
        print(len(finos), 'tamanho finos 1\n')
        # intermediarios = self.intermediarios
        # intermediarios = rng.subtract(intermediarios, finos)
        # intermediarios2 = self.mtu.get_bridge_adjacencies(finos, 2, 3)
        # intermediarios2 = rng.subtract(intermediarios2, finos)
        # intermediarios = rng.unite(intermediarios, intermediarios2)

        ######################################################################
        # ni = ID do elemento no nível i

        aux=0
        meshset_by_L2 = mb.get_child_meshsets(self.L2_meshset)
        print('\n')
        print("INICIOU GERAÇÃO DA MALHA ADM")
        print('\n')
        tempo0_ADM=time.time()
        t0 = tempo0_ADM

        for m1 in self.meshsets_nv1:
            elem_by_L1 = mb.get_entities_by_handle(m1)
            intersect1 = rng.intersect(elem_by_L1, finos)
            nn1 = len(intersect1)
            if nn1 > 0:
                finos = rng.unite(finos, elem_by_L1)

        print(len(finos), 'tamanho finos 2\n')

        intermediarios = rng.Range(np.array(self.intermediarios))
        intermediarios2 = self.mtu.get_bridge_adjacencies(finos, 2, 3)
        intermediarios = rng.unite(intermediarios, intermediarios2)
        intermediarios = rng.subtract(intermediarios, finos)
        print(len(intermediarios), 'tamanho intermediarios 1\n')

        for m2 in self.meshsets_nv2:
            elem_by_L2 = mb.get_entities_by_handle(m2)
            intersect1 = rng.intersect(elem_by_L2, intermediarios)
            nn1 = len(intersect1)
            if nn1 > 0:
                intermediarios = rng.unite(intermediarios, elem_by_L2)

        print(len(intermediarios), 'tamanho intermediarios 2\n')

        print(len(finos) + len(intermediarios), '\n')
        print(len(all_volumes), '\n')
        import pdb; pdb.set_trace()


        # elems_nv3 = rng.subtract(all_volumes, rng.unite(intermediarios, finos))

        n1=0
        n2=0
        for m2 in self.meshsets_nv2:
            verif_nv3 = True
            verif_nv2 = False
            verif_nv1 = False
            elems_tot = []
            meshset_by_L1 = mb.get_child_meshsets(m2)
            for m1 in meshset_by_L1:
                elem_by_L1 = mb.get_entities_by_handle(m1)
                ids_local_volumes = np.array(list((map_volumes[v] for v in elem_by_L1)))
                intersect1 = rng.intersect(elem_by_L1, finos)
                nn1 = len(intersect1)
                if nn1 > 0:
                    verif_nv3 = False
                    verif_nv1 = True
                    ids_l1 = np.arange(n1, n1+nn1)
                    ids_l2 = np.arange(n2, n2+nn1)
                    list_L1_ID[ids_local_volumes] = ids_l1
                    list_L2_ID[ids_local_volumes] = ids_l2
                    list_L3_ID[ids_local_volumes] = np.repeat(1, len(elem_by_L1))
                    elems_tot += list(elem_by_L1)
                    n1 += nn1
                    n2 += nn1
                    continue
                intersect1 = rng.intersect(elem_by_L1, intermediarios)
                nn1 = len(intersect1)
                if nn1 > 0:
                    verif_nv3 = False
                    verif_nv2 = True
                    ids_l1 = np.repeat(n1+1, len(elem_by_L1))
                    ids_l2 = np.repeat(n2+1, len(elem_by_L1))
                    list_L1_ID[ids_local_volumes] = ids_l1
                    list_L2_ID[ids_local_volumes] = ids_l2
                    list_L3_ID[ids_local_volumes] = np.repeat(2, len(elem_by_L1))
                    elems_tot += list(elem_by_L1)
                    n1 += 1
                    n2 += 1
                    continue
            elems_tot = rng.Range(elems_tot)
            elem_by_L2 = mb.get_entities_by_handle(m2)
            sub1 = rng.subtract(elem_by_L2, elems_tot)
            nn2 = len(sub1)

            if verif_nv3:
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    ids_local_volumes = np.array(list((map_volumes[v] for v in elem_by_L1)))
                    ids_l1 = np.repeat(n1+1, len(elem_by_L1))
                    list_L1_ID[ids_local_volumes] = ids_l1
                    n1 += 1
                ids_local_volumes = np.array(list((map_volumes[v] for v in elem_by_L2)))
                ids_l2 = np.repeat(n2+1, len(elem_by_L2))
                list_L2_ID[ids_local_volumes] = ids_l2
                list_L3_ID[ids_local_volumes] = np.repeat(3, len(elem_by_L2))
                n2 += 1
                continue

            if verif_nv2:
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    intersect1 = rng.intersect(elem_by_L1, elems_tot)
                    nn1 = len(intersect1)
                    if nn1 > 0:
                        continue
                    ids_local_volumes = np.array(list((map_volumes[v] for v in elem_by_L1)))
                    ids_l1 = np.repeat(n1+1, len(elem_by_L1))
                    ids_l2 = np.repeat(n2+1, len(elem_by_L1))
                    list_L1_ID[ids_local_volumes] = ids_l1
                    list_L2_ID[ids_local_volumes] = ids_l2
                    list_L3_ID[ids_local_volumes] = np.repeat(2, len(elem_by_L1))
                    elems_tot += list(elem_by_L1)
                    n1 += 1
                    n2 += 1
                continue

            if verif_nv1:
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    intersect1 = rng.intersect(elem_by_L1, elems_tot)
                    nn1 = len(intersect1)
                    if nn1 > 0:
                        continue
                    ids_local_volumes = np.array(list((map_volumes[v] for v in elem_by_L1)))
                    ids_l1 = np.arange(n1, n1+nn1)
                    ids_l2 = np.repeat(n2, n2+nn1)
                    list_L1_ID[ids_local_volumes] = ids_l1
                    list_L2_ID[ids_local_volumes] = ids_l2
                    list_L3_ID[ids_local_volumes] = np.repeat(1, len(elem_by_L1))
                    n1 += nn1
                    n2 += nn1
                continue

        mb.tag_set_data(self.tags['l1_ID'], all_volumes, list_L1_ID)
        mb.tag_set_data(self.tags['l2_ID'], all_volumes, list_L2_ID)
        mb.tag_set_data(self.tags['l3_ID'], all_volumes, list_L3_ID)

    def generate_adm_mesh_v3(self, mb, all_volumes, loop=0):

        nn = len(all_volumes)
        map_volumes = dict(zip(all_volumes, range(nn)))

        list_L1_ID = []
        list_L2_ID = []
        list_L3_ID = []
        volumes = []

        L2_meshset = self.L2_meshset
        finos = mb.tag_get_data(self.tags['finos'], 0, flat=True)[0]
        finos = mb.get_entities_by_handle(finos)
        finos = set(rng.intersect(self.vertices, finos))
        intermediarios = set(self.intermediarios) & set(self.vertices)
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
            print(f'n2 {n2}')
            meshset_by_L1 = mb.get_child_meshsets(m2)
            for m1 in meshset_by_L1:
                elem_by_L1 = mb.get_entities_by_handle(m1)
                int_finos = set(elem_by_L1) & finos # interseccao do elementos do meshset com os volumes do nivel1
                int_interm = set(elem_by_L1) & intermediarios # interseccao do elementos do meshset com os volumes do nivel2
                # for elem1 in elem_by_L1:
                #     if elem1 in finos:
                #         aux=1
                #         tem_poço_no_vizinho=True
                #     if elem1 in intermediarios:
                #         tem_poço_no_vizinho=True
                if int_finos:
                    aux=1
                    tem_poço_no_vizinho=True
                if int_interm:
                    tem_poço_no_vizinho=True

                if aux==1:
                    aux=0
                    level = 1
                    # finos = finos | set(elem_by_L1)
                    volumes.append(np.array(elem_by_L1))
                    nn1 = len(elem_by_L1)
                    ids_l1 = np.arange(n1, n1+nn1)
                    ids_l2 = np.arange(n2, n2+nn1)
                    list_L1_ID.append(ids_l1)
                    list_L2_ID.append(ids_l2)
                    list_L3_ID.append(np.repeat(level, nn1))
                    n1 += nn1
                    n2 += nn1

            if tem_poço_no_vizinho:
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    # conj = rng.intersect(elem_by_L1, (finos))
                    conj = set(elem_by_L1) & finos
                    if conj:
                        level = 2
                        volumes.append(np.array(elem_by_L1))
                        list_L1_ID.append(np.repeat(n1, len(elem_by_L1)))
                        list_L2_ID.append(np.repeat(n2, len(elem_by_L1)))
                        list_L3_ID.append(np.repeat(level, len(elem_by_L1)))
                        n1 += 1
                        n2 += 1

            else:
                level = 3
                for m1 in meshset_by_L1:
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    volumes.append(np.array(elem_by_L1))
                    list_L1_ID.append(np.repeat(n1, len(elem_by_L1)))
                    n1 += 1

                elem_by_L2 = mb.get_entities_by_handle(m2)
                list_L2_ID.append(np.repeat(n2, len(elem_by_L2)))
                list_L3_ID.append(np.repeat(level, len(elem_by_L2)))
                n2 += 1


        # ------------------------------------------------------------------------------
        print('Definição da malha ADM: ',time.time()-t0)
        t0=time.time()

        # fazendo os ids comecarem de 0 em todos os niveis
        # tags = [self.tags['l1_ID'], self.tags['l2_ID']]
        # for tag in tags:
        #     all_gids = mb.tag_get_data(tag, all_volumes, flat=True)
        #     minim = all_gids.min()
        #     all_gids -= minim
        #     mb.tag_set_data(tag, all_volumes, all_gids)

        # list_L1_ID -= list_L1_ID.min()
        # list_L2_ID -= list_L2_ID.min()
        volumes = np.concatenate(volumes)
        list_L1_ID = np.concatenate(list_L1_ID)
        list_L2_ID = np.concatenate(list_L2_ID)
        list_L3_ID = np.concatenate(list_L3_ID)
        import pdb; pdb.set_trace()
        list_L1_ID -= list_L1_ID.min()
        list_L2_ID -= list_L2_ID.min()

        # vv = mb.create_meshset()
        # mb.add_entities(vv, all_volumes)
        # mb.write_file('outro.vtk', [vv])


        mb.tag_set_data(self.tags['l1_ID'], volumes, list_L1_ID)
        mb.tag_set_data(self.tags['l2_ID'], volumes, list_L2_ID)
        mb.tag_set_data(self.tags['l3_ID'], volumes, list_L3_ID)

    def generate_adm_mesh(self, mb, all_volumes, loop=0):

        nn = len(all_volumes)
        meshsets_nv1 = set() # volumes do nivel 1 que sao nivel 1
        meshsets_nv2 = set() # meshsets do nivel 2
        meshsets_nv3 = set() # meshsets do nivel 3

        list_L1_ID = []
        list_L2_ID = []
        list_L3_ID = []
        volumes = []

        finos = mb.tag_get_data(self.tags['finos'], 0, flat=True)[0]
        finos = set(mb.get_entities_by_handle(finos))
        intermediarios2 = set(rng.subtract(self.mtu.get_bridge_adjacencies(rng.Range(finos), 2, 3), rng.Range(finos)))
        intermediarios = (set(self.intermediarios) - finos) | intermediarios2
        ######################################################################
        # ni = ID do elemento no nível i
        n1=0
        n2=0
        n_vols = 0
        meshset_by_L2 = mb.get_child_meshsets(self.L2_meshset)
        print('\n')
        print("INICIOU GERAÇÃO DA MALHA ADM")
        print('\n')
        tempo0_ADM=time.time()
        t0 = tempo0_ADM
        for m2 in meshset_by_L2:
            #1
            meshsets_nv2aqui = set()
            n_vols_l3 = 0
            nivel3 = True
            nivel2 = False
            nivel1 = False
            meshset_by_L1 = mb.get_child_meshsets(m2)
            for m1 in meshset_by_L1:
                #2
                meshsets_nv2aqui.add(m1)
                elem_by_L1 = mb.get_entities_by_handle(m1)
                nn1 = len(elem_by_L1)
                n_vols += nn1
                n_vols_l3 += nn1
                int_finos = set(elem_by_L1) & finos # interseccao do elementos do meshset com os volumes do nivel1
                int_interm = set(elem_by_L1) & intermediarios # interseccao do elementos do meshset com os volumes do nivel2
                if int_finos:
                    #3
                    volumes.append(elem_by_L1)
                    meshsets_nv1.add(m1)
                    nivel3 = False
                    nivel1 = True
                    level = 1
                    list_L1_ID.append(np.arange(n1, n1+nn1))
                    list_L2_ID.append(np.arange(n2, n2+nn1))
                    list_L3_ID.append(np.repeat(level, nn1))
                    n1 += nn1
                    n2 += nn1
                #2
                elif int_interm:
                    #3
                    volumes.append(elem_by_L1)
                    meshsets_nv2.add(m1)
                    nivel3 = False
                    nivel2 = True
                    level = 2
                    list_L1_ID.append(np.repeat(n1, nn1))
                    list_L2_ID.append(np.repeat(n2, nn1))
                    list_L3_ID.append(np.repeat(level, nn1))
                    n1 += 1
                    n2 += 1
            #1
            if nivel3:
                #2
                level = 3
                for m1 in meshset_by_L1:
                    #3
                    elem_by_L1 = mb.get_entities_by_handle(m1)
                    nn1 = len(elem_by_L1)
                    volumes.append(elem_by_L1)
                    list_L1_ID.append(np.repeat(n1, nn1))
                    n1 += 1
                #2
                list_L2_ID.append(np.repeat(n2, n_vols_l3))
                list_L3_ID.append(np.repeat(level, n_vols_l3))
                n2 += 1
            #1
            elif nivel2:
                #2
                meshsets_fora = meshsets_nv2aqui - meshsets_nv2
                if nivel1:
                    #3
                    meshsets_fora = meshsets_fora - meshsets_nv1
                #2
                if meshsets_fora:
                    #3
                    for m1 in meshsets_fora:
                        #4
                        elem_by_L1 = mb.get_entities_by_handle(m1)
                        volumes.append(elem_by_L1)
                        nn1 = len(elem_by_L1)
                        level = 2
                        list_L1_ID.append(np.repeat(n1, nn1))
                        list_L2_ID.append(np.repeat(n2, nn1))
                        list_L3_ID.append(np.repeat(level, nn1))
                        n1 += 1
                        n2 += 1

        volumes = np.concatenate(volumes)
        list_L1_ID = np.concatenate(list_L1_ID)
        list_L2_ID = np.concatenate(list_L2_ID)
        list_L3_ID = np.concatenate(list_L3_ID)

        mb.tag_set_data(self.tags['l1_ID'], volumes, list_L1_ID)
        mb.tag_set_data(self.tags['l2_ID'], volumes, list_L2_ID)
        mb.tag_set_data(self.tags['l3_ID'], volumes, list_L3_ID)
