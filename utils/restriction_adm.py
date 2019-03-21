import numpy as np
# from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
# import time
# import pyximport; pyximport.install()
import os
# from PyTrilinos import Epetra, AztecOO, EpetraExt  # , Amesos
# import math
# import os
# import shutil
# import random
import sys
# import configparser
import io
import yaml
import scipy.sparse as sp

def get_OR_adm_nv1(mb, all_volumes, ID_reord_tag, L1_ID_tag, L3_ID_tag):
    elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    elems_nv1 = rng.subtract(all_volumes, elems_nv0)
    gids_nv1_elems_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, elems_nv1, flat=True))
    gids_elems_nv0 = mb.tag_get_data(ID_reord_tag, elems_nv0, flat=True)
    gids_nv1_elems_nv0 = mb.tag_get_data(L1_ID_tag, elems_nv0, flat=True)
    all_ids_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, all_volumes, flat=True))

    OR = sp.lil_matrix((len(all_ids_nv1), len(all_volumes)))
    OR[gids_nv1_elems_nv0, gids_elems_nv0] = np.ones(len(elems_nv0))

    ms1 = set()

    for id in gids_nv1_elems_nv1:
        elems = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L1_ID_tag]), np.array([id]))
        gids_nv0_elems = mb.tag_get_data(ID_reord_tag, elems, flat=True)
        OR[np.repeat(id, len(elems)), gids_nv0_elems] = np.ones(len(elems))

    return OR

def get_OR_adm_nv2(mb, all_volumes, L1_ID_tag, L2_ID_tag, L3_ID_tag, level, fine_to_primal1_classic_tag, fine_to_primal2_classic_tag, primal_id_tag1, primal_id_tag2):
    ms0 = set()
    all_ids_lv1_adm = np.unique(mb.tag_get_data(L1_ID_tag, all_volumes, flat=True))
    all_ids_lv2_adm = np.unique(mb.tag_get_data(L2_ID_tag, all_volumes, flat=True))

    OR_adm = sp.lil_matrix((len(all_ids_lv2_adm), len(all_ids_lv1_adm)))

    elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    ids_nv1_adm = mb.tag_get_data(L1_ID_tag, elems_nv0, flat=True)
    ids_nv2_adm = mb.tag_get_data(L2_ID_tag, elems_nv0, flat=True)
    OR_adm[ids_nv2_adm, ids_nv1_adm] = np.ones(len(ids_nv1_adm))

    elems_nv1 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
    ncs1 = np.unique(mb.tag_get_data(fine_to_primal1_classic_tag, elems_nv1, flat=True))
    ids_nv1_adm = []
    ids_nv2_adm = []

    for nc in ncs1:
        meshset = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([nc]))
        elems = mb.get_entities_by_handle(meshset[0])
        id_adm_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, elems, flat=True))
        if len(id_adm_nv1) > 1:
            print('erro')
            import pdb; pdb.set_trace()
        id_adm_nv1 = id_adm_nv1[0]
        id_adm_nv2 = np.unique(mb.tag_get_data(L2_ID_tag, elems, flat=True))[0]
        ids_nv1_adm.append(id_adm_nv1)
        ids_nv2_adm.append(id_adm_nv2)

    OR_adm[ids_nv2_adm, ids_nv1_adm] = np.ones(len(ids_nv1_adm))

    elems_nv2 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([3]))
    ncs2 = np.unique(mb.tag_get_data(fine_to_primal2_classic_tag, elems_nv2, flat=True))
    ids_nv1_adm = []
    ids_nv2_adm = []

    for nc in ncs2:
        meshset = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([nc]))
        elems_2 = mb.get_entities_by_handle(meshset[0])
        id_adm_nv2 = np.unique(mb.tag_get_data(L2_ID_tag, elems_2, flat=True))
        if len(id_adm_nv2) > 1:
            print('erro')
            import pdb; pdb.set_trace()
        childs = mb.get_child_meshsets(meshset[0])
        for child in childs:
            elems = mb.get_entities_by_handle(child)
            id_adm_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, elems, flat=True))[0]
            ids_nv1_adm.append(id_adm_nv1)
            ids_nv2_adm.append(id_adm_nv2)

    OR_adm[ids_nv2_adm, ids_nv1_adm] = np.ones(len(ids_nv1_adm))

    return OR_adm
