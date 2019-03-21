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


def get_OP_adm_nv1(mb, all_volumes, OP_AMS, ID_reord_tag, L1_ID_tag, L3_ID_tag, d1_tag, fine_to_primal1_classic_tag):
    elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    elems_nv1 = rng.subtract(all_volumes, elems_nv0)
    gids_nv1_elems_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, elems_nv1, flat=True))

    gids_elems_nv0 = mb.tag_get_data(ID_reord_tag, elems_nv0, flat=True)
    gids_adm_nv1_elems_nv0 = mb.tag_get_data(L1_ID_tag, elems_nv0, flat=True)
    all_ids_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, all_volumes, flat=True))
    OP_adm_nv1 = sp.lil_matrix((len(all_volumes), len(all_ids_nv1)))

    vertex_elems = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([d1_tag]), np.array([3]))
    id_nv1_op_classic_vertex_elems = mb.tag_get_data(fine_to_primal1_classic_tag, vertex_elems, flat=True)
    id_adm_nv1_vertex_elems = mb.tag_get_data(L1_ID_tag, vertex_elems, flat=True)

    OP_adm_nv1[:, id_adm_nv1_vertex_elems] = OP_AMS[:, id_nv1_op_classic_vertex_elems]

    # for id_adm, id_classic in zip(id_adm_nv1_vertex_elems, id_nv1_op_classic_vertex_elems):
    #     OP_adm_nv1[:,id_adm] = OP_AMS[:,id_classic]

    OP_adm_nv1[gids_elems_nv0] = sp.lil_matrix((len(gids_elems_nv0), len(all_ids_nv1)))
    OP_adm_nv1[gids_elems_nv0, gids_adm_nv1_elems_nv0] = np.ones(len(gids_elems_nv0))

    return OP_adm_nv1

def get_OP_adm_nv2(mb, all_volumes, wirebasket_ids_nv1, OP_AMS2, L1_ID_tag, L2_ID_tag, L3_ID_tag, primal_id_tag1, primal_id_tag2):
    # elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
    # elems_nv1 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))

    # elems_nv2 = rng.subtract(all_volumes, rng.unite(elems_nv0, elems_nv1))
    # verts_nv2 = rng.subtract(elems_verts_nv2, elems_nv0)



    all_ids_adm_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, all_volumes, flat=True))
    all_ids_adm_nv2 = np.unique(mb.tag_get_data(L2_ID_tag, all_volumes, flat=True))
    OP_adm_nv2 = sp.lil_matrix((len(all_ids_adm_nv1), len(all_ids_adm_nv2)))

    meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))

    meshsets_nv1_que_estao_no_nv2 = []
    id_adm_lv2_dos_meshsets_nv1_que_estao_no_nv2 = []
    id_adm_lv1_dos_meshsets_nv1_que_estao_no_nv2 = []
    todos_meshsets_que_estao_no_nivel_1 = []
    ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1 = []

    for m2 in meshsets_nv2:
        childs = mb.get_child_meshsets(m2)

        for m in childs:
            # mb.add_parent_meshset(m, m2)
            elems = mb.get_entities_by_handle(m)
            id_adm_nv1 = np.unique(mb.tag_get_data(L1_ID_tag, elems, flat=True))
            if len(id_adm_nv1) > 1:
                continue
            todos_meshsets_que_estao_no_nivel_1.append(m)
            ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1.append(id_adm_nv1[0])

        elems = mb.get_entities_by_handle(m2)
        id_lv2 = np.unique(mb.tag_get_data(L2_ID_tag, elems, flat=True))
        if len(id_lv2) > 1:
            continue
        meshsets_nv1_que_estao_no_nv2.append(childs)
        id_adm_lv2_dos_meshsets_nv1_que_estao_no_nv2.append(id_lv2[0])
        ids_adm_lv1 = []
        for m in childs:
            elems_1 = mb.get_entities_by_handle(m)
            id_adm_lv1 = np.unique(mb.tag_get_data(L1_ID_tag, elems_1, flat=True))
            ids_adm_lv1.append(id_adm_lv1)
        id_adm_lv1_dos_meshsets_nv1_que_estao_no_nv2.append(ids_adm_lv1[:])

    ncs_de_todos_meshsets_que_estao_no_nivel_1 = mb.tag_get_data(primal_id_tag1, todos_meshsets_que_estao_no_nivel_1, flat=True)
    map_ncs_de_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1 = dict(zip(ncs_de_todos_meshsets_que_estao_no_nivel_1, ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1))
    map_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1 = dict(zip(todos_meshsets_que_estao_no_nivel_1, ids_adm_nv1_de_todos_meshsets_que_estao_no_nivel_1))

    nc_vertex_elems = wirebasket_ids_nv1[3]
    meshsets_vertex_elems_nv1 = [mb.get_entities_by_type_and_tag(
        mb.get_root_set(), types.MBENTITYSET, np.array([primal_id_tag1]),
        np.array([i]))[0] for i in nc_vertex_elems]

    vertices_pi_chapeu = []
    ids_lv2_vertices_pi_chapeu = []
    for m in meshsets_vertex_elems_nv1:
        if m in todos_meshsets_que_estao_no_nivel_1:
            vertices_pi_chapeu.append(m)
            elems = mb.get_entities_by_handle(m)
            id_lv2_adm = np.unique(mb.tag_get_data(L2_ID_tag, elems, flat=True))
            ids_lv2_vertices_pi_chapeu.append(id_lv2_adm)

    for i, m in enumerate(vertices_pi_chapeu):
        id_adm_lv2_vertice = ids_lv2_vertices_pi_chapeu[i]
        parent_meshset = mb.get_parent_meshsets(m)
        nc2 = mb.tag_get_data(primal_id_tag2, parent_meshset, flat=True)[0]
        col_op2 = OP_AMS2[:,nc2]
        indices = sp.find(col_op2)
        nc_vert_pi_chapeu_nv1 = mb.tag_get_data(primal_id_tag1, m, flat=True)[0]
        # print(indices)
        # print(nc_vert_pi_chapeu_nv1)
        # import pdb; pdb.set_trace()
        lines = []
        vals = []
        for j, ind in enumerate(indices[0]):
            if ind not in ncs_de_todos_meshsets_que_estao_no_nivel_1:
                continue
            id_adm_nv1 = map_ncs_de_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1[ind]
            lines.append(id_adm_nv1)
            vals.append(indices[2][j])
        col = np.repeat(id_adm_lv2_vertice, len(vals)).astype(np.int32)
        lines = np.array(lines).astype(np.int32)
        vals = np.array(vals)

        OP_adm_nv2[lines, col] = vals

    todos = rng.Range(todos_meshsets_que_estao_no_nivel_1)

    for meshsets in  meshsets_nv1_que_estao_no_nv2:
        todos = rng.subtract(todos, meshsets)

    for m in todos:
        elems = mb.get_entities_by_handle(m)
        id_adm_2 = np.unique(mb.tag_get_data(L2_ID_tag, elems, flat=True))[0]
        id_adm_1 = map_todos_meshsets_que_estao_no_nivel_1_in_id_adm_1[m]
        OP_adm_nv2[id_adm_1] = np.zeros(len(all_ids_adm_nv2))
        OP_adm_nv2[id_adm_1, id_adm_2] = 1.0

    elems_nv0 = mb.get_entities_by_type_and_tag(
        mb.get_root_set(), types.MBHEX, np.array([L3_ID_tag]),
        np.array([1]))

    ids_adm_lv2_elems_nv0 = mb.tag_get_data(L2_ID_tag, elems_nv0, flat=True)
    ids_adm_lv1_elems_nv0 = mb.tag_get_data(L1_ID_tag, elems_nv0, flat=True)

    OP_adm_nv2[ids_adm_lv1_elems_nv0, ids_adm_lv2_elems_nv0] = np.ones(len(elems_nv0))



    return OP_adm_nv2
