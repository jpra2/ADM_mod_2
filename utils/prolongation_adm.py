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

    for id_adm, id_classic in zip(id_adm_nv1_vertex_elems, id_nv1_op_classic_vertex_elems):
        OP_adm_nv1[:,id_adm] = OP_AMS[:,id_classic]

    OP_adm_nv1[gids_elems_nv0] = sp.lil_matrix((len(gids_elems_nv0), len(all_ids_nv1)))
    OP_adm_nv1[gids_elems_nv0, gids_adm_nv1_elems_nv0] = np.ones(len(gids_elems_nv0))

    return OP_adm_nv1
