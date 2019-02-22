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


def get_OR_classic_nv1(mb, all_volumes, ID_reord_tag, primal_id_tag1, fine_to_primal1_classic_tag):
    meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
    OR1 = sp.lil_matrix((len(meshsets_nv1), len(all_volumes)))
    for m in meshsets_nv1:
        elems = mb.get_entities_by_handle(m)
        gids = mb.tag_get_data(ID_reord_tag, elems, flat=True)
        nc = mb.tag_get_data(fine_to_primal1_classic_tag, elems, flat=True)
        OR1[nc, gids] = np.ones(len(elems))

    return OR1

def get_OR_classic_nv2(mb, primal_id_tag1, primal_id_tag2):
    meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))
    meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
    OR = sp.lil_matrix((len(meshsets_nv2), len(meshsets_nv1)))

    for m2 in meshsets_nv2:
        childs = mb.get_child_meshsets(m2)
        nc2 = mb.tag_get_data(primal_id_tag2, m2, flat=True)[0]
        nc1 = np.array([mb.tag_get_data(primal_id_tag1, child, flat=True)[0] for child in childs])
        nc2 = np.repeat(nc2, len(nc1))
        OR[nc2, nc1] = np.ones(len(nc1))

    return OR
