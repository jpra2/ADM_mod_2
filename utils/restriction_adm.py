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
