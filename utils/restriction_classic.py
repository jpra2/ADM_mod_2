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
from scipy.sparse import linalg


def get_OR_classic_nv1(mb, all_volumes, L1_ID_tag, ID_reord_tag, L3_ID_tag):
    elems_nv0 = mb.get_entities_by_type_and_tag()
