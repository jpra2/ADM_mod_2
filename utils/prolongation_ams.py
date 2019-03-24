import numpy as np
# from math import pi, sqrt
from pymoab import core, types, rng, topo_util, skinner
import time
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

def get_op_AMS_TPFA(T_mod, wirebasket_numbers):
    ni = wirebasket_numbers[0]
    nf = wirebasket_numbers[1]
    ne = wirebasket_numbers[2]
    nv = wirebasket_numbers[3]

    idsi = ni
    idsf = idsi+nf
    idse = idsf+ne
    idsv = idse+nv
    loc = [idsi, idsf, idse, idsv]

    ntot = sum(wirebasket_numbers)

    OP = sp.lil_matrix((ntot, nv))
    OP = insert_identity(OP, wirebasket_numbers)
    OP, M = step1(T_mod, OP, loc)
    OP, M = step2(T_mod, OP, loc, M)
    OP = step3(T_mod, OP, loc, M)
    # rr = OP.sum(axis=1)

    return OP

def insert_identity(op, wirebasket_numbers):
        nv = wirebasket_numbers[3]
        nne = sum(wirebasket_numbers) - nv
        lines = np.arange(nne, nne+nv).astype(np.int32)
        values = np.ones(nv)
        matrix = sp.lil_matrix((nv, nv))
        rr = np.arange(nv).astype(np.int32)
        matrix[rr, rr] = values

        op[lines] = matrix

        return op

def step1(t_mod, op, loc):
        """
        elementos de aresta
        """
        lim = 1e-13

        nnf = loc[1]
        nne = loc[2]
        nnv = loc[3]
        ne = nne - nnf
        nv = nnv - nne
        nf = loc[1] - loc[0]
        M = t_mod[nnf:nne, nnf:nne]
        M = linalg.spsolve(M.tocsc(copy=True), sp.identity(ne).tocsc())
        # M2 = -1*t_mod[nnf:nne, nne:nnv]
        # M = M.dot(M2)
        M = M.dot(-1*t_mod[nnf:nne, nne:nnv])


        op[nnf:nne] = M
        return op, M

def step2(t_mod, op, loc, MM):
    """
    elementos de face
    """
    nni = loc[0]
    nnf = loc[1]
    nne = loc[2]
    nnv = loc[3]
    ne = nne - nnf
    nv = nnv - nne
    nf = loc[1] - loc[0]
    ni = loc[0]

    M = t_mod[nni:nnf, nni:nnf]
    M = linalg.spsolve(M.tocsc(copy=True), sp.identity(nf).tocsc())
    # M2 = -1*t_mod[nni:nnf, nnf:nne] # nfxne
    # M = M.dot(M2)
    M = M.dot(-1*t_mod[nni:nnf, nnf:nne])
    M = M.dot(MM)

    op[nni:nnf] = M
    return op, M

def step3(t_mod, op, loc, MM):
    """
    elementos internos
    """
    nni = loc[0]
    nnf = loc[1]
    nne = loc[2]
    nnv = loc[3]
    ne = nne - nnf
    nv = nnv - nne
    nf = loc[1] - loc[0]
    ni = loc[0]

    M = t_mod[0:nni, 0:nni]
    M = linalg.spsolve(M.tocsc(copy=True), sp.identity(ni).tocsc())
    M = M.dot(-1*t_mod[0:nni, nni:nnf])
    M = M.dot(MM)


    op[0:nni] = M
    return op

def get_OP_AMS_TPFA_by_AS(As, wirebasket_numbers):
    
    ni = wirebasket_numbers[0]
    nf = wirebasket_numbers[1]
    ne = wirebasket_numbers[2]
    nv = wirebasket_numbers[3]

    nni = ni
    nnf = nni + nf
    nne = nnf + ne
    nnv = nne + nv

    lines = np.arange(nne, nnv).astype(np.int32)
    ntot = sum(wirebasket_numbers)
    op = sp.lil_matrix((ntot, nv))
    op[lines] = As['Ivv'].tolil()

    M = As['Aee']
    M = linalg.spsolve(M.tocsc(), sp.identity(ne).tocsc())
    M = M.dot(-1*As['Aev'])
    op[nnf:nne] = M.tolil()

    M2 = As['Aff']
    M2 = linalg.spsolve(M2.tocsc(), sp.identity(nf).tocsc())
    M2 = M2.dot(-1*As['Afe'])
    M = M2.dot(M)
    op[nni:nnf] = M.tolil()

    M2 = As['Aii']
    M2 = linalg.spsolve(M2.tocsc(), sp.identity(ni).tocsc())
    M2 = M2.dot(-1*As['Aif'])
    M = M2.dot(M)
    op[0:nni] = M.tolil()

    return op
