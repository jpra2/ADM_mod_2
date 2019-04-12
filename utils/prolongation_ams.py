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
from scipy.sparse import linalg, find, csc_matrix
import time

__all__ = ['get_op_AMS_TPFA']

def get_op_AMS_TPFA_dep(T_mod, wirebasket_numbers):
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
    rr = OP.sum(axis=1)

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

def lu_inv4(M,lines):
    M = M.tocsc()
    lines=np.array(lines)
    cols=lines
    L=len(lines)
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        l=lines
        c=range(len(l))
        d=np.repeat(1,L)
        B=sp.csr_matrix((d,(l,c)),shape=(M.shape[0],L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            l=lines[s*i:s*(i+1)]
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(sp.hstack([inversa,csc_matrix(LU.solve(B))]))
            print(time.time()-tinv,i*s,'/',len(lines),'/',M.shape[0],"tempo de inversão")

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(sp.hstack([inversa,csc_matrix(LU.solve(B))]))
    tk1=time.time()
    f=find(inversa.tocsr())
    l=f[0]
    cc=f[1]
    d=f[2]
    pos_to_col=dict(zip(range(len(cols)),cols))
    cg=[pos_to_col[c] for c in cc]
    inversa=csc_matrix((d,(l,cg)),shape=(M.shape[0],M.shape[0]))
    print(tk1-tinv,L,time.time()-tk1,len(lines),'/',M.shape[0],"tempo de inversão")
    return inversa

def get_op_AMS_TPFA(As):
    # ids_arestas=np.where(Aev.sum(axis=1)==0)[0]
    # ids_arestas_slin_m0=np.setdiff1d(range(na),ids_arestas)
    ids_arestas_slin_m0 = np.nonzero(As['Aev'].sum(axis=1))[0]

    # ids_faces=np.where(Afe.sum(axis=1)==0)[0]
    # ids_faces_slin_m0=np.setdiff1d(range(nf),ids_faces)
    ids_faces_slin_m0 = np.nonzero(As['Afe'].sum(axis=1))[0]

    # ids_internos=np.where(Aif.sum(axis=1)==0)[0]
    # ids_internos_slin_m0=np.setdiff1d(range(ni),ids_internos)
    ids_internos_slin_m0=np.nonzero(As['Aif'].sum(axis=1))[0]

    invAee=lu_inv4(As['Aee'],ids_arestas_slin_m0)
    M2=-invAee*As['Aev']
    PAD=sp.vstack([M2,As['Ivv']])

    invAff=lu_inv4(As['Aff'],ids_faces_slin_m0)
    M3=-invAff*(As['Afe']*M2)
    PAD=sp.vstack([M3,PAD])

    invAii=lu_inv4(As['Aii'],ids_internos_slin_m0)
    PAD=sp.vstack([-invAii*(As['Aif']*M3),PAD])

    return PAD
