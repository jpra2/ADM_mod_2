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
from scipy.sparse import linalg, find, csc_matrix, vstack
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

def lu_inv2(M):
    L=M.shape[0]
    s=1000
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        t0=time.time()
        lc=range(L)
        d=np.repeat(1,L)
        B=csc_matrix((d,(lc,lc)),shape=(L,L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B))
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            t0=time.time()
            l=range(s*i,s*(i+1))
            B=csc_matrix((d,(l,c)),shape=(L,s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B))
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=range(s*n,L)
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(L,r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))
    #print(time.time()-tinv,M.shape[0],"tempo de inversão")
    return inversa

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

def solve_block_matrix(topology, pos_0, mb, k_eq_tag, n0):
    lgp=[]
    cgp=[]
    dgp=[]
    c0=0

    st=0
    ts=0
    ta=0
    tc=0

    fl=[]
    fc=[]
    fd=[]
    for cont in range(int(len(topology)/6)):
        t1=time.time()
        Gids=topology[6*cont]
        all_faces_topo=topology[6*cont+1]
        ADJs1=topology[6*cont+2]
        ADJs2=topology[6*cont+3]
        if pos_0 > 0:
            adjsg1=topology[6*cont+4]
            adjsg2=topology[6*cont+5]
            inds1=np.where(adjsg1<pos_0)[0]
            inds2=np.where(adjsg2<pos_0)[0]
            inds_elim=np.unique(np.concatenate([inds1,inds2]))
            all_faces_topo=np.delete(all_faces_topo,inds_elim)
            ADJs1=np.delete(ADJs1,inds_elim)
            ADJs2=np.delete(ADJs2,inds_elim)
        ks_all=np.array(mb.tag_get_data(k_eq_tag,np.array(all_faces_topo),flat=True))
        ts+=time.time()-t1
        t2=time.time()
        int1=np.where(ADJs1<len(Gids))
        int2=np.where(ADJs2<len(Gids))
        pos_int_i=np.intersect1d(int1,int2)
        pos_int_e1=np.setdiff1d(int1,pos_int_i)
        pos_int_e2=np.setdiff1d(int2,pos_int_i)

        Lid_1=ADJs1[pos_int_i]
        Lid_2=ADJs2[pos_int_i]
        ks=ks_all[pos_int_i]

        lines1=[]
        cols1=[]
        data1=[]

        lines1.append(Lid_1)
        cols1.append(Lid_2)
        data1.append(ks)

        lines1.append(Lid_2)
        cols1.append(Lid_1)
        data1.append(ks)

        lines1.append(Lid_1)
        cols1.append(Lid_1)
        data1.append(-ks)

        lines1.append(Lid_2)
        cols1.append(Lid_2)
        data1.append(-ks)

        Lid_1=ADJs1[pos_int_e1]
        ks=ks_all[pos_int_e1]
        lines1.append(Lid_1)
        cols1.append(Lid_1)
        data1.append(-ks)

        Lid_2=ADJs2[pos_int_e2]
        ks=ks_all[pos_int_e2]
        lines1.append(Lid_2)
        cols1.append(Lid_2)
        data1.append(-ks)


        lines1=np.concatenate(np.array(lines1))
        cols1=np.concatenate(np.array(cols1))
        data1=np.concatenate(np.array(data1))
        M_local=csc_matrix((data1,(lines1,cols1)),shape=(len(Gids),len(Gids)))
        ta+=time.time()-t2
        tinvert=time.time()
        try:
            inv_local=lu_inv2(M_local)
        except:
            import pdb; pdb.set_trace()



        st+=time.time()-tinvert

        t3=time.time()
        ml=find(inv_local)
        fl.append(ml[0]+c0)
        fc.append(ml[1]+c0)
        fd.append(ml[2])
        lgp.append(Gids-pos_0)
        tc+=time.time()-t3
        c0+=len(Gids)

    fl=np.concatenate(np.array(fl))
    fc=np.concatenate(np.array(fc))
    fd=np.concatenate(np.array(fd))

    m_loc=csc_matrix((fd,(fl,fc)),shape=(n0,n0))
    lgp=np.concatenate(np.array(lgp))
    cgp=range(n0)
    dgp=np.ones(len(lgp))
    permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(n0,n0))
    invMatrix=permut_g*m_loc*permut_g.transpose()

    return(invMatrix)

def get_op_AMS_TPFA_top(mb, faces_adjs_by_dual, intern_adjs_by_dual, ni, nf, k_eq_tag, As):

    invbAii=solve_block_matrix(intern_adjs_by_dual, 0, mb, k_eq_tag, ni)
    invbAff = solve_block_matrix(faces_adjs_by_dual, ni, mb, k_eq_tag, nf)
    ID_reordenado_tag = mb.tag_get_handle('ID_reord_tag')
    ids_arestas_slin_m0=np.nonzero(As['Aev'].sum(axis=1))[0]
    Aev = As['Aev']
    Ivv = As['Ivv']
    Aif = As['Aif']
    Afe = As['Afe']
    invAee=lu_inv4(As['Aee'].tocsc(), ids_arestas_slin_m0)
    M2=-invAee*Aev
    PAD=vstack([M2,Ivv])
    invAff=invbAff
    M3=-invAff*(Afe*M2)
    PAD=vstack([M3,PAD])
    invAii=invbAii
    PAD=vstack([-invAii*(Aif*M3),PAD])

    return PAD
