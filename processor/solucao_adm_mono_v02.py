import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
import yaml


parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

input_file = data_loaded['input_file']
ext_h5m_adm = input_file + '_malha_adm.h5m'

mb = core.Core()
root_set = mb.get_root_set()
mtu = topo_util.MeshTopoUtil(mb)
os.chdir(flying_dir)
mb.load_file(ext_h5m_adm)
list_names_tags = np.load('list_names_tags.npy')
os.chdir(parent_dir)

all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)

dict_tags = utpy.get_all_tags_2(mb, list_names_tags)
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'

def lu_inv(M):
    L=M.shape[0]
    s=1000
    if L<s:
        tinv=time.time()
        LU=linalg.splu(M)
        inversa=csc_matrix(LU.solve(np.eye(M.shape[0])))
        print(time.time()-tinv,M.shape[0],"tempo de inversão, ordem")
    else:
        div=1
        for i in range(1,int(L/s)+1):
            if L%i==0:
                div=i
        l=int(L/div)
        ident=np.eye(l)
        zeros=np.zeros((l,l),dtype=int)
        tinv=time.time()
        LU=linalg.splu(M)
        print(div,M.shape[0],"Num divisões, Tamanho")
        for j in range(div):
            for k in range(j):
                try:
                    B=np.concatenate([B,zeros])
                except NameError:
                    B=zeros
            if j==0:
                B=ident
            else:
                B=np.concatenate([B,ident])
            for i in range(div-j-1):
                B=np.concatenate([B,zeros])
            if j==0:
                inversa=csc_matrix(LU.solve(B))
                del(B)
            else:
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))
                del(B)
        print(time.time()-tinv,M.shape[0],div,"tempo de inversão, ordem")
    return inversa

tempo0_ADM = time.time()

primal_id_tag1 = dict_tags['PRIMAL_ID_1']
primal_id_tag2 = dict_tags['PRIMAL_ID_2']
fine_to_primal2_classic_tag = dict_tags['FINE_TO_PRIMAL2_CLASSIC']
fine_to_primal1_classic_tag = dict_tags['FINE_TO_PRIMAL1_CLASSIC']
D1_tag = dict_tags['d1']

meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))
n1 = len(meshsets_nv1)
n2 = len(meshsets_nv2)

internos=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))

ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)

nni=ni
nnf=nni+nf
nne=nnf+na
nnv=nne+nv

volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['P']]), np.array([None]))
volumes_n = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['Q']]), np.array([None]))


tmod1=time.time()
ID_reordenado_tag = dict_tags['ID_reord_tag']

lii=[]
lif=[]
lff=[]
lfe=[]
lee=[]
lev=[]

cii=[]
cif=[]
cff=[]
cfe=[]
cee=[]
cev=[]

dii=[]
dif=[]
dff=[]
dfe=[]
dee=[]
dev=[]

# index = list_names_tags.index('K_EQ')
# k_eq_tag = tags_1[index]
# index = list_names_tags.index('S_GRAV')
# s_grav_tag = tags_1[index]
area_tag = dict_tags['AREA']
perm_tag = dict_tags['PERM']

boundary_faces = mb.get_entities_by_handle(mb.tag_get_data(dict_tags['FACES_BOUNDARY'], 0, flat=True)[0])

gids2 = mb.tag_get_data(ID_reordenado_tag, all_volumes, flat=True)
map_global = dict(zip(all_volumes, gids2))
s = np.zeros(len(all_volumes))

print("def As")
ty=time.time()

for f in rng.subtract(all_faces, boundary_faces):
    keq, s_grav, adjs = oth.get_kequiv_by_face_quad(mb, mtu, f, perm_tag, area_tag)
    keq = 1
    Gid_1 = map_global[adjs[0]]
    Gid_2 = map_global[adjs[1]]

    if Gid_1<ni and Gid_2<ni:
        lii.append(Gid_1)
        cii.append(Gid_2)
        dii.append(keq)

        lii.append(Gid_2)
        cii.append(Gid_1)
        dii.append(keq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-keq)

        lii.append(Gid_2)
        cii.append(Gid_2)
        dii.append(-keq)

    elif Gid_1<ni and Gid_2>ni and Gid_2<ni+nf:
        lif.append(Gid_1)
        cif.append(Gid_2-ni)
        dif.append(keq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-keq)

    elif Gid_2<ni and Gid_1>ni and Gid_1<ni+nf:
        lif.append(Gid_2)
        cif.append(Gid_1-ni)
        dif.append(keq)

        lii.append(Gid_2)
        cii.append(Gid_2)
        dii.append(-keq)

    elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni and Gid_2<ni+nf:
        lff.append(Gid_1-ni)
        cff.append(Gid_2-ni)
        dff.append(keq)

        lff.append(Gid_2-ni)
        cff.append(Gid_1-ni)
        dff.append(keq)

        lff.append(Gid_1-ni)
        cff.append(Gid_1-ni)
        dff.append(-keq)

        lff.append(Gid_2-ni)
        cff.append(Gid_2-ni)
        dff.append(-keq)

    elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni+nf and Gid_2<ni+nf+na:
        lfe.append(Gid_1-ni)
        cfe.append(Gid_2-ni-nf)
        dfe.append(keq)

        lff.append(Gid_1-ni)
        cff.append(Gid_1-ni)
        dff.append(-keq)

    elif Gid_2>=ni and Gid_2<ni+nf and Gid_1>=ni+nf and Gid_1<ni+nf+na:
        lfe.append(Gid_2-ni)
        cfe.append(Gid_1-ni-nf)
        dfe.append(keq)

        lff.append(Gid_2-ni)
        cff.append(Gid_2-ni)
        dff.append(-keq)

    elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf and Gid_2<ni+nf+na:
        lee.append(Gid_1-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(keq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(keq)

        lee.append(Gid_1-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(-keq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(-keq)

    elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf+na:
        lev.append(Gid_1-ni-nf)
        cev.append(Gid_2-ni-nf-na)
        dev.append(keq)

        lee.append(Gid_1-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(-keq)

    elif Gid_2>=ni+nf and Gid_2<ni+nf+na and Gid_1>=ni+nf+na:
        lev.append(Gid_2-ni-nf)
        cev.append(Gid_1-ni-nf-na)
        dev.append(keq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(-keq)

    s[map_global[adjs[0]]] += s_grav
    s[map_global[adjs[1]]] -= s_grav

print("took: ",time.time()-ty)
print("get As")
ty=time.time()
Aii=csc_matrix((dii,(lii,cii)),shape=(ni,ni))
Aif=csc_matrix((dif,(lif,cif)),shape=(ni,nf))
Aff=csc_matrix((dff,(lff,cff)),shape=(nf,nf))
Afe=csc_matrix((dfe,(lfe,cfe)),shape=(nf,na))
Aee=csc_matrix((dee,(lee,cee)),shape=(na,na))
Aev=csc_matrix((dev,(lev,cev)),shape=(na,nv))
Ivv=scipy.sparse.identity(nv)

print("took: ",time.time()-ty)

print("get_OP_AMS")
ty=time.time()

#th=time.time()
#M2=-linalg.inv(Aee)*Aev
#print(time.time()-th,"Direto")


M2=-lu_inv(Aee)*Aev
P=vstack([M2,Ivv]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

M3=-lu_inv(Aff)*Afe*M2
del(M2)
P=vstack([M3,P])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)
P=vstack([-lu_inv(Aii)*Aif*M3,P]) ##P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)
del(M3)
print("took to get_OP_AMS",time.time()-ty)
AMS_TO_ADM={}
i=0
for v in vertices:
    ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
    AMS_TO_ADM[str(i)] = ID_ADM
    i+=1

lines=[]
cols=[]
data=[]
ty=time.time()
print("organize OP_ADM")
#OP_ADM=np.zeros((len(M1.all_volumes),n1))

###for v in M1.all_volumes:
###    nivel=int(M1.mb.tag_get_data(L3_ID_tag,v))
###    d1=int(M1.mb.tag_get_data(D1_tag,v))
###    ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
###    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
###    if nivel==1 or d1==3:
###        lines.append(ID_global)
###        cols.append(ID_ADM)
###        data.append(1)
###
###        #OP_ADM[ID_global][ID_ADM]=1
###    else:
###        for i in range(nv):
###            p=P[ID_global][i]
###            if p>0:
###                id_ADM=AMS_TO_ADM[str(i)]
###                lines.append(ID_global)
###                cols.append(id_ADM)
###                data.append(p)
                #OP_ADM[ID_global][id_ADM]=p
################
P=csc_matrix(P)
ty=time.time()
print("iniciou____")
nivel_1 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['l3_ID']]), np.array([1]))
print("get nivel 1___")
t0 = time.time()
matriz=scipy.sparse.find(P)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)

for v in nivel_1:
    ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
    ID_global=int(mb.tag_get_data(ID_reordenado_tag,v))
    lines.append(ID_global)
    cols.append(ID_ADM)
    data.append(1)

    dd=np.where(LIN==ID_global)
    LIN=np.delete(LIN,dd,axis=0)
    COL=np.delete(COL,dd,axis=0)
    DAT=np.delete(DAT,dd,axis=0)

print("set_nivel 1")

print("loop", time.time()-ty)

ID_ADM=[AMS_TO_ADM[str(k)] for k in COL]
lines=np.concatenate([lines,LIN])
cols=np.concatenate([cols,ID_ADM])
data=np.concatenate([data,DAT])
del(COL)
del(LIN)
del(DAT)
del(ID_ADM)
#for k in range(len(LIN)):
#    lines.append(LIN[k])
#    ID_ADM=AMS_TO_ADM[str(COL[k])]
#    cols.append(ID_ADM)
#    data.append(DAT[k])
###############

gids_nv1_adm = np.unique(mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True))
n1_adm = len(gids_nv1_adm)

print("op_adm", time.time()-ty)
OP_ADM=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),n1_adm))

print("took",time.time()-t0)
print("Obtendo OP_ADM_2")
t0=time.time()
#OR_ADM=np.zeros((n1,len(M1.all_volumes)),dtype=np.int)
tmod3=time.time()
lines=[]
cols=[]
data=[]
for v in all_volumes:
     elem_Global_ID = int(mb.tag_get_data(ID_reordenado_tag, v, flat=True))
     elem_ID1 = int(mb.tag_get_data(dict_tags['l1_ID'], v, flat=True))
     lines.append(elem_ID1)
     cols.append(elem_Global_ID)
     data.append(1)
     #OR_ADM[elem_ID1][elem_Global_ID]=1
OR_ADM=csc_matrix((data,(lines,cols)),shape=(n1_adm,len(all_volumes)))

print(time.time()-tmod3,"Tmod3 _________")

#OR_AMS=np.zeros((nv,len(M1.all_volumes)),dtype=np.int)
lines=[]
cols=[]
data=[]
for v in all_volumes:
     elem_Global_ID = int(mb.tag_get_data(ID_reordenado_tag, v))
     AMS_ID = int(mb.tag_get_data(fine_to_primal1_classic_tag, v))
     lines.append(AMS_ID)
     cols.append(elem_Global_ID)
     data.append(1)
     #OR_AMS[AMS_ID][elem_Global_ID]=1
OR_AMS=csc_matrix((data,(lines,cols)),shape=(nv,len(all_volumes)))

###
###COL_TO_AMS={}
###i=0
###for v in vertices:
###    ID_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v)
###    COL_TO_AMS[str(i)] = ID_AMS
###    i+=1
###
####OP_AMS=np.zeros((len(M1.all_volumes),nv))
tmod4=time.time()
###lines=[]
###cols=[]
###data=[]
###for v in M1.all_volumes:
###    elem_Global_ID = int(M1.mb.tag_get_data(M1.ID_reordenado_tag, v))
###    for i in range(len(P[elem_Global_ID])):
###        p=P[elem_Global_ID][i]
###        if p>0:
###            ID_AMS=int(COL_TO_AMS[str(i)])
###            lines.append(elem_Global_ID)
###            cols.append(ID_AMS)
###            data.append(p)
###print(time.time()-tmod4,"tmod4 ________")
###            #OP_AMS[elem_Global_ID][ID_AMS]=p
###OP_AMS=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),nv))

OP_AMS=P
print(time.time()-tmod4,"tmod4 ________")
t_ass=time.time()

#T=np.zeros((len(M1.all_volumes),len(M1.all_volumes)))
lines=[]
cols=[]
data=[]
for f in rng.subtract(all_faces, boundary_faces):
    keq, s_grav, adjs = oth.get_kequiv_by_face_quad(mb, mtu, f, perm_tag, area_tag)
    Gid_1 = map_global[adjs[0]]
    Gid_2 = map_global[adjs[1]]

    lines.append(Gid_1)
    cols.append(Gid_2)
    data.append(keq)
    #T[Gid_1][Gid_2]=1
    lines.append(Gid_2)
    cols.append(Gid_1)
    data.append(keq)
    #T[Gid_2][Gid_1]=1
    lines.append(Gid_1)
    cols.append(Gid_1)
    data.append(-keq)
    #T[Gid_1][Gid_1]-=1
    lines.append(Gid_2)
    cols.append(Gid_2)
    data.append(-keq)
    #T[Gid_2][Gid_2]-=1

T=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),len(all_volumes)))
t_assembly=time.time()-t_ass
#----------------------------------------------------
T_AMS=OR_AMS*T*OP_AMS
T_ADM=OR_ADM*T*OP_ADM

#tmod12=time.time()
#
#int_meshset=M1.mb.create_meshset()
#fac_meshset=M1.mb.create_meshset()
#are_meshset=M1.mb.create_meshset()
#ver_meshset=M1.mb.create_meshset()
#
#
#for v in vertices:
#    vafi = M1.mb.tag_get_data(D2_tag, v)[0] #Vértice ->3 aresta->2
#    if vafi==0:
#        M1.mb.add_entities(int_meshset,[v])
#    elif vafi==1:
#        M1.mb.add_entities(fac_meshset,[v])
#    elif vafi==2:
#        M1.mb.add_entities(are_meshset,[v])
#    elif vafi==3:
#        M1.mb.add_entities(ver_meshset,[v])
#
##G=np.zeros((nv,nv))

#inte=M1.mb.get_entities_by_handle(int_meshset)
#fac=M1.mb.get_entities_by_handle(fac_meshset)
#are=M1.mb.get_entities_by_handle(are_meshset)
#ver=M1.mb.get_entities_by_handle(ver_meshset)
#print(time.time()-tmod12,"s/mod_____12  ")
#
#del(inte)
#del(fac)
#del(are)
#del(ver)
D2_tag = dict_tags['d2']
v=mb.create_meshset()
mb.add_entities(v,vertices)
tmod12=time.time()
inte=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

print(time.time()-tmod12,"modificação 12")
lines=[]
cols=[]
data=[]

nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)
for i in range(nint):
    v=inte[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(i)
    cols.append(ID_AMS)
    data.append(1)
    #G[i][ID_AMS]=1
i=0
for i in range(nfac):
    v=fac[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+i][ID_AMS]=1
i=0
for i in range(nare):
    v=are[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+nfac+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+nfac+i][ID_AMS]=1
i=0

for i in range(nver):
    v=ver[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+nfac+nare+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+nfac+nare+i][ID_AMS]=1

G=csc_matrix((data,(lines,cols)),shape=(nv,nv))


W_AMS=G*T_AMS*(G.transpose())

MPFA_NO_NIVEL_2=mb.tag_get_data(dict_tags['MPFA'], 0, flat=True)[0]


#-------------------------------------------------------------------------------
ni=nint
nf=nfac
na=nare
nv=nver

Aii=W_AMS[0:ni,0:ni]
Aif=W_AMS[0:ni,ni:ni+nf]
Aie=W_AMS[0:ni,ni+nf:ni+nf+na]
Aiv=W_AMS[0:ni,ni+nf+na:ni+nf+na+nv]

lines=[]
cols=[]
data=[]
if MPFA_NO_NIVEL_2 ==False:
    for i in range(ni):
        lines.append(i)
        cols.append(i)
        val = Aie.sum(axis=1)[i]+Aiv.sum(axis=1)[i]
        data.append(float(val[0][0]))
    S=csc_matrix((data,(lines,cols)),shape=(ni,ni))
    Aii += S
    del(S)

Afi=W_AMS[ni:ni+nf,0:ni]
Aff=W_AMS[ni:ni+nf,ni:ni+nf]
Afe=W_AMS[ni:ni+nf,ni+nf:ni+nf+na]
Afv=W_AMS[ni:ni+nf,ni+nf+na:ni+nf+na+nv]

lines=[]
cols=[]
data_fi=[]
data_fv=[]
for i in range(nf):
    lines.append(i)
    cols.append(i)
    data_fi.append(float(Afi.sum(axis=1)[i]))
    data_fv.append(float(Afi.sum(axis=1)[i]))

Sfi=csc_matrix((data_fi,(lines,cols)),shape=(nf,nf))
Aff += Sfi
if MPFA_NO_NIVEL_2==False:
    Sfv=csc_matrix((data_fv,(lines,cols)),shape=(nf,nf))
    Aff +=Sfv

Aei=W_AMS[ni+nf:ni+nf+na,0:ni]
Aef=W_AMS[ni+nf:ni+nf+na,ni:ni+nf]
Aee=W_AMS[ni+nf:ni+nf+na,ni+nf:ni+nf+na]
Aev=W_AMS[ni+nf:ni+nf+na,ni+nf+na:ni+nf+na+nv]
lines=[]
cols=[]
data=[]
for i in range(na):
    lines.append(i)
    cols.append(i)
    data.append(float(Aei.sum(axis=1)[i])+float(Aef.sum(axis=1)[i]))
S=csc_matrix((data,(lines,cols)),shape=(na,na))
Aee += S

Ivv=scipy.sparse.identity(nv)
M2=-csc_matrix(lu_inv(Aee))*Aev
P2=vstack([M2,Ivv])

invAff=lu_inv(Aff)

if MPFA_NO_NIVEL_2:
    M3=-invAff*Afe*M2-invAff*Afv
    P2=vstack([M3,P2])
else:
    Mf=-Aff*Afe*M2
    P2=vstack([Mf,P2])

if MPFA_NO_NIVEL_2:

    M3=lu_inv(Aii)*(-Aif*M3+Aie*lu_inv(Aee)*Aev-Aiv)
    P2=vstack([M3,P2])
else:
    P2=vstack([-lu_inv(Aii)*Aif*Mf,P2])

OP_AMS_2=G.transpose()*P2

COL_TO_ADM_2={}
# ver é o meshset dos vértices da malha dual grossa
for v in ver:
    ID_AMS=int(mb.tag_get_data(fine_to_primal2_classic_tag,v))
    ID_ADM=int(mb.tag_get_data(dict_tags['l2_ID'],v))
    COL_TO_ADM_2[str(ID_AMS)] = ID_ADM
P2=P2.toarray()
#---Vértices é o meshset dos véttices da malha dual do nivel intermediário------

#OP_ADM_2=np.zeros((len(T_ADM),n2))

gids_adm_nv2 = np.unique(mb.tag_get_data(dict_tags['l2_ID'], all_volumes, flat=True))
n2_adm = len(gids_adm_nv2)

lines=[]
cols=[]
data=[]
print("took",time.time()-t0)
print("Resolvendo sistema ADM_2")
t0=time.time()
My_IDs_2=[]
for v in all_volumes:
    ID_global=int(mb.tag_get_data(dict_tags['l1_ID'],v))
    if ID_global not in My_IDs_2:
        My_IDs_2.append(ID_global)
        ID_ADM=int(mb.tag_get_data(dict_tags['l2_ID'],v))
        nivel=mb.tag_get_data(dict_tags['l3_ID'],v)
        d1=mb.tag_get_data(D2_tag,v)
        ID_AMS = int(mb.tag_get_data(fine_to_primal2_classic_tag, v))
        # nivel<3 refere-se aos volumes na malha fina (nivel=1) e intermédiária (nivel=2)
        # d1=3 refere-se aos volumes que são vértice na malha dual de grossa
        if nivel<3:
            lines.append(ID_global)
            cols.append(ID_ADM)
            data.append(1)
            #OP_ADM_2[ID_global][ID_ADM]=1
        else:
            for i in range(len(P2[ID_AMS])):
                p=P2[ID_AMS][i]
                if p>0:
                    id_ADM=COL_TO_ADM_2[str(i)]
                    lines.append(ID_global)
                    cols.append(id_ADM)
                    data.append(float(p))
                    #OP_ADM_2[ID_global][id_ADM]=p
OP_ADM_2=csc_matrix((data,(lines,cols)),shape=(n1_adm,n2_adm))

#OR_ADM_2=np.zeros((n2,len(T_ADM)),dtype=np.int)
lines=[]
cols=[]
data=[]
for v in all_volumes:
    elem_ID2 = int(mb.tag_get_data(dict_tags['l2_ID'], v, flat=True))
    elem_Global_ID = int(mb.tag_get_data(dict_tags['l1_ID'], v, flat=True))
    lines.append(elem_ID2)
    cols.append(elem_Global_ID)
    data.append(1)
    #OR_ADM_2[elem_ID2][elem_Global_ID]=1
OR_ADM_2=csc_matrix((data,(lines,cols)),shape=(n2_adm,n1_adm))
T_ADM_2=OR_ADM_2*T_ADM*OP_ADM_2


#-------------------------------------------------------------------------------
# Insere condições de dirichlet nas matrizes
###tmod=time.time()
###for i in range(len(volumes_d)):
###    v=volumes_d[i]
###    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
###    #ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
###    ID_ADM_2=int(M1.mb.tag_get_data(L2_ID_tag,v))
###    T[ID_global]=np.zeros(T.shape[0])
###    T[ID_global,ID_global]=1
###    #for j in range(len(M1.all_volumes)):
###    #    if T[ID_global,j]>0:
###    #        T[ID_global,j]=0
###    #    if j==ID_global:
###    #        T[ID_global,j]=1
###    #for j in range(len(T_ADM[i])):
###    #    T_ADM[ID_ADM][j]=0
###    #    if j==ID_ADM:
###    #        T_ADM[ID_ADM][j]=1
###    T_ADM_2[ID_ADM_2]=np.zeros(T_ADM_2.shape[0])
###    T_ADM_2[ID_ADM_2,ID_ADM_2]=1
###    #for j in range(n2):
###    #    if T_ADM_2[ID_ADM_2,j]>0:
###    #        T_ADM_2[ID_ADM_2,j]=0
###    #    if j==ID_ADM_2:
###    #        T_ADM_2[ID_ADM_2,j]=1
###
###print(time.time()-tmod,"modificação ____")

press_tag = dict_tags['P']
vaz_tag = dict_tags['Q']

volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([press_tag]), np.array([None]))
volumes_n = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([vaz_tag]), np.array([None]))

tmod = time.time()
ID_global = mb.tag_get_data(ID_reordenado_tag, volumes_d, flat=True)
#ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
ID_ADM_2 = mb.tag_get_data(dict_tags['l2_ID'], volumes_d, flat=True)
T[ID_global] = scipy.sparse.csc_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global] = np.ones(len(ID_global))

T_ADM_2[ID_ADM_2]=scipy.sparse.csc_matrix((len(ID_ADM_2),T_ADM_2.shape[0]))
T_ADM_2[ID_ADM_2,ID_ADM_2]=np.ones(len(ID_ADM_2))
print(time.time()-tmod,"jjshhhdh______")

# Gera a matriz dos coeficientes
#b=np.zeros((len(M1.all_volumes),1))
lines=[]
cols=[]
data=[]
for d in volumes_d:
    # ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,d))
    ID_global=map_global[d]
    press = mb.tag_get_data(press_tag, d, flat=True)[0]
    lines.append(ID_global)
    cols.append(0)
    data.append(press)
    #b[ID_global]=press
for n in volumes_n:
    # ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,n))
    ID_global=map_global[n]
    vazao = mb.tag_get_data(vaz_tag, n, flat=True)[0]
    lines.append(ID_global)
    cols.append(0)
    data.append(vazao)
    #b[ID_global]=vazao
# del(b)
b=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))
b_ADM_2=OR_ADM_2*OR_ADM*b

t0=time.time()
SOL_ADM_2=linalg.spsolve(T_ADM_2,b_ADM_2)
print("resolveu ADM_2: ",time.time()-t0)
print("Prolongando sol ADM")
t0=time.time()
SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM_2
print("Prolongou sol ADM: ",time.time()-t0)
print("TEMPO TOTAL PARA SOLUÇÃO ADM:", time.time()-tempo0_ADM)
print("")
'''
SOL_TPFA = np.load('SOL_TPFA.npy')
'''
print("resolvendo TPFA")
t0=time.time()
SOL_TPFA=linalg.spsolve(T,b)
print("resolveu TPFA: ",time.time()-t0+t_assembly,t_assembly)
np.save('SOL_TPFA.npy', SOL_TPFA)

erro=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erro[i]=(SOL_TPFA[i]-SOL_ADM_fina[i])/SOL_TPFA[i]

# erro = 100*np.absolute((SOL_TPFA - SOL_ADM_fina)/SOL_TPFA)
erro = 100*np.absolute(erro)

ERRO_tag = mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_TPFA_tag = mb.tag_get_handle("Pressao_TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_ADM_tag = mb.tag_get_handle("Pressao_ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

for v in all_volumes:

    # gid=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    gid=map_global[v]
    mb.tag_set_data(ERRO_tag,v,erro[gid])
    mb.tag_set_data(Sol_TPFA_tag,v,SOL_TPFA[gid])
    mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])

av = mb.create_meshset()
mb.add_entities(av, all_volumes)
os.chdir(output_dir)
mb.write_file('teste_3D_unstructured_18.vtk',[av])
print('New file created')
print(erro.max())
print(erro.min())
import pdb; pdb.set_trace()
