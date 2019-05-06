import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find, lil_matrix
import yaml


parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
sol_direta_dir =  os.path.join(bifasico_dir, 'sol_direta')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils

import pdb; pdb.set_trace()
os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

input_file = data_loaded['input_file']
ext_h5m_adm = input_file + '_malha_adm.h5m'

mb = core.Core()
mtu = topo_util.MeshTopoUtil(mb)
os.chdir(flying_dir)
mb.load_file(ext_h5m_adm)
root_set = mb.get_root_set()
list_names_tags = np.load('list_names_tags.npy')
os.chdir(parent_dir)

all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)

dict_tags = utpy.get_all_tags_2(mb, list_names_tags)
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
oth.gravity = mb.tag_get_data(dict_tags['GRAVITY'], 0, flat=True)


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
wirebasket_elems = list(internos) + list(faces) + list(arestas) + list(vertices)

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

ln=[]
cn=[]
dn=[]

lines=[]
cols=[]
data=[]
vazao=1
for d in volumes_d:
    ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],d))
    press = mb.tag_get_data(dict_tags['P'], d, flat=True)[0]
    lines.append(ID_global)
    cols.append(0)
    data.append(press)

    ln.append(ID_global)
    cn.append(0)
    dn.append(-vazao)

    #b[ID_global]=press
for n in volumes_n:
    vazao2 = mb.tag_get_data(dict_tags['Q'], n, flat=True)[0]
    ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],n))
    lines.append(ID_global)
    cols.append(0)
    data.append(vazao2)

    ln.append(ID_global)
    cn.append(0)
    dn.append(vazao)
    #b[ID_global]=vazao
# del(b)
b=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))

bn=csc_matrix((dn,(ln,cn)),shape=(len(all_volumes),1))

lii=[]
lif=[]
lff=[]
lfe=[]
lee=[]
lev=[]
lvv=[]

cii=[]
cif=[]
cff=[]
cfe=[]
cee=[]
cev=[]
cvv=[]

dii=[]
dif=[]
dff=[]
dfe=[]
dee=[]
dev=[]
dvv=[]

boundary_faces = mb.get_entities_by_handle(mb.tag_get_data(dict_tags['FACES_BOUNDARY'], 0, flat=True)[0])
area_tag = dict_tags['AREA']
perm_tag = dict_tags['PERM']

faces_in = rng.subtract(all_faces, boundary_faces)
all_keqs = mb.tag_get_data(dict_tags['K_EQ'], faces_in, flat=True)
map_all_keqs = dict(zip(faces_in, all_keqs))

ID_reordenado_tag = dict_tags['ID_reord_tag']
print("def As")
ty=time.time()

lines_sgr = []
cols_sgr = []
data_sgr = []
b_s_grav = np.zeros(len(all_volumes))
gravv_tag = mb.tag_get_handle('S_GRAV_VOLUME', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
for i, f in enumerate(faces_in):
    keq = all_keqs[i]
    s_grav, adjs = oth.get_sgrav_adjs_by_face(mb, mtu, f, keq)
    Gid_1=int(mb.tag_get_data(ID_reordenado_tag,adjs[0]))
    Gid_2=int(mb.tag_get_data(ID_reordenado_tag,adjs[1]))
    lines_sgr.append(Gid_1)
    lines_sgr.append(Gid_2)
    cols_sgr.append(0)
    cols_sgr.append(0)
    data_sgr.append(-s_grav)
    data_sgr.append(s_grav)
    b_s_grav[Gid_1] -= s_grav
    b_s_grav[Gid_2] += s_grav



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

    elif Gid_1<ni and Gid_2>=ni and Gid_2<ni+nf:
        lif.append(Gid_1)
        cif.append(Gid_2-ni)
        dif.append(keq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-keq)

    elif Gid_2<ni and Gid_1>=ni and Gid_1<ni+nf:
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

    elif Gid_1>=ni+nf+na and Gid_2>=ni+nf+na:
        lvv.append(Gid_1)
        cvv.append(Gid_2)
        dvv.append(keq)

        lvv.append(Gid_2)
        cvv.append(Gid_1)
        dvv.append(keq)

        lvv.append(Gid_1)
        cvv.append(Gid_1)
        dvv.append(-keq)

        lvv.append(Gid_2)
        cvv.append(Gid_2)
        dvv.append(-keq)

print("took: ",time.time()-ty)
print("get As")
ty=time.time()
mb.tag_set_data(gravv_tag, wirebasket_elems, b_s_grav)

Aii=csc_matrix((dii,(lii,cii)),shape=(ni,ni))
Aif=csc_matrix((dif,(lif,cif)),shape=(ni,nf))
Aff=csc_matrix((dff,(lff,cff)),shape=(nf,nf))
Afe=csc_matrix((dfe,(lfe,cfe)),shape=(nf,na))
Aee=csc_matrix((dee,(lee,cee)),shape=(na,na))
Aev=csc_matrix((dev,(lev,cev)),shape=(na,nv))
Avv=csc_matrix((dvv,(lvv,cvv)),shape=(nv,nv))

Ivv=scipy.sparse.identity(nv)

# volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['P']]), np.array([None]))
# volumes_n = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['Q']]), np.array([None]))
#
# ln=[]
# cn=[]
# dn=[]
#
# lines=[]
# cols=[]
# data=[]
# vazao=1
# for d in volumes_d:
#     ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],d))
#     press = mb.tag_get_data(dict_tags['P'], d, flat=True)[0]
#     lines.append(ID_global)
#     cols.append(0)
#     data.append(press)
#
#     ln.append(ID_global)
#     cn.append(0)
#     dn.append(-vazao)
#
#     #b[ID_global]=press
# for n in volumes_n:
#     vazao2 = mb.tag_get_data(dict_tags['Q'], n, flat=True)[0]
#     ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],n))
#     lines.append(ID_global)
#     cols.append(0)
#     data.append(vazao2)
#
#     ln.append(ID_global)
#     cn.append(0)
#     dn.append(vazao)
#     #b[ID_global]=vazao
# # del(b)
# b=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))
#
# bn=csc_matrix((dn,(ln,cn)),shape=(len(all_volumes),1))
#
if oth.gravity == True:

    ids_volumes_d = mb.tag_get_data(ID_reordenado_tag, volumes_d, flat=True)
    b_s_grav[ids_volumes_d] = np.zeros(len(volumes_d))
    lines = np.arange(len(all_volumes))
    cols = np.repeat(0, len(all_volumes))
    data = b_s_grav
    b_s_grav = csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))
    b += b_s_grav

#
# ln=[]
# cn=[]
# dn=[]
#
# vazao=1
# for d in volumes_d:
#     ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],d))
#     press = mb.tag_get_data(dict_tags['P'], d, flat=True)[0]
#     b[ID_global,0] = press
#
#     ln.append(ID_global)
#     cn.append(0)
#     dn.append(-vazao)
#
#     #b[ID_global]=press
# for n in volumes_n:
#     vazao2 = mb.tag_get_data(dict_tags['Q'], n, flat=True)[0]
#     ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],n))
#     b[ID_global,0] += vazao2
#
#     ln.append(ID_global)
#     cn.append(0)
#     dn.append(vazao)
#     #b[ID_global]=vazao
# # del(b)
# # b=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))
#
# bn=csc_matrix((dn,(ln,cn)),shape=(len(all_volumes),1))

print("took: ",time.time()-ty)

print("get_OP_AMS")
ty=time.time()



#th=time.time()
#M2=-linalg.inv(Aee)*Aev
#print(time.time()-th,"Direto")

invAee=oth.lu_inv(Aee)
M2=-invAee*Aev
P=vstack([M2,Ivv]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

invAff=oth.lu_inv(Aff)
M3=-invAff*Afe*M2
del(M2)
P=vstack([M3,P])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)

invAii=oth.lu_inv(Aii)
P=vstack([-invAii*Aif*M3,P]) ##P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)
del(M3)
print("took to get_OP_AMS",time.time()-ty)

b_ams1_wire=b
b_ams1_int=b_ams1_wire[0:ni,0]
b_ams1_fac=b_ams1_wire[ni:ni+nf,0]
b_ams1_are=b_ams1_wire[ni+nf:ni+nf+na,0]
b_ams1_ver=b_ams1_wire[ni+nf+na:ni+nf+na+nv,0]

corr_1=csc_matrix((nv,1))
corr_1=vstack([invAee*b_ams1_are,corr_1])
corr_1=vstack([((invAff*b_ams1_fac)-(invAff*Afe*invAee*b_ams1_are)),corr_1])
corr_1=vstack([(invAii*b_ams1_int-invAii*Aif*invAff*b_ams1_fac+invAii*(Aif*invAff*Afe*invAee)*b_ams1_are),corr_1])

del(invAii)
del(invAff)
del(invAee)
corr_1=corr_1.toarray()

ID_global=mb.tag_get_data(ID_reordenado_tag, all_volumes)
nivel=mb.tag_get_data(dict_tags['l3_ID'], all_volumes)
ids0_d=mb.tag_get_data(ID_reordenado_tag, volumes_d, flat=True)

lines=[int(ID_global[i]) for i in range(len(ID_global)) if nivel[i]>1 and ID_global[i] in ids0_d]
data=[float(corr_1[i]) for i in range(len(ID_global)) if nivel[i]>1 and ID_global[i] in ids0_d]
cols=np.zeros(len(lines))

corr_adm1_d=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))

lines=[i for i in range(len(ID_global)) if (nivel[ID_global[i]]>1 and i not in ids0_d)]
data=[float(corr_1[i]) for i in range(len(ID_global)) if (nivel[ID_global[i]]>1 and i not in ids0_d)]
cols=np.zeros(len(lines))

corr_adm1_sd=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))
corr_adm1_sd=csc_matrix(corr_1)


ld=[]
cd=[]
dd=[]

for v in volumes_d:
    ID_global= int(mb.tag_get_data(ID_reordenado_tag,v))
    ld.append(ID_global)
    cd.append(0)
    dd.append(float(corr_1[ID_global]))
corr_1=csc_matrix(corr_1)
corr_d1=csc_matrix((dd,(ld,cd)),shape=(len(all_volumes),1))

l=[]
c=[]
d=[]
for v in all_volumes:
    ID_global= int(mb.tag_get_data(ID_reordenado_tag,v))
    l.append(ID_global)
    c.append(0)
    d.append(float(corr_1.toarray()[ID_global]))

corr_sd1=csc_matrix((d,(l,c)),shape=(len(all_volumes),1))

AMS_TO_ADM={}
for v in vertices:
    ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    AMS_TO_ADM[str(ID_AMS)] = ID_ADM

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
nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['l3_ID']]), np.array([1]))
print("get nivel 1___")

matriz=scipy.sparse.find(P)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)
t0 = time.time()
cont=0
for v in nivel_0:
    ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
    ID_global=int(mb.tag_get_data(ID_reordenado_tag,v))
    lines.append(ID_global)
    cols.append(ID_ADM)
    data.append(1)

    dd=np.where(LIN==ID_global)
    LIN=np.delete(LIN,dd,axis=0)
    COL=np.delete(COL,dd,axis=0)
    DAT=np.delete(DAT,dd,axis=0)

print("set_nivel 0")

print("loop", time.time()-ty)

gids_nv1_adm = np.unique(mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True))
n1_adm = len(gids_nv1_adm)
gids_nv2_adm = np.unique(mb.tag_get_data(dict_tags['l2_ID'], all_volumes, flat=True))
n2_adm = len(gids_nv2_adm)


ID_ADM=[AMS_TO_ADM[str(k)] for k in COL]
lines=np.concatenate([lines,LIN])
cols=np.concatenate([cols,ID_ADM])
data=np.concatenate([data,DAT])
print("op_adm", time.time()-ty)
OP_ADM=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),n1_adm))


#for v in volumes_d: print(find(OP_ADM)[2][M1.mb.tag_get_data(M1.ID_reordenado_tag,v)])


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
for f in all_faces:
    adjs = mtu.get_bridge_adjacencies(f, 2, 3)
    if len(adjs)>1:
        keq, s_grav, adjs = oth.get_kequiv_by_face_quad(mb, mtu, f, perm_tag, area_tag)

        Gid_1=int(mb.tag_get_data(ID_reordenado_tag,adjs[0]))
        Gid_2=int(mb.tag_get_data(ID_reordenado_tag,adjs[1]))
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

T_orig=T
t_assembly=time.time()-t_ass

#----------------------------------------------------
T_AMS=OR_AMS*T*OP_AMS
T_ADM=OR_ADM*T*OP_ADM

D2_tag = dict_tags['d2']

v=mb.create_meshset()
mb.add_entities(v,vertices)
tmod12=time.time()
inte=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

for meshset in meshsets_nv2:
    elems = mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, ver)
    nc = mb.tag_get_data(fine_to_primal2_classic_tag, vert, flat=True)[0]
    mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))
    mb.tag_set_data(primal_id_tag2, meshset, nc)


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



lines=[]
cols=[]
data=[]
i=0
for v in vertices:
    ID_AMS_1=i
    #ID_AMS_1=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
    ID_AMS_2=int(mb.tag_get_data(fine_to_primal2_classic_tag,v))
    lines.append(ID_AMS_2)
    cols.append(ID_AMS_1)
    data.append(1)
    i+=1
OR_AMS_2=csc_matrix((data,(lines,cols)),shape=(nver,nv))

W_AMS=G*T_AMS*G.transpose()

MPFA_NO_NIVEL_2=mb.tag_get_data(dict_tags['MPFA'], 0, flat=True)[0]

#-------------------------------------------------------------------------------
nv1=nv

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
        data.append(float(Aie.sum(axis=1)[i])+float(Aiv.sum(axis=1)[i]))
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
    data_fv.append(float(Afv.sum(axis=1)[i]))

Sfi=csc_matrix((data_fi,(lines,cols)),shape=(nf,nf))
Aff += Sfi
if MPFA_NO_NIVEL_2==False:
    Sfv=csc_matrix((data_fv,(lines,cols)),shape=(nf,nf))
    Aff +=Sfv

Aei=W_AMS[ni+nf:ni+nf+na,0:ni]
Aef=W_AMS[ni+nf:ni+nf+na,ni:ni+nf]
Aee=W_AMS[ni+nf:ni+nf+na,ni+nf:ni+nf+na]
Aev=W_AMS[ni+nf:ni+nf+na,ni+nf+na:ni+nf+na+nv]

Avv=W_AMS[ni+nf+na:ni+nf+na+nv,ni+nf+na:ni+nf+na+nv]

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
invAee=oth.lu_inv(Aee)
M2=-csc_matrix(invAee)*Aev
P2=vstack([M2,Ivv])

invAff=oth.lu_inv(Aff)

if MPFA_NO_NIVEL_2:
    M3=-invAff*Afe*M2-invAff*Afv
    P2=vstack([M3,P2])
else:
    Mf=-invAff*Afe*M2
    P2=vstack([Mf,P2])
invAii=oth.lu_inv(Aii)
if MPFA_NO_NIVEL_2:
    M3=invAii*(-Aif*M3+Aie*invAee*Aev-Aiv)
    P2=vstack([M3,P2])
else:
    P2=vstack([-invAii*Aif*Mf,P2])

L1_ID_tag = dict_tags['l1_ID']
L2_ID_tag = dict_tags['l2_ID']
L3_ID_tag = dict_tags['l3_ID']


COL_TO_ADM_2={}
# ver é o meshset dos vértices da malha dual grossa
for i in range(nv):
    v=ver[i]
    ID_AMS=int(mb.tag_get_data(fine_to_primal2_classic_tag,v))
    ID_ADM=int(mb.tag_get_data(L2_ID_tag,v))
    COL_TO_ADM_2[str(i)] = ID_ADM

P2=G.transpose()*P2
#m=find(OP_AMS_2)
#lines=m[0]
#cols=[COL_TO_AMS_2[str(i)] for i in m[1]]
#data=m[2]
#OP_AMS_2=csc_matrix((data,(lines,cols)),shape=(nv1,nver))
#
#P2=OP_AMS_2
#---Vértices é o meshset dos vértices da malha dual do nivel intermediário------
'''
p2=find(P2)
LIN=p2[0]
COL=p2[1]
DAT=p2[2]
dd=np.where(DAT<0.07)
LIN=np.delete(LIN,dd,axis=0)
COL=np.delete(COL,dd,axis=0)
DAT=np.delete(DAT,dd,axis=0)

P2=csc_matrix((DAT,(LIN,COL)),shape=(len(vertices),len(ver)))

for i in range(P2.shape[0]): P2[i]=P2[i]*(1/P2[i].sum())
'''
OP_AMS_2=P2

P2=P2.toarray()
#OP_ADM_2=np.zeros((len(T_ADM),n2))

lines=[]
cols=[]
data=[]
print("took",time.time()-t0)
print("Resolvendo sistema ADM_2")
t0=time.time()
My_IDs_2=[]
for v in all_volumes:
    ID_global=int(mb.tag_get_data(L1_ID_tag,v))
    if ID_global not in My_IDs_2:
        My_IDs_2.append(ID_global)
        ID_ADM=int(mb.tag_get_data(L2_ID_tag,v))
        nivel=mb.tag_get_data(L3_ID_tag,v)
        d1=mb.tag_get_data(D2_tag,v)
        ID_AMS = int(mb.tag_get_data(fine_to_primal1_classic_tag, v))
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

print(time.time()-t0,"organize OP_ADM_2_______________________________::::::::::::")
OP_ADM_2=csc_matrix((data,(lines,cols)),shape=(n1_adm,n2_adm))

#for i in range(P2.shape[0]): print(len(np.where(p2[0]==i)[0]))
#####################################################

lines=[]
cols=[]
data=[]
P2=OP_AMS_2

vm=mb.create_meshset()
mb.add_entities(vm,vertices)
for i in range(len(ver)):
    OP_ams2_tag=mb.tag_get_handle("OP_ams2_tag_"+str(i), 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    vals=OP_AMS_2[:,i].toarray()
    mb.tag_set_data(OP_ams2_tag,vertices,vals)
mb.write_file('delete_me.vtk',[vm])

ty=time.time()
print("iniciou____")
m_vert=mb.create_meshset()
mb.add_entities(m_vert,vertices)
nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_1=mb.get_entities_by_type_and_tag(m_vert, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
print("get níveis 0 e 1___")

P2=csc_matrix(P2)
matriz=scipy.sparse.find(P2)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)
i=1
for v in nivel_1:
    ID_ADM_1=int(mb.tag_get_data(L1_ID_tag,v))
    ID_ADM_2=int(mb.tag_get_data(L2_ID_tag,v))
    ID_AMS = int(mb.tag_get_data(fine_to_primal1_classic_tag, v))
    lines.append(ID_ADM_1)
    cols.append(ID_ADM_2)
    data.append(1)

    dd=np.where(LIN==ID_AMS)
    #print(ID_ADM_1,ID_ADM_2,ID_AMS,len(dd[0]),"-------")
    LIN=np.delete(LIN,dd,axis=0)
    COL=np.delete(COL,dd,axis=0)
    DAT=np.delete(DAT,dd,axis=0)
    i+=1
i=1
for v in nivel_0:
    ID_ADM_1=int(mb.tag_get_data(L1_ID_tag,v))
    ID_ADM_2=int(mb.tag_get_data(L2_ID_tag,v))
    ID_AMS = int(mb.tag_get_data(fine_to_primal1_classic_tag, v))
    lines.append(ID_ADM_1)
    cols.append(ID_ADM_2)
    data.append(1)

    dd=np.where(LIN==ID_AMS)
    LIN=np.delete(LIN,dd,axis=0)
    COL=np.delete(COL,dd,axis=0)
    DAT=np.delete(DAT,dd,axis=0)
    #print(ID_AMS,ID_ADM_1,ID_ADM_2,dd)
print("set_níveis 0 e 1")
print("loop", time.time()-ty)

LIN_ADM=[AMS_TO_ADM[str(k)] for k in LIN]
COL_ADM=[COL_TO_ADM_2[str(k)] for k in COL]
lines=np.concatenate([lines,LIN_ADM])
cols=np.concatenate([cols,COL_ADM])
data=np.concatenate([data,DAT])
#
#del(COL)
#del(LIN)
#del(DAT)
print("op_adm_2", time.time()-ty)
n1 = n1_adm
n2 = n2_adm
OP_ADM_3=csc_matrix((data,(lines,cols)),shape=(n1,n2))



#OP_ADM_2=OP_ADM_3

####################################################
#OR_ADM_2=np.zeros((n2,len(T_ADM)),dtype=np.int)
lines=[]
cols=[]
data=[]
OR_ADM_2=np.zeros((n2,n1),dtype=np.int)
for v in all_volumes:
    elem_ID2 = int(mb.tag_get_data(L2_ID_tag, v, flat=True))
    elem_Global_ID = int(mb.tag_get_data(L1_ID_tag, v, flat=True))
    #lines.append(elem_ID2)
    #cols.append(elem_Global_ID)
    #data.append(1)
    OR_ADM_2[elem_ID2][elem_Global_ID]=1
#OR_ADM_2=csc_matrix((data,(lines,cols)),shape=(n2,n1))
OR_ADM_2=csc_matrix(OR_ADM_2)
T_ADM_2=OR_ADM_2*T_ADM*OP_ADM_2
#for i in range(OR_ADM.shape[0]): print(sum(OR_ADM.toarray()[i]))

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
tmod=time.time()

ID_global=mb.tag_get_data(ID_reordenado_tag,volumes_d, flat=True)
ID_ADM=int(mb.tag_get_data(L1_ID_tag,v))
ID_ADM_2=mb.tag_get_data(L2_ID_tag,volumes_d, flat=True)
T[ID_global]=scipy.sparse.csc_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global]=np.ones(len(ID_global))

########################## apagar para usar pressão-vazão
# ID_globaln=mb.tag_get_data(ID_reordenado_tag,volumes_n, flat=True)
# T[ID_globaln]=scipy.sparse.csc_matrix((len(ID_globaln),T.shape[0]))
# T[ID_globaln,ID_globaln]=np.ones(len(ID_globaln))
########################## fim de apagar
T_ADM_2[ID_ADM_2]=scipy.sparse.csc_matrix((len(ID_ADM_2),T_ADM_2.shape[0]))
T_ADM_2[ID_ADM_2,ID_ADM_2]=np.ones(len(ID_ADM_2))
print(time.time()-tmod,"jjshhhdh______")

b_ams2_wire=G*OR_AMS*bn
b_ams2_int=b_ams2_wire[0:ni,0]
b_ams2_fac=b_ams2_wire[ni:ni+nf,0]
b_ams2_are=b_ams2_wire[ni+nf:ni+nf+na,0]
b_ams2_ver=b_ams2_wire[ni+nf+na:ni+nf+na+nv,0]

corr=csc_matrix((nv,1))
corr=vstack([invAee*b_ams2_are,corr])
corr=vstack([invAff*b_ams2_fac-invAff*Afe*invAee*b_ams2_are,corr])
corr=vstack([invAii*b_ams2_int-invAii*Aif*invAff*b_ams2_fac+invAii*(Aif*invAff*Afe*invAee-Aie*invAee)*b_ams2_are,corr])

c2=csc_matrix((nv,ni+nf+na+nv))
c2=vstack([hstack([hstack([csc_matrix((na,ni+nf)),Aee]),csc_matrix((na,nv))]),c2])
c2=vstack([hstack([hstack([hstack([csc_matrix((nf,ni)),Aff]),-invAff*Afe*invAee]),csc_matrix((nf,nv))]),c2])
c2=vstack([hstack([hstack([hstack([invAii,-invAii*Aif*invAff]),invAii*(Aif*invAff*Afe*invAee-Aie*invAee)]),csc_matrix((ni,nv))]),c2])
c2=csc_matrix(c2)
corr=G.transpose()*corr
ld=[]
cd=[]
dd=[]
for v in volumes_d:
    AMS_ID=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    ld.append(AMS_ID)
    cd.append(0)
    dd.append(float(corr.toarray()[AMS_ID]))
corr_d=csc_matrix((dd,(ld,cd)),shape=(len(vertices),1))

l=[]
c=[]
d=[]
for v in vertices:
    AMS_ID=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    l.append(AMS_ID)
    c.append(0)
    d.append(float(corr.toarray()[AMS_ID]))
corr_sd=csc_matrix((d,(l,c)),shape=(len(vertices),1))

corr=corr.toarray()
'''
d_meshset=M1.mb.create_meshset()

M1.mb.add_entities(d_meshset,volumes_d)

ver_dirichlet_1=M1.mb.get_entities_by_type_and_tag(d_meshset, types.MBHEX, np.array([D1_tag]), np.array([3]))

line_dirichlet_1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,ver_dirichlet_1)

corr[line_dirichlet_1,0]=0'''

ID_ADM_1=mb.tag_get_data(L1_ID_tag,vertices)
nivel=mb.tag_get_data(L3_ID_tag,vertices)
ids1_d=sorted(set(mb.tag_get_data(fine_to_primal1_classic_tag,volumes_d, flat=True)))

lines=[int(ID_ADM_1[i]) for i in range(len(ID_ADM_1)) if nivel[i]==3 and i in ids1_d]
data=[float(corr[i]) for i in range(len(ID_ADM_1)) if nivel[i]==3 and i in ids1_d]
cols=np.zeros(len(lines))
corr_adm2_d=csc_matrix((data,(lines,cols)),shape=(n1,1))

lines=[int(ID_ADM_1[i]) for i in range(len(ID_ADM_1)) if nivel[i]==3 and i not in ids1_d]
data=[float(corr[i]) for i in range(len(ID_ADM_1)) if nivel[i]==3 and i not in ids1_d]
cols=np.zeros(len(lines))
corr_adm2_sd=csc_matrix((data,(lines,cols)),shape=(n1,1))

corr=csc_matrix(corr)

t0=time.time()

###############
SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)
SOL_ADM_fina_1=OP_ADM*SOL_ADM_1
pms_adm_nv1_tag  = mb.tag_get_handle('PMS_ADM_NV1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
mb.tag_set_data(pms_adm_nv1_tag, wirebasket_elems, SOL_ADM_fina_1)
##############

###############
SOL_AMS_1=linalg.spsolve(OR_AMS*T*OP_AMS,OR_AMS*b)
SOL_AMS_fina_1=OP_AMS*SOL_AMS_1
##############

###############
SOL_AMS_2=linalg.spsolve(OR_AMS_2*OR_AMS*T*OP_AMS*OP_AMS_2,OR_AMS_2*(OR_AMS*(b+corr_d1)+corr_d))
SOL_AMS_fina_2=(OP_AMS*(OP_AMS_2*SOL_AMS_2+corr_sd.transpose().toarray()[0])+corr_sd1.transpose().toarray()[0])  #+corr.transpose().toarray()[0]    +corr_1.transpose().toarray()[0]


print("resolveu ADM_2: ",time.time()-t0)
print("Prolongando sol ADM")
t0=time.time()
T1=OR_ADM*T*OP_ADM
b1=OR_ADM*b

SOL_ADM_2=linalg.spsolve(OR_ADM_2*T1*OP_ADM_2,OR_ADM_2*(b1)) #+T1*corr_adm2_sd
SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM_2#+OP_ADM*corr_adm2_sd.transpose().toarray()[0] #+corr_adm1_sd.transpose().toarray()[0]
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
    erro[i]=abs(SOL_TPFA[i]-SOL_ADM_fina[i])

erroAMS1=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erroAMS1[i]=100*abs((SOL_TPFA[i]-SOL_AMS_fina_1[i])/SOL_TPFA[i])

erroADM1=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erroADM1[i]=100*abs((SOL_TPFA[i]-SOL_ADM_fina_1[i])/SOL_TPFA[i])

erroAMS2=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erroAMS2[i]=100*abs((SOL_TPFA[i]-SOL_AMS_fina_2[i])/SOL_TPFA[i])

ERRO_tag=mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROams1_tag=mb.tag_get_handle("erroAMS1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROams2_tag=mb.tag_get_handle("erroAMS2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROadm1_tag=mb.tag_get_handle("erroADM1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_TPFA_tag=mb.tag_get_handle("Pressão TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_ADM_tag=mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

for v in all_volumes:
    gid=int(mb.tag_get_data(ID_reordenado_tag,v))
    mb.tag_set_data(ERRO_tag,v,erro[gid])
    mb.tag_set_data(ERROams1_tag,v,erroAMS1[gid])
    mb.tag_set_data(ERROams2_tag,v,SOL_AMS_fina_2[gid])
    mb.tag_set_data(ERROadm1_tag,v,erroADM1[gid])
    mb.tag_set_data(Sol_TPFA_tag,v,SOL_TPFA[gid])
    mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])

CORR_tag=mb.tag_get_handle("corr_ams2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
CORR1_tag=mb.tag_get_handle("corr_ams1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
CORR_ADM_2_tag=mb.tag_get_handle("corr_adm2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
CORR_ADM_1_tag=mb.tag_get_handle("corr_adm1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
corr=corr.toarray()
corr_1=corr_1.toarray()
corr_adm2_sd=corr_adm2_sd.toarray()
corr_adm1_sd=corr_adm1_sd.toarray()

i=0
for v in all_volumes:
    gid=int(mb.tag_get_data(ID_reordenado_tag,v))
    ID_AMS=int(mb.tag_get_data(fine_to_primal1_classic_tag,v))
    ID_ADM_1=int(mb.tag_get_data(L1_ID_tag,v))
    mb.tag_set_data(CORR1_tag,v,abs(corr_1[gid]))
    mb.tag_set_data(CORR_ADM_2_tag,v,abs(corr_adm2_sd[ID_ADM_1]))
    mb.tag_set_data(CORR_ADM_1_tag,v,abs(corr_adm1_sd[gid]))
    mb.tag_set_data(CORR_tag,v,abs(corr[ID_AMS]))


################################################################################
p_tag = pms_adm_nv1_tag
gids_nv0 = mb.tag_get_data(dict_tags['ID_reord_tag'], all_volumes, flat=True)
map_global = dict(zip(all_volumes, gids_nv0))
TT, bb = oth.fine_transmissibility_structured(mb, mtu, map_global, faces_in=rng.subtract(all_faces, boundary_faces))
# name_tag_faces_boundary_meshsets
coarse_flux_nv2_tag = mb.tag_get_handle('Q_nv2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
oth1 = oth(mb, mtu)
tag_faces_bound_nv2 = mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(2))
all_faces_boundary_nv2 = mb.tag_get_data(tag_faces_bound_nv2, 0, flat=True)[0]
all_faces_boundary_nv2 = mb.get_entities_by_handle(all_faces_boundary_nv2)
for m in meshsets_nv1:
    qtot = 0.0
    elems = mb.get_entities_by_handle(m)
    gids_nv1_adm = np.unique(mb.tag_get_data(dict_tags['l1_ID'], elems, flat=True))
    if len(gids_nv1_adm) > 1:
        continue
    faces = mtu.get_bridge_adjacencies(elems, 3, 2)
    b_faces = rng.intersect(faces, all_faces_boundary_nv2)
    for face in b_faces:
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
        # keq, s_grav, elems2 = oth.get_kequiv_by_face_quad(mb, mtu, face, dict_tags['PERM'], dict_tags['AREA'])
        p = mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (p[1] - p[0])*keq
        if oth.gravity == True:
            flux += s_grav
        if elems2[0] in elems:
            qtot += flux
        else:
            qtot -= flux
    qtot = abs(qtot)

    mb.tag_set_data(coarse_flux_nv2_tag, elems, np.repeat(qtot, len(elems)))
    # mb.tag_set_data(coarse_flux_nv2_tag, elems, np.repeat(res, len(elems)))
#############################################################################

p_tag = Sol_ADM_tag
# name_tag_faces_boundary_meshsets
coarse_flux_nv3_tag = mb.tag_get_handle('Q_nv3', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
# oth1 = oth(M1.mb, M1.mtu)
tag_faces_bound_nv3 = mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(3))
all_faces_boundary_nv3 = mb.tag_get_data(tag_faces_bound_nv3, 0, flat=True)[0]
all_faces_boundary_nv3 = mb.get_entities_by_handle(all_faces_boundary_nv3)

for m in meshsets_nv2:
    qtot = 0.0
    elems = mb.get_entities_by_handle(m)
    gids_nv2_adm = np.unique(mb.tag_get_data(dict_tags['l2_ID'], elems, flat=True))
    if len(gids_nv2_adm) > 1:
        continue
    faces = mtu.get_bridge_adjacencies(elems, 3, 2)
    faces = rng.intersect(faces, all_faces_boundary_nv3)
    for face in faces:
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
        # keq, s_grav, elems2 = oth.get_kequiv_by_face_quad(mb, mtu, face, dict_tags['PERM'], dict_tags['AREA'])
        p = mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (p[1] - p[0])*keq
        if oth.gravity == True:
            flux += s_grav
        if elems2[0] in elems:
            qtot += flux
        else:
            qtot -= flux
    qtot = abs(qtot)

    mb.tag_set_data(coarse_flux_nv3_tag, elems, np.repeat(qtot, len(elems)))

#calculo da pressao corrigida:
meshset_vertices_nv2 = mb.create_meshset()
meshset_vertices_nv3 = mb.create_meshset()

vertices_nv2 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
mb.add_entities(meshset_vertices_nv2, vertices_nv2)
vertices_nv3 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D2_tag]), np.array([3]))
vertices_nv3 = rng.intersect(vertices_nv2, vertices_nv3)
mb.add_entities(meshset_vertices_nv3, vertices_nv3)

elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))

vertices_nv2 = mb.get_entities_by_type_and_tag(meshset_vertices_nv2, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))

vertices_nv3 = mb.get_entities_by_type_and_tag(meshset_vertices_nv3, types.MBHEX, np.array([L3_ID_tag]), np.array([3]))
pcorr_tag = mb.tag_get_handle('PCORR2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

p_tag = Sol_ADM_tag
for vert in vertices_nv2:
    primal_id = mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([fine_to_primal1_classic_tag]), np.array([primal_id]))
    n = len(elems_in_meshset)
    map_volumes = dict(zip(elems_in_meshset, range(n)))
    faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
    faces = rng.subtract(faces, boundary_faces)
    faces_boundary = rng.intersect(faces, all_faces_boundary_nv2)
    b = np.zeros(n)

    for face in faces_boundary:
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
        pmss = mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (pmss[1] - pmss[0])*keq
        if oth.gravity:
            flux += s_grav

        flux *= -1

        if elems2[0] in elems_in_meshset:
            b[map_volumes[elems2[0]]] += flux
        else:
            b[map_volumes[elems2[1]]] -= flux

    lines = []
    cols = []
    data = []
    for face in rng.subtract(faces, faces_boundary):
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)

        s_grav *= -1

        id0 = map_volumes[elems2[0]]
        id1 = map_volumes[elems2[1]]
        lines += [id0, id1]
        cols += [id1, id0]
        data += [keq, keq]
        if oth.gravity:
            b[id0] += s_grav
            b[id1] -= s_grav

    T = csc_matrix((data,(lines,cols)), shape=(n, n))
    T = T.tolil()
    d1 = np.array(T.sum(axis=1)).reshape(1, n)[0]*(-1)
    T.setdiag(d1)

    idv = map_volumes[vert]
    T[idv] = np.zeros(n)
    pms_vert = mb.tag_get_data(p_tag, vert, flat=True)[0]
    T[idv, idv] = 1.0
    b[idv] = pms_vert

    resp = oth.get_solution(T, b)
    mb.tag_set_data(pcorr_tag, elems_in_meshset, resp)

for vert in vertices_nv3:
    primal_id = mb.tag_get_data(fine_to_primal2_classic_tag, vert, flat=True)[0]
    elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([fine_to_primal2_classic_tag]), np.array([primal_id]))
    n = len(elems_in_meshset)
    map_volumes = dict(zip(elems_in_meshset, range(n)))
    faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
    faces = rng.subtract(faces, boundary_faces)
    faces_boundary = rng.intersect(faces, all_faces_boundary_nv3)
    b = np.zeros(n)

    for face in faces_boundary:
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
        pmss = mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (pmss[1] - pmss[0])*keq
        if oth.gravity:
            flux += s_grav

        flux *= -1

        if elems2[0] in elems_in_meshset:
            b[map_volumes[elems2[0]]] += flux
        else:
            b[map_volumes[elems2[1]]] -= flux

    lines = []
    cols = []
    data = []
    for face in rng.subtract(faces, faces_boundary):
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)

        s_grav *= -1

        id0 = map_volumes[elems2[0]]
        id1 = map_volumes[elems2[1]]
        lines += [id0, id1]
        cols += [id1, id0]
        data += [keq, keq]
        if oth.gravity:
            b[id0] += s_grav
            b[id1] -= s_grav

    T = csc_matrix((data,(lines,cols)), shape=(n, n))
    T = T.tolil()
    d1 = np.array(T.sum(axis=1)).reshape(1, n)[0]*(-1)
    T.setdiag(d1)

    idv = map_volumes[vert]
    T[idv] = np.zeros(n)
    pms_vert = mb.tag_get_data(p_tag, vert, flat=True)[0]
    T[idv, idv] = 1.0
    b[idv] = pms_vert

    resp = oth.get_solution(T, b)
    mb.tag_set_data(pcorr_tag, elems_in_meshset, resp)

#### calcular o fluxo multiescala
p_tag = Sol_ADM_tag
fluxo_mult_tag = mb.tag_get_handle('FLUXO_MULTIESCALA', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
###volumes que estao no nivel 3
for vert in vertices_nv3:
    primal_id = mb.tag_get_data(fine_to_primal2_classic_tag, vert, flat=True)[0]
    elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([fine_to_primal2_classic_tag]), np.array([primal_id]))
    n = len(elems_in_meshset)
    map_volumes = dict(zip(elems_in_meshset, range(n)))
    faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
    faces = rng.subtract(faces, boundary_faces)
    faces_boundary = rng.intersect(faces, all_faces_boundary_nv3)
    fluxos = np.zeros(n)

    for face in faces_boundary:
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
        pmss = mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (pmss[1] - pmss[0])*keq
        if oth.gravity:
            flux += s_grav

        flux *= -1

        if elems2[0] in elems_in_meshset:
            fluxos[map_volumes[elems2[0]]] += flux
        else:
            fluxos[map_volumes[elems2[1]]] -= flux

    for face in rng.subtract(faces, faces_boundary):
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)

        # s_grav *= -1

        id0 = map_volumes[elems2[0]]
        id1 = map_volumes[elems2[1]]
        ps_corr = mb.tag_get_data(pcorr_tag, elems2, flat=True)
        flux = (ps_corr[1] - ps_corr[0])*keq

        if oth.gravity:
            flux += s_grav

        flux *= -1

        fluxos[map_volumes[elems2[0]]] += flux
        fluxos[map_volumes[elems2[1]]] -= flux

    mb.tag_set_data(fluxo_mult_tag, elems_in_meshset, fluxos)

###volumes no nivel 2
for vert in vertices_nv2:
    primal_id = mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([fine_to_primal1_classic_tag]), np.array([primal_id]))
    n = len(elems_in_meshset)
    map_volumes = dict(zip(elems_in_meshset, range(n)))
    faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
    faces = rng.subtract(faces, boundary_faces)
    faces_boundary = rng.intersect(faces, all_faces_boundary_nv2)
    fluxos = np.zeros(n)

    for face in faces_boundary:
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
        pmss = mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (pmss[1] - pmss[0])*keq
        if oth.gravity:
            flux += s_grav

        flux *= -1

        if elems2[0] in elems_in_meshset:
            fluxos[map_volumes[elems2[0]]] += flux
        else:
            fluxos[map_volumes[elems2[1]]] -= flux

    for face in rng.subtract(faces, faces_boundary):
        keq = map_all_keqs[face]
        s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)

        # s_grav *= -1

        id0 = map_volumes[elems2[0]]
        id1 = map_volumes[elems2[1]]
        ps_corr = mb.tag_get_data(pcorr_tag, elems2, flat=True)
        flux = (ps_corr[1] - ps_corr[0])*keq

        if oth.gravity:
            flux += s_grav

        flux *= -1

        fluxos[map_volumes[elems2[0]]] += flux
        fluxos[map_volumes[elems2[1]]] -= flux

    mb.tag_set_data(fluxo_mult_tag, elems_in_meshset, fluxos)

###volumes no nivel 0 ou 1

faces = mtu.get_bridge_adjacencies(elems_nv0, 3, 2)
faces = rng.subtract(faces, boundary_faces)
volumes_2 = mtu.get_bridge_adjacencies(elems_nv0, 2, 3) # volumes no nivel_0 uniao com os seus vizinhos
volumes_3 = rng.subtract(volumes_2, elems_nv0)
faces_3 = mtu.get_bridge_adjacencies(volumes_3, 3, 2)
faces_boundary = rng.intersect(faces, faces_3)
n = len(elems_nv0)
map_volumes = dict(zip(elems_nv0, range(n)))
fluxos = np.zeros(n)

for face in faces_boundary:
    keq = map_all_keqs[face]
    s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
    pmss = mb.tag_get_data(p_tag, elems2, flat=True)
    flux = (pmss[1] - pmss[0])*keq
    if oth.gravity:
        flux += s_grav

    flux *= -1
    vvv = True
    try:
        id = map_volumes[elems2[0]]
    except KeyError:
        id = map_volumes[elems2[1]]
        vvv = False

    if vvv:
        fluxos[id] += flux
    else:
        fluxos[id] -= flux

for face in rng.subtract(faces, faces_boundary):
    keq = map_all_keqs[face]
    s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
    pmss = mb.tag_get_data(p_tag, elems2, flat=True)
    flux = (pmss[1] - pmss[0])*keq
    if oth.gravity:
        flux += s_grav

    flux *= -1

    id0 = map_volumes[elems2[0]]
    id1 = map_volumes[elems2[1]]
    fluxos[id0] += flux
    fluxos[id1] -= flux

mb.tag_set_data(fluxo_mult_tag, elems_nv0, fluxos)

fluxo_tpfa_tag = mb.tag_get_handle('FLUXO_TPFA', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
p_tag = Sol_TPFA_tag
n = len(all_volumes)
map_volumes = dict(zip(all_volumes, range(n)))
faces = rng.subtract(all_faces, boundary_faces)
fluxos = np.zeros(n)

for face in faces:
    keq = map_all_keqs[face]
    s_grav, elems2 = oth.get_sgrav_adjs_by_face(mb, mtu, face, keq)
    ptpfa = mb.tag_get_data(p_tag, elems2, flat=True)
    flux = (ptpfa[1] - ptpfa[0])*keq
    if oth.gravity:
        flux += s_grav

    flux *= -1

    id0 = map_volumes[elems2[0]]
    id1 = map_volumes[elems2[1]]
    fluxos[id0] += flux
    fluxos[id1] -= flux

mb.tag_set_data(fluxo_tpfa_tag, all_volumes, fluxos)




av = mb.create_meshset()
mb.add_entities(av, all_volumes)
teste_tag=mb.tag_get_handle("teste", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
teste=OP_ADM*corr_adm2_sd
for v in all_volumes: mb.tag_set_data(teste_tag,v,abs(teste[int(mb.tag_get_data(ID_reordenado_tag,v))]))
os.chdir(sol_direta_dir)
mb.write_file('teste_3D_unstructured_18.h5m')
mb.write_file('teste_3D_unstructured_tpfa.vtk',[av])

print('New file created')

import pdb; pdb.set_trace()
print(min(erro),max(erro))




# (-T*B1*C2*X1*G+B1*C2 + OP_ADM*C2*OR_ADM)*q
# Ai=OR_ADM*OR_ADM_2*lu_inv(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2)*OR_ADM_2*OP_ADM*(scipy.sparse.identity(len(M1.all_volumes))-T*OP_ADM*corr)
