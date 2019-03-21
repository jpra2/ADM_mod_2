import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import yaml
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')
loader = importlib.machinery.SourceFileLoader('mesh_manager', parent_dir + '/mesh_manager.py')
MeshManager = loader.load_module('mesh_manager').MeshManager

os.chdir(input_dir)
with open("inputs.yaml", 'r') as stream:
    data_loaded = yaml.load(stream)

input_file = data_loaded['input_file']
ext_msh = input_file + '.msh'
M1=MeshManager(ext_msh)
os.chdir(parent_dir)

nnn_tag = M1.mb.tag_get_handle('FILE_NAME', len(input_file), types.MB_TYPE_OPAQUE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(nnn_tag, 0, input_file)


all_volumes=M1.all_volumes
all_centroids = M1.all_centroids
gravity = data_loaded['gravity']
if gravity == True or gravity == False:
    pass
else:
    raise ValueError('gravity == True or False')
bifasico = data_loaded['bifasico']
if bifasico == True or bifasico == False:
    pass
else:
    raise ValueError('bifasico == True or False')
mpfa = data_loaded['MPFA']
if mpfa == True or mpfa == False:
    pass
else:
    raise ValueError('mpfa == True or False')

M1.mb.tag_set_data(M1.gravity_tag, 0, gravity)
M1.mb.tag_set_data(M1.bifasico_tag, 0, bifasico)
M1.mb.tag_set_data(M1.mpfa_tag, 0, mpfa)

if bifasico == True:
    M1.create_tags_bif()
    wells_injector = M1.mb.create_meshset()
    M1.mb.tag_set_data(M1.wells_injector_tag, 0, wells_injector)
    M1.mb.tag_set_data(M1.sor_tag, 0, float(data_loaded['dados_bifasico']['Sor']))
    M1.mb.tag_set_data(M1.swc_tag, 0, float(data_loaded['dados_bifasico']['Swc']))
    M1.mb.tag_set_data(M1.mi_w_tag, 0, float(data_loaded['dados_bifasico']['mi_w']))
    M1.mb.tag_set_data(M1.mi_o_tag, 0, float(data_loaded['dados_bifasico']['mi_o']))
    M1.mb.tag_set_data(M1.gama_w_tag, 0, float(data_loaded['dados_bifasico']['gama_w']))
    M1.mb.tag_set_data(M1.gama_o_tag, 0, float(data_loaded['dados_bifasico']['gama_o']))
    M1.mb.tag_set_data(M1.nw_tag, 0, float(data_loaded['dados_bifasico']['nwater']))
    M1.mb.tag_set_data(M1.no_tag, 0, float(data_loaded['dados_bifasico']['noil']))
    M1.mb.tag_set_data(M1.loops_tag, 0, float(data_loaded['dados_bifasico']['loops']))
    M1.mb.tag_set_data(M1.total_time_tag, 0, float(data_loaded['dados_bifasico']['total_time']))


#Determinação das dimensões do reservatório
coords_nodes = M1.mb.get_coords(M1.all_nodes).reshape([len(M1.all_nodes), 3])
xmin_2 = coords_nodes[:,0].min()
xmax_2 = coords_nodes[:,0].max()
ymin_2 = coords_nodes[:,1].min()
ymax_2 = coords_nodes[:,1].max()
zmin_2 = coords_nodes[:,2].min()
zmax_2 = coords_nodes[:,2].max()
del coords_nodes
xmin = xmin_2
xmax = xmax_2
ymin = ymin_2
ymax = ymax_2
zmin = zmin_2
zmax = zmax_2
Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin  # Dimensões do reservatório
M1.mb.tag_set_data(M1.ltot_tag, 0, [Lx, Ly, Lz])
M1.mb.tag_set_data(M1.tz_tag, 0, Lz)


inds_wells = []
for well in data_loaded['Wells_structured']:
    w = data_loaded['Wells_structured'][well]
    if w['type_region'] == 'box':
        box_volumes = np.array([np.array(w['region1']), np.array(w['region2'])])
        inds0 = np.where(all_centroids[:,0] > box_volumes[0,0])[0]
        inds1 = np.where(all_centroids[:,1] > box_volumes[0,1])[0]
        inds2 = np.where(all_centroids[:,2] > box_volumes[0,2])[0]
        c1 = set(inds0) & set(inds1) & set(inds2)
        inds0 = np.where(all_centroids[:,0] < box_volumes[1,0])[0]
        inds1 = np.where(all_centroids[:,1] < box_volumes[1,1])[0]
        inds2 = np.where(all_centroids[:,2] < box_volumes[1,2])[0]
        c2 = set(inds0) & set(inds1) & set(inds2)
        inds_vols = list(c1 & c2)
        inds_wells += inds_vols
        volumes = rng.Range(np.array(M1.all_volumes)[inds_vols])
    else:
        raise NameError("Defina o tipo de regiao em type_region: 'box'")

    value = float(w['value'])

    if w['type_prescription'] == 'dirichlet':
        if gravity == False:
            pressao = np.repeat(value, len(volumes))

        else:
            z_elems_d = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes])
            delta_z = z_elems_d + Lz
            pressao = M1.gama*(delta_z) + press

        M1.mb.tag_set_data(M1.press_value_tag, volumes, pressao)

    elif  w['type_prescription'] == 'neumann':
        value = value/len(volumes)
        if w['type_well'] == 'injector':
            value = -value
        M1.mb.tag_set_data(M1.vazao_value_tag, volumes, np.repeat(value, len(volumes)))

    else:
        raise NameError("type_prescription == 'neumann' or 'dirichlet'")

    if bifasico == True and w['type_well'] == 'injector':
        M1.mb.add_entities(wells_injector, volumes)



if bifasico == True:
    loader = importlib.machinery.SourceFileLoader('bif_utils', utils_dir + '/bif_utils.py')
    bif_utils = loader.load_module('bif_utils').bifasico(M1.mb, M1.mtu, M1.all_volumes)
    bif_utils.set_sat_in(M1.all_volumes)
    bif_utils.set_lamb(M1.all_volumes)
    bif_utils.set_mobi_faces_ini(M1.all_volumes, rng.subtract(M1.all_faces, M1.all_boundary_faces))
    # mb, all_volumes, all_faces_in, wells_inj, k_eq_tag, mobi_in_faces_tag, mobi_w_in_faces_tag
# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
l1=3
l2=9
la = [3,3,3]
lb = [9,9,9]
# Posição aproximada de cada completação
Cent_weels=all_centroids[inds_wells]

# Distância, em relação ao poço, até onde se usa malha fina
r0=float(data_loaded['rs']['r0'])
M1.mb.tag_set_data(M1.r0_tag, 0, r0)

# Distância, em relação ao poço, até onde se usa malha intermediária
r1=float(data_loaded['rs']['r1'])
M1.mb.tag_set_data(M1.r1_tag, 0, r1)
#--------------fim dos parâmetros de entrada------------------------------------
print("")
print("INICIOU PRÉ PROCESSAMENTO")
tempo0_pre=time.time()
def Min_Max(e):
    verts = M1.mb.get_connectivity(e)     #Vértices de um elemento da malha fina
    coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
    xmin, xmax = coords[0][0], coords[0][0]
    ymin, ymax = coords[0][1], coords[0][1]
    zmin, zmax = coords[0][2], coords[0][2]
    for c in coords:
        if c[0]>xmax: xmax=c[0]
        if c[0]<xmin: xmin=c[0]
        if c[1]>ymax: ymax=c[1]
        if c[1]<ymin: ymin=c[1]
        if c[2]>zmax: zmax=c[2]
        if c[2]<zmin: zmin=c[2]
    return([xmin,xmax,ymin,ymax,zmin,zmax])

#--------------Definição das dimensões dos elementos da malha fina--------------
# Esse bloco deve ser alterado para uso de malhas não estruturadas
all_volumes=M1.all_volumes
# print("Volumes:",all_volumes)
verts = M1.mb.get_connectivity(all_volumes[0])     #Vértices de um elemento da malha fina
coords = np.array([M1.mb.get_coords([vert]) for vert in verts])
xmin, xmax = coords[0][0], coords[0][0]
ymin, ymax = coords[0][1], coords[0][1]
zmin, zmax = coords[0][2], coords[0][2]
for c in coords:
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]
dx0, dy0, dz0 = xmax-xmin, ymax-ymin, zmax-zmin # Tamanho de cada elemento na malha fina
#-------------------------------------------------------------------------------
print("definiu dimensões")
# ----- Definição dos volumes que pertencem à malha fina e armazenamento em uma lista----
# finos -> Lista qua armazena os volumes com completação e também aqueles com distância (em relação ao centroide)
#aos volumes com completação menor que "r0"
finos=[]
intermediarios=[]
pocos_meshset=M1.mb.create_meshset()

# Determina se cada um dos elementos está a uma distância inferior a "r0" de alguma completação
# O quadrado serve para pegar os volumes qualquer direção
for e in all_volumes:
    centroid=M1.mtu.get_average_position([e])
    # Cent_wells -> Lista com o centroide de cada completação
    for c in Cent_weels:
        dx=(centroid[0]-c[0])**2
        dy=(centroid[1]-c[1])**2
        dz=(centroid[2]-c[2])**2
        distancia=dx+dy+dz
        if dx<r0**2 and dy<r0**2 and dz<r0**2:
            finos.append(e)
            if dx<dx0/4+.1 and dy<dy0/4+.1 and dz<dz0/4+.1:
                M1.mb.add_entities(pocos_meshset,[e])
        if distancia<r1**2 and dx<r1**2/2:
            intermediarios.append(e)
M1.mb.tag_set_data(M1.finos_tag, 0, pocos_meshset)

intermediarios_meshset = M1.mb.create_meshset()
M1.mb.add_entities(intermediarios_meshset, intermediarios)
M1.mb.tag_set_data(M1.intermediarios_tag, 0, intermediarios_meshset)

print("definiu volumes na malha fina")

pocos=M1.mb.get_entities_by_handle(pocos_meshset)

volumes_d = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBHEX, np.array([M1.press_value_tag]), np.array([None]))
volumes_n = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBHEX, np.array([M1.vazao_value_tag]), np.array([None]))

finos_meshset = M1.mb.create_meshset()
M1.mb.add_entities(finos_meshset, rng.unite(volumes_d, volumes_n))
M1.mb.add_entities(finos_meshset, pocos)
M1.mb.add_entities(finos_meshset, finos)
M1.mb.tag_set_data(M1.finos_tag, 0, finos_meshset)


print("definiu poços")

xmin = xmin_2
xmax = xmax_2
ymin = ymin_2
ymax = ymax_2
zmin = zmin_2
zmax = zmax_2
#-------------------------------------------------------------------------------

# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
# O valor 0.01 é adicionado para corrigir erros de ponto flutuante
for i in range(int(Lx/lb[0])):    lx2.append(xmin+i*lb[0])
for i in range(int(Ly/lb[1])):    ly2.append(ymin+i*lb[1])
for i in range(int(Lz/lb[2])):    lz2.append(zmin+i*lb[2])

lx2.append(Lx)
ly2.append(Ly)
lz2.append(Lz)

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
lxd2=[lx2[0]+la[0]/2]
if len(lx2)>2:
    for i in np.linspace((lx2[1]+lx2[2])/2,(lx2[-2]+lx2[-3])/2,len(lx2)-3):
        lxd2.append(i)
lxd2.append(lx2[-1]-la[0]/2)

lyd2=[ly2[0]+la[1]/2]
if len(ly2)>2:
    for i in np.linspace((ly2[1]+ly2[2])/2,(ly2[-2]+ly2[-3])/2,len(ly2)-3):
        lyd2.append(i)
lyd2.append(ly2[-1]-la[1]/2)

lzd2=[lz2[0]+la[2]/2]
if len(lz2)>2:
    for i in np.linspace((lz2[1]+lz2[2])/2,(lz2[-2]+lz2[-3])/2,len(lz2)-3):
        lzd2.append(i)
lzd2.append(lz2[-1]-la[2]/2)

print("definiu planos do nível 2")

# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(lb[0]/la[0])):   lx1.append(i*la[0])
for i in range(int(lb[1]/la[1])):   ly1.append(i*la[1])
for i in range(int(lb[2]/la[2])):   lz1.append(i*la[2])

lxd1=[xmin+dx0/100]
for i in np.linspace(xmin+1.5*la[0],xmax-1.5*la[0],int((Lx-3*la[0])/la[0]+1.1)):
    lxd1.append(i)
lxd1.append(xmin+Lx-dx0/100)

lyd1=[ymin+dy0/100]
for i in np.linspace(ymin+1.5*la[1],ymax-1.5*la[1],int((Ly-3*la[1])/la[1]+1.1)):
    lyd1.append(i)
lyd1.append(ymin+Ly-dy0/100)

lzd1=[zmin+dz0/100]
for i in np.linspace(zmin+1.5*la[2],zmax-1.5*la[2],int((Lz-3*la[2])/la[2]+1.1)):
    lzd1.append(i)
lzd1.append(xmin+Lz-dz0/100)

print("definiu planos do nível 1")
node=M1.all_nodes[0]
coords=M1.mb.get_coords([node])
min_dist_x=coords[0]
min_dist_y=coords[1]
min_dist_z=coords[2]
#-------------------------------------------------------------------------------
'''
#----Correção do posicionamento dos planos que definem a dual
# Evita que algum nó pertença a esses planos e deixe as faces da dual descontínuas
for i in range(len(lxd1)):
    for j in range(len(lyd1)):
        for k in range(len(lzd1)):
            for n in range(len(M1.all_nodes)):
                coord=M1.mb.get_coords([M1.all_nodes[n]])
                dx=lxd1[i]-coord[0]
                dy=lyd1[j]-coord[1]
                dz=lzd1[k]-coord[2]
                if np.abs(dx)<0.0000001:
                    print('Plano x = ',lxd1[i],'corrigido com delta x = ',dx)
                    lxd1[i]-=0.000001
                    i-=1
                if np.abs(dy)<0.0000001:
                    print('Plano y = ',lyd1[j],'corrigido com delta y = ',dy)
                    lyd1[j]-=0.000001
                    j-=1
                if np.abs(dz)<0.0000001:
                    print('Plano z = ',lzd1[k],'corrigido dom delta z = ', dz)
                    lzd1[k]-=0.000001
                    k-=1
#-------------------------------------------------------------------------------
print("corrigiu planos do nível 1")'''
t0=time.time()
# ---- Criação e preenchimento da árvore de meshsets----------------------------
# Esse bloco é executado apenas uma vez em um problema bifásico, sua eficiência
# não é criticamente importante.
L2_meshset=M1.mb.create_meshset()       # root Meshset
D2_meshset=M1.mb.create_meshset()

###########################################################################################
#jp:modifiquei as tags abaixo para sparse
D1_tag=M1.mb.tag_get_handle("d1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
D2_tag=M1.mb.tag_get_handle("d2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################

fine_to_primal1_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL1_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
fine_to_primal2_classic_tag = M1.mb.tag_get_handle("FINE_TO_PRIMAL2_CLASSIC", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
AV_meshset=M1.mb.create_meshset()

primal_id_tag1 = M1.mb.tag_get_handle("PRIMAL_ID_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
primal_id_tag2 = M1.mb.tag_get_handle("PRIMAL_ID_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
nc1=0
nc2 = 0
lc1 = {}
#lc1 é um mapeamento de ids no nivel 1 para o meshset correspondente
#{id:meshset}

dict_m = {}

# M1.boundary_faces_nv1_tag = M1.mb.tag_get_handle("BOUNDARY_FACES_NV_1", 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
# M1.boundary_faces_nv2_tag = M1.mb.tag_get_handle("BOUNDARY_FACES_NV_2", 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)

M1.mb.add_entities(AV_meshset, all_volumes)

t1 = time.time()
for i in range(len(lx2)-1):
    t1=time.time()
    for j in range(len(ly2)-1):
        for k in range(len(lz2)-1):
            volumes_nv2 = []
            l2_meshset=M1.mb.create_meshset()
            d2_meshset=M1.mb.create_meshset()
            all_volumes=M1.mb.get_entities_by_handle(AV_meshset)
            for elem in all_volumes:
                centroid=M1.mtu.get_average_position([elem])
                if (centroid[0]>lx2[i]) and (centroid[0]<ly2[i]+lb[1]) and (centroid[1]>ly2[j])\
                and (centroid[1]<ly2[j]+lb[1]) and (centroid[2]>lz2[k]) and (centroid[2]<lz2[k]+lb[2]):
                    M1.mb.add_entities(l2_meshset,[elem])
                    M1.mb.remove_entities(AV_meshset,[elem])
                    elem_por_L2=M1.mb.get_entities_by_handle(l2_meshset)

                if i<(len(lxd2)-1) and j<(len(lyd2)-1) and k<(len(lzd2)-1):
                    if (centroid[0]>lxd2[i]-la[0]/2) and (centroid[0]<lxd2[i+1]+la[0]/2) and (centroid[1]>lyd2[j]-la[1]/2)\
                    and (centroid[1]<lyd2[j+1]+la[1]/2) and (centroid[2]>lzd2[k]-la[2]/2) and (centroid[2]<lzd2[k+1]+la[2]/2):

                        M1.mb.add_entities(d2_meshset,[elem])
                        f1a2v3=0
                        if (centroid[0]-lxd2[i])**2<la[0]**2/4 or (centroid[0]-lxd2[i+1])**2<la[0]**2/4 :
                            f1a2v3+=1
                        if (centroid[1]-lyd2[j])**2<la[1]**2/4 or (centroid[1]-lyd2[j+1])**2<la[1]**2/4:
                            f1a2v3+=1
                        if (centroid[2]-lzd2[k])**2<la[2]**2/4 or (centroid[2]-lzd2[k+1])**2<la[2]**2/4:
                            f1a2v3+=1
                        M1.mb.tag_set_data(D2_tag, elem, f1a2v3)
                        M1.mb.tag_set_data(fine_to_primal2_classic_tag, elem, nc2)
            M1.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg = M1.mb.get_entities_by_handle(l2_meshset)
            # boundary_faces = utpy.get_boundary_of_volumes(M1.mb, sg)
            # bound_meshset = M1.mb.create_meshset()
            # M1.mb.add_entities(bound_meshset, boundary_faces)
            # M1.mb.tag_set_data(M1.boundary_faces_nv2_tag, l2_meshset, bound_meshset)

            print(k, len(sg), time.time()-t1)
            t1=time.time()
            d1_meshset=M1.mb.create_meshset()

            M1.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)
            nc2+=1

            for m in range(len(lx1)):
                a=la[0]*i+m
                for n in range(len(ly1)):
                    b=la[1]*j+n
                    for o in range(len(lz1)):
                        c=la[2]*k+o
                        l1_meshset=M1.mb.create_meshset()
                        for e in elem_por_L2:
                            # Refactory here
                            # Verificar se o uso de um vértice reduz o custo
                            centroid=M1.mtu.get_average_position([e])
                            if (centroid[0]>lx2[i]+lx1[m]) and (centroid[0]<lx2[i]+lx1[m]+la[0])\
                            and (centroid[1]>ly2[j]+ly1[n]) and (centroid[1]<ly2[j]+ly1[n]+la[1])\
                            and (centroid[2]>lz2[k]+lz1[o]) and (centroid[2]<lz2[k]+lz1[o]+la[2]):
                                M1.mb.add_entities(l1_meshset,[e])
                            if a<(len(lxd1)-1) and b<(len(lyd1)-1) and c<(len(lzd1)-1):
                                if (centroid[0]>lxd1[a]-1.5*dx0) and (centroid[0]<lxd1[a+1]+1.5*dx0)\
                                and (centroid[1]>lyd1[b]-1.5*dy0) and (centroid[1]<lyd1[b+1]+1.5*dy0)\
                                and (centroid[2]>lzd1[c]-1.5*dz0) and (centroid[2]<lzd1[c+1]+1.5*dz0):
                                    M1.mb.add_entities(d1_meshset,[elem])
                                    f1a2v3=0
                                    M_M=Min_Max(e)
                                    if (M_M[0]<lxd1[a] and M_M[1]>lxd1[a]) or (M_M[0]<lxd1[a+1] and M_M[1]>lxd1[a+1]):
                                        f1a2v3+=1
                                    if (M_M[2]<lyd1[b] and M_M[3]>lyd1[b]) or (M_M[2]<lyd1[b+1] and M_M[3]>lyd1[b+1]):
                                        f1a2v3+=1
                                    if (M_M[4]<lzd1[c] and M_M[5]>lzd1[c]) or (M_M[4]<lzd1[c+1] and M_M[5]>lzd1[c+1]):
                                        f1a2v3+=1
                                    M1.mb.tag_set_data(D1_tag, e,f1a2v3)
                                    M1.mb.tag_set_data(fine_to_primal1_classic_tag, e, nc1)


                        M1.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                        nc1+=1
                        M1.mb.add_child_meshset(l2_meshset,l1_meshset)
                        # sg = M1.mb.get_entities_by_handle(l1_meshset)
                        # boundary_faces = utpy.get_boundary_of_volumes(M1.mb, sg)
                        # bound_meshset = M1.mb.create_meshset()
                        # M1.mb.add_entities(bound_meshset, boundary_faces)
                        # M1.mb.tag_set_data(M1.boundary_faces_nv1_tag, l1_meshset, bound_meshset)

# meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
# meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))

l2_meshset_tag = M1.mb.tag_get_handle("L2_MESHSET", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
M1.mb.tag_set_data(l2_meshset_tag, 0, L2_meshset)
#-------------------------------------------------------------------------------
print('Criação da árvore: ',time.time()-t0)
ta=time.time()
all_volumes=M1.all_volumes

M1.mb.write_file('testando1.vtk')
import pdb; pdb.set_trace()

# vert_meshset=M1.mb.create_meshset()
#
# for e in all_volumes:
#     d1_tag = int(M1.mb.tag_get_data(D1_tag, e, flat=True))
#     if d1_tag==3:
#         M1.mb.add_entities(vert_meshset,[e])
# all_vertex_d1=M1.mb.get_entities_by_handle(vert_meshset)

all_vertex_d1 = M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
mm=0

for x in lxd1:
    for y in lyd1:
        for z in lzd1:
            v1 = all_vertex_d1[0]
            c=M1.mtu.get_average_position([v1])
            d=(c[0]-x)**2+(c[1]-y)**2+(c[2]-z)**2
            for e in all_vertex_d1:
                c=M1.mtu.get_average_position([e])
                dist=(c[0]-x)**2+(c[1]-y)**2+(c[2]-z)**2
                if dist<d:
                    d=dist
                    v1=e
            M1.mb.tag_set_data(D1_tag, v1, 4)

for e in all_vertex_d1:
    d1_tag = int(M1.mb.tag_get_data(D1_tag, e, flat=True))
    if d1_tag==3:
        M1.mb.tag_set_data(D1_tag, e, 2)
    elif d1_tag==4:
        M1.mb.tag_set_data(D1_tag, e, 3)

meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))
#
# vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices = all_vertex_d1
M1.mb.tag_set_data(fine_to_primal1_classic_tag,vertices,np.arange(0,len(vertices)))

for meshset in meshsets_nv1:
    elems = M1.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, vertices)
    nc = M1.mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    M1.mb.tag_set_data(fine_to_primal1_classic_tag, elems, np.repeat(nc, len(elems)))
    M1.mb.tag_set_data(primal_id_tag1, meshset, nc)


internos=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))

# print(time.time()-tmod1,"mod1 ______")

ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)

ID_reordenado_tag = M1.ID_reordenado_tag

nni=ni
nnf=nni+nf
nne=nnf+na
nnv=nne+nv
l_elems=[internos,faces,arestas,vertices]
l_ids=[0,nni,nnf,nne,nnv]
for i, elems in enumerate(l_elems):
    M1.mb.tag_set_data(ID_reordenado_tag, elems, np.arange(l_ids[i],l_ids[i+1]))



for meshset in meshsets_nv2:
    nc = M1.mb.tag_get_data(primal_id_tag2, meshset, flat=True)[0]
    elems = M1.mb.get_entities_by_handle(meshset)
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))

names_tags_criadas_aqui = ['d1', 'd2', 'FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC',
                           'PRIMAL_ID_1', 'PRIMAL_ID_2', 'L2_MESHSET', 'FILE_NAME']

n_levels = 2
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
all_meshsets = [meshsets_nv1, meshsets_nv2]
t0 = time.time()
for i in range(n_levels):
    meshsets = all_meshsets[i]
    names_tags_criadas_aqui.append(name_tag_faces_boundary_meshsets + str(i+2))
    tag_boundary = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets + str(i+2), 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
    utpy.set_faces_in_boundary_by_meshsets(M1.mb, M1.mtu, meshsets, tag_boundary)
t1 = time.time()
print('tempo faces contorno')
print(t1-t0)



#
#
# M1.boundary_faces_nv1_tag = M1.mb.tag_get_handle("BOUNDARY_FACES_NV_1", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# M1.boundary_faces_nv2_tag = M1.mb.tag_get_handle("BOUNDARY_FACES_NV_2", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
#
# t1 = time.time()
# for m in meshsets_nv1:
#     nc = M1.mb.tag_get_data(primal_id_tag1, m, flat=True)[0]
#     elems = M1.mb.get_entities_by_handle(m)
#     boundary_faces = utpy.get_boundary_of_volumes(M1.mb, elems)
#     boundary_faces_nv1_meshset = M1.mb.create_meshset()
#     M1.mb.add_entities(boundary_faces_nv1_meshset, boundary_faces)
#     M1.mb.tag_set_data(M1.boundary_faces_nv1_tag, boundary_faces_nv1_meshset, nc)
#
# for m in meshsets_nv2:
#     nc = M1.mb.tag_get_data(primal_id_tag2, m, flat=True)[0]
#     elems = M1.mb.get_entities_by_handle(m)
#     boundary_faces = utpy.get_boundary_of_volumes(M1.mb, elems)
#     boundary_faces_nv2_meshset = M1.mb.create_meshset()
#     M1.mb.add_entities(boundary_faces_nv2_meshset, boundary_faces)
#     M1.mb.tag_set_data(M1.boundary_faces_nv2_tag, boundary_faces_nv2_meshset, nc)
#
# t2 = time.time()
# print('tempo faces')
# print(t2-t1)



print(time.time()-ta,"correção")
print("TEMPO TOTAL DE PRÉ PROCESSAMENTO:",time.time()-tempo0_pre)
print(" ")
os.chdir(flying_dir)
ext_h5m = input_file + '.h5m'
ext_vtk = input_file + '.vtk'
M1.mb.write_file(ext_h5m)
av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)
M1.mb.write_file(ext_vtk,[av])
