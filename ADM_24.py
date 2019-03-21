import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
import cython
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find

parent_dir = os.path.dirname(os.path.abspath(__file__))
# parent_parent_dir = os.path.dirname(parent_dir)
parent_parent_dir = parent_dir
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils


class MeshManager:
    def __init__(self,mesh_file, dim=3):
        self.dimension = dim
        self.mb = core.Core()
        self.root_set = self.mb.get_root_set()
        self.mtu = topo_util.MeshTopoUtil(self.mb)

        self.mb.load_file(mesh_file)

        self.physical_tag = self.mb.tag_get_handle("MATERIAL_SET")
        self.physical_sets = self.mb.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, np.array(
            (self.physical_tag,)), np.array((None,)))

        self.dirichlet_tag = self.mb.tag_get_handle(
            "Dirichlet", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.neumann_tag = self.mb.tag_get_handle(
            "Neumann", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        #self.perm_tag = self.mb.tag_get_handle(
        #    "Permeability", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

        self.source_tag = self.mb.tag_get_handle(
            "Source term", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)


        self.all_volumes = self.mb.get_entities_by_dimension(0, self.dimension)

        self.all_nodes = self.mb.get_entities_by_dimension(0, 0)

        self.mtu.construct_aentities(self.all_nodes)
        self.all_faces = self.mb.get_entities_by_dimension(0, self.dimension-1)

        self.dirichlet_faces = set()
        self.neumann_faces = set()

        '''self.GLOBAL_ID_tag = self.mb.tag_get_handle(
            "Global_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)'''

        self.create_tags()
        self.set_k()
        #self.set_information("PERM", self.all_volumes, 3)
        self.get_boundary_faces()
        self.gravity = False
        self.gama = 10

    def create_tags(self):
        self.perm_tag = self.mb.tag_get_handle("PERM", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.finos_tag = self.mb.tag_get_handle("finos", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_dirichlet_tag = self.mb.tag_get_handle("WELLS_D", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_neumann_tag = self.mb.tag_get_handle("WELLS_N", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.press_value_tag = self.mb.tag_get_handle("P", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.vazao_value_tag = self.mb.tag_get_handle("Q", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.all_faces_boundary_tag = self.mb.tag_get_handle("FACES_BOUNDARY", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.area_tag = self.mb.tag_get_handle("AREA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.GLOBAL_ID_tag = self.mb.tag_get_handle("G_ID_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.ID_reordenado_tag = self.mb.tag_get_handle("ID_reord_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.keq_tag = self.mb.tag_get_handle("K_EQ", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.s_grav_tag = self.mb.tag_get_handle("S_GRAV", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    def create_vertices(self, coords):
        new_vertices = self.mb.create_vertices(coords)
        self.all_nodes.append(new_vertices)
        return new_vertices

    def create_element(self, poly_type, vertices):
        new_volume = self.mb.create_element(poly_type, vertices)
        self.all_volumes.append(new_volume)
        return new_volume

    def set_information(self, information_name, physicals_values,
                        dim_target, set_connect=False):
        information_tag = self.mb.tag_get_handle(information_name)
        for physical_value in physicals_values:
            for a_set in self.physical_sets:
                physical_group = self.mb.tag_get_data(self.physical_tag,
                                                      a_set, flat=True)

                if physical_group == physical:
                    group_elements = self.mb.get_entities_by_dimension(a_set, dim_target)

                    if information_name == 'Dirichlet':
                        # print('DIR GROUP', len(group_elements), group_elements)
                        self.dirichlet_faces = self.dirichlet_faces | set(
                                                    group_elements)

                    if information_name == 'Neumann':
                        # print('NEU GROUP', len(group_elements), group_elements)
                        self.neumann_faces = self.neumann_faces | set(
                                                  group_elements)

                    for element in group_elements:
                        self.mb.tag_set_data(information_tag, element, value)

                        if set_connect:
                            connectivities = self.mtu.get_bridge_adjacencies(
                                                                element, 0, 0)
                            self.mb.tag_set_data(
                                information_tag, connectivities,
                                np.repeat(value, len(connectivities)))

    def set_k(self):
        k = 1.0
        perm_tensor = [k, 0, 0,
                       0, k, 0,
                       0, 0, k]
        for v in self.all_volumes:
            self.mb.tag_set_data(self.perm_tag, v, perm_tensor)
            #v_tags=self.mb.tag_get_tags_on_entity(v)
            #print(self.mb.tag_get_data(v_tags[1],v,flat=True))

    def set_area(self, face):


        points = self.mtu.get_bridge_adjacencies(face, 2, 0)
        points = [self.mb.get_coords([vert]) for vert in points]
        if len(points) == 3:
            n1 = np.array(points[0] - points[1])
            n2 = np.array(points[0] - points[2])
            area = (np.linalg.norm(np.cross(n1, n2)))/2.0

        #calculo da area para quadrilatero regular
        elif len(points) == 4:
            n = np.array([np.array(points[0] - points[1]), np.array(points[0] - points[2]), np.array(points[0] - points[3])])
            norms = np.array(list(map(np.linalg.norm, n)))
            ind_norm_max = np.where(norms == max(norms))[0]
            n = np.delete(n, ind_norm_max, axis = 0)

            area = np.linalg.norm(np.cross(n[0], n[1]))

        self.mb.tag_set_data(self.area_tag, face, area)

    def get_boundary_nodes(self):
        all_faces = self.dirichlet_faces | self.neumann_faces
        boundary_nodes = set()
        for face in all_faces:
            nodes = self.mtu.get_bridge_adjacencies(face, 2, 0)
            boundary_nodes.update(nodes)
        return boundary_nodes

    def get_faces_boundary(self):
        """
        cria os meshsets
        all_faces_set: todas as faces do dominio
        all_faces_boundary_set: todas as faces no contorno
        """
        all_faces_boundary_set = self.mb.create_meshset()

        for face in self.all_faces:
            size = len(self.mb.get_adjacencies(face, 3))
            self.set_area(face)
            if size < 2:
                self.mb.add_entities(all_faces_boundary_set, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_faces_boundary_set)

    def get_non_boundary_volumes(self, dirichlet_nodes, neumann_nodes):
        volumes = self.all_volumes
        non_boundary_volumes = []
        for volume in volumes:
            volume_nodes = set(self.mtu.get_bridge_adjacencies(volume, 0, 0))
            if (volume_nodes.intersection(dirichlet_nodes | neumann_nodes)) == set():
                non_boundary_volumes.append(volume)
        return non_boundary_volumes

    def set_media_property(self, property_name, physicals_values,
                           dim_target=3, set_nodes=False):

        self.set_information(property_name, physicals_values,
                             dim_target, set_connect=set_nodes)

    def set_boundary_condition(self, boundary_condition, physicals_values,
                               dim_target=3, set_nodes=False):

        self.set_information(boundary_condition, physicals_values,
                             dim_target, set_connect=set_nodes)

    def get_tetra_volume(self, tet_nodes):
        vect_1 = tet_nodes[1] - tet_nodes[0]
        vect_2 = tet_nodes[2] - tet_nodes[0]
        vect_3 = tet_nodes[3] - tet_nodes[0]
        vol_eval = abs(np.dot(np.cross(vect_1, vect_2), vect_3))/6.0
        return vol_eval

    def get_boundary_faces(self):
        all_boundary_faces = self.mb.create_meshset()
        for face in self.all_faces:
            self.set_area(face)
            elems = self.mtu.get_bridge_adjacencies(face, 2, 3)
            if len(elems) < 2:
                self.mb.add_entities(all_boundary_faces, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_boundary_faces)
        self.all_boundary_faces = self.mb.get_entities_by_handle(all_boundary_faces)

    @staticmethod
    def imprima(self, text = None):
        m1 = self.mb.create_meshset()
        self.mb.add_entities(m1, self.all_nodes)
        m2 = self.mb.create_meshset()
        self.mb.add_entities(m2, self.all_faces)
        m3 = self.mb.create_meshset()
        self.mb.add_entities(m3, self.all_volumes)
        if text == None:
            text = "output"
        extension = ".vtk"
        text1 = text + "-nodes" + extension
        text2 = text + "-face" + extension
        text3 = text + "-volume" + extension
        self.mb.write_file(text1,[m1])
        self.mb.write_file(text2,[m2])
        self.mb.write_file(text3,[m3])
        print(text, "Arquivos gerados")
#--------------Início dos parâmetros de entrada-------------------
M1= MeshManager('27x27x27.msh')          # Objeto que armazenará as informações da malha
all_volumes=M1.all_volumes

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)

M1.all_centroids=np.array([M1.mtu.get_average_position([v]) for v in all_volumes])
all_centroids = M1.all_centroids

y0 = 27.0
y1 = 1.0
y2 = 26.0

#box_volumes_d = np.array([np.array([0.0, 0.0, 0.0]), np.array([y1, y1, y1])])
#box_volumes_n = np.array([np.array([y2, y2, y2]), np.array([y0, y0, y0])])

box_volumes_d = np.array([np.array([y2, 0.0, 0.0]), np.array([y0, y0, y0])])
box_volumes_n = np.array([np.array([0.0, 0.0, 0.0]), np.array([y1, y0, y0])])

box_volumes_f = np.array([np.array([0.0, 0.0, 0.0]), np.array([y1, y1, y1])])

# volumes com pressao prescrita
inds0 = np.where(all_centroids[:,0] > box_volumes_d[0,0])[0]
inds1 = np.where(all_centroids[:,1] > box_volumes_d[0,1])[0]
inds2 = np.where(all_centroids[:,2] > box_volumes_d[0,2])[0]
c1 = set(inds0) & set(inds1) & set(inds2)
inds0 = np.where(all_centroids[:,0] < box_volumes_d[1,0])[0]
inds1 = np.where(all_centroids[:,1] < box_volumes_d[1,1])[0]
inds2 = np.where(all_centroids[:,2] < box_volumes_d[1,2])[0]
c2 = set(inds0) & set(inds1) & set(inds2)
inds_vols_d = list(c1 & c2)
volumes_d = rng.Range(np.array(M1.all_volumes)[inds_vols_d])

# volumes finos
inds0 = np.where(all_centroids[:,0] > box_volumes_f[0,0])[0]
inds1 = np.where(all_centroids[:,1] > box_volumes_f[0,1])[0]
inds2 = np.where(all_centroids[:,2] > box_volumes_f[0,2])[0]
c1 = set(inds0) & set(inds1) & set(inds2)
inds0 = np.where(all_centroids[:,0] < box_volumes_f[1,0])[0]
inds1 = np.where(all_centroids[:,1] < box_volumes_f[1,1])[0]
inds2 = np.where(all_centroids[:,2] < box_volumes_f[1,2])[0]
c2 = set(inds0) & set(inds1) & set(inds2)
inds_vols_f = list(c1 & c2)
volumes_f = rng.Range(np.array(M1.all_volumes)[inds_vols_f])


# volumes com vazao prescrita
inds0 = np.where(all_centroids[:,0] > box_volumes_n[0,0])[0]
inds1 = np.where(all_centroids[:,1] > box_volumes_n[0,1])[0]
inds2 = np.where(all_centroids[:,2] > box_volumes_n[0,2])[0]
c1 = set(inds0) & set(inds1) & set(inds2)
inds0 = np.where(all_centroids[:,0] < box_volumes_n[1,0])[0]
inds1 = np.where(all_centroids[:,1] < box_volumes_n[1,1])[0]
inds2 = np.where(all_centroids[:,2] < box_volumes_n[1,2])[0]
c2 = set(inds0) & set(inds1) & set(inds2)
inds_vols_n = list(c1 & c2)
volumes_n = rng.Range(np.array(M1.all_volumes)[inds_vols_n])

inds_pocos = inds_vols_d + inds_vols_n
Cent_wels = all_centroids[inds_pocos]

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
l1=3
l2=9
# Posição aproximada de cada completação



# Distância, em relação ao poço, até onde se usa malha fina
r0=1

# Distância, em relação ao poço, até onde se usa malha intermediária
r1=1
#--------------fim dos parâmetros de entrada------------------------------------
print("")
print("INICIOU PRÉ PROCESSAMENTO")
tempo0_pre=time.time()
def Min_Max(e):
    verts = M1.mb.get_connectivity(e)
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

#for e in all_volumes:
#    centroid=M1.mtu.get_average_position([e])
#    # Cent_wells -> Lista com o centroide de cada completação
#    for c in Cent_wels:
#        dx=(centroid[0]-c[0])**2
#        dy=(centroid[1]-c[1])**2
#        dz=(centroid[2]-c[2])**2
#        distancia=dx+dy+dz
#        if dx<r0**2 and dy<r0**2 and dz<r0**2:
#            finos.append(e)
#            if dx<dx0/4+.1 and dy<dy0/4+.1 and dz<dz0/4+.1:
#                M1.mb.add_entities(pocos_meshset,[e])
#        if distancia<r1**2 and dx<r1**2/2:
#            intermediarios.append(e)
M1.mb.tag_set_data(M1.finos_tag, 0,pocos_meshset)
finos=list(rng.unite(rng.unite(volumes_d,volumes_n),volumes_f))

print("definiu volumes na malha fina")

pocos=M1.mb.get_entities_by_handle(pocos_meshset)

finos_meshset = M1.mb.create_meshset()

print("definiu poços")
#-------------------------------------------------------------------------------
#Determinação das dimensões do reservatório e dos volumes das malhas intermediária e grossa
for v in M1.all_nodes:       # M1.all_nodes -> todos os vértices da malha fina
    c=M1.mb.get_coords([v])  # Coordenadas de um nó
    if c[0]>xmax: xmax=c[0]
    if c[0]<xmin: xmin=c[0]
    if c[1]>ymax: ymax=c[1]
    if c[1]<ymin: ymin=c[1]
    if c[2]>zmax: zmax=c[2]
    if c[2]<zmin: zmin=c[2]

Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin  # Dimensões do reservatório
#-------------------------------------------------------------------------------

# Criação do vetor que define a "grade" que separa os volumes da malha grossa
# Essa grade é absoluta (relativa ao reservatório como um todo)
lx2, ly2, lz2 = [], [], []
# O valor 0.01 é adicionado para corrigir erros de ponto flutuante
for i in range(int(Lx/l2+1.01)):    lx2.append(xmin+i*l2)
for i in range(int(Ly/l2+1.01)):    ly2.append(ymin+i*l2)
for i in range(int(Lz/l2+1.01)):    lz2.append(zmin+i*l2)

#-------------------------------------------------------------------------------
press = 100.0
vazao = -7290.0
dirichlet_meshset = M1.mb.create_meshset()
neumann_meshset = M1.mb.create_meshset()

if M1.gravity == False:
    pressao = np.repeat(press, len(volumes_d))

# # colocar gravidade
elif M1.gravity == True:
    pressao = []
    z_elems_d = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes_d])
    delta_z = z_elems_d + Lz
    pressao = M1.gama*(delta_z) + press
###############################################
else:
    print("Defina se existe gravidade (True) ou nao (False)")

M1.mb.add_entities(dirichlet_meshset, volumes_d)
M1.mb.add_entities(neumann_meshset, volumes_n)
M1.mb.add_entities(finos_meshset, rng.unite(rng.unite(volumes_n, volumes_d),volumes_f))

#########################################################################################
#jp: modifiquei as tags para sparse
neumann=M1.mb.tag_get_handle("neumann", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
dirichlet=M1.mb.tag_get_handle("dirichlet", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
###############################################################################################

M1.mb.tag_set_data(neumann, volumes_n, np.repeat(1, len(volumes_n)))
M1.mb.tag_set_data(dirichlet, volumes_d, np.repeat(1, len(volumes_d)))

n_volumes_n = len(volumes_n)
vazao = vazao/n_volumes_n
M1.mb.tag_set_data(M1.wells_neumann_tag, 0, neumann_meshset)
M1.mb.tag_set_data(M1.wells_dirichlet_tag, 0, dirichlet_meshset)
M1.mb.tag_set_data(M1.finos_tag, 0, finos_meshset)
M1.mb.tag_set_data(M1.press_value_tag, volumes_d, pressao)
M1.mb.tag_set_data(M1.vazao_value_tag, volumes_n, np.repeat(vazao, len(volumes_n)))

#-------------------------------------------------------------------------------
lxd2=[lx2[0]+l1/2]
if len(lx2)>2:
    for i in np.linspace((lx2[1]+lx2[2])/2,(lx2[-2]+lx2[-3])/2,len(lx2)-3):
        lxd2.append(i)
lxd2.append(lx2[-1]-l1/2)

lyd2=[ly2[0]+l1/2]
if len(ly2)>2:
    for i in np.linspace((ly2[1]+ly2[2])/2,(ly2[-2]+ly2[-3])/2,len(ly2)-3):
        lyd2.append(i)
lyd2.append(ly2[-1]-l1/2)

lzd2=[lz2[0]+l1/2]
if len(lz2)>2:
    for i in np.linspace((lz2[1]+lz2[2])/2,(lz2[-2]+lz2[-3])/2,len(lz2)-3):
        lzd2.append(i)
lzd2.append(lz2[-1]-l1/2)

print("definiu planos do nível 2")

# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(l2/l1)):   lx1.append(i*l1)
for i in range(int(l2/l1)):   ly1.append(i*l1)
for i in range(int(l2/l1)):   lz1.append(i*l1)

lxd1=[xmin+dx0/100]
for i in np.linspace(xmin+1.5*l1,xmax-1.5*l1,int((Lx-3*l1)/l1+1.1)):
    lxd1.append(i)
lxd1.append(xmin+Lx-dx0/100)

lyd1=[ymin+dy0/100]
for i in np.linspace(ymin+1.5*l1,ymax-1.5*l1,int((Ly-3*l1)/l1+1.1)):
    lyd1.append(i)
lyd1.append(ymin+Ly-dy0/100)

lzd1=[zmin+dz0/100]
for i in np.linspace(zmin+1.5*l1,zmax-1.5*l1,int((Lz-3*l1)/l1+1.1)):
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
for e in all_volumes: M1.mb.add_entities(AV_meshset,[e])
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
                if (centroid[0]>lx2[i]) and (centroid[0]<ly2[i]+l2) and (centroid[1]>ly2[j])\
                and (centroid[1]<ly2[j]+l2) and (centroid[2]>lz2[k]) and (centroid[2]<lz2[k]+l2):
                    M1.mb.add_entities(l2_meshset,[elem])
                    M1.mb.remove_entities(AV_meshset,[elem])
                    elem_por_L2=M1.mb.get_entities_by_handle(l2_meshset)

                if i<(len(lxd2)-1) and j<(len(lyd2)-1) and k<(len(lzd2)-1):
                    if (centroid[0]>lxd2[i]-l1/2) and (centroid[0]<lxd2[i+1]+l1/2) and (centroid[1]>lyd2[j]-l1/2)\
                    and (centroid[1]<lyd2[j+1]+l1/2) and (centroid[2]>lzd2[k]-l1/2) and (centroid[2]<lzd2[k+1]+l1/2):

                        M1.mb.add_entities(d2_meshset,[elem])
                        f1a2v3=0
                        if (centroid[0]-lxd2[i])**2<l1**2/4 or (centroid[0]-lxd2[i+1])**2<l1**2/4 :
                            f1a2v3+=1
                        if (centroid[1]-lyd2[j])**2<l1**2/4 or (centroid[1]-lyd2[j+1])**2<l1**2/4:
                            f1a2v3+=1
                        if (centroid[2]-lzd2[k])**2<l1**2/4 or (centroid[2]-lzd2[k+1])**2<l1**2/4:
                            f1a2v3+=1
                        M1.mb.tag_set_data(D2_tag, elem, f1a2v3)
                        M1.mb.tag_set_data(fine_to_primal2_classic_tag, elem, nc2)
            M1.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg=M1.mb.get_entities_by_handle(l2_meshset)
            print(k, len(sg), time.time()-t1)
            t1=time.time()
            d1_meshset=M1.mb.create_meshset()

            M1.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)
            nc2+=1

            for m in range(len(lx1)):
                a=l1*i+m
                for n in range(len(ly1)):
                    b=l1*j+n
                    for o in range(len(lz1)):
                        c=l1*k+o
                        l1_meshset=M1.mb.create_meshset()
                        for e in elem_por_L2:
                            # Refactory here
                            # Verificar se o uso de um vértice reduz o custo
                            centroid=M1.mtu.get_average_position([e])
                            if (centroid[0]>lx2[i]+lx1[m]) and (centroid[0]<lx2[i]+lx1[m]+l1)\
                            and (centroid[1]>ly2[j]+ly1[n]) and (centroid[1]<ly2[j]+ly1[n]+l1)\
                            and (centroid[2]>lz2[k]+lz1[o]) and (centroid[2]<lz2[k]+lz1[o]+l1):
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
#-------------------------------------------------------------------------------
print('Criação da árvore: ',time.time()-t0)
ta=time.time()
all_volumes=M1.all_volumes
'''
vert_meshset=M1.mb.create_meshset()

for e in all_volumes:
    d1_tag = int(M1.mb.tag_get_data(D1_tag, e, flat=True))
    if d1_tag==3:
        M1.mb.add_entities(vert_meshset,[e])
all_vertex_d1=M1.mb.get_entities_by_handle(vert_meshset)
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
print(time.time()-ta,"correção")'''
print("TEMPO TOTAL DE PRÉ PROCESSAMENTO:",time.time()-tempo0_pre)
print(" ")

t0=time.time()
# --------------Atribuição dos IDs de cada nível em cada volume-----------------
# Esse bloco é executado uma vez a cada iteração em um problema bifásico,
# sua eficiência é criticamente importante.

##########################################################################################
# Tag que armazena o ID do volume no nível 1
# jp: modifiquei as tags abaixo para o tipo sparse
L1_ID_tag=M1.mb.tag_get_handle("l1_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# L1ID_tag=M1.mb.tag_get_handle("l1ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# Tag que armazena o ID do volume no nível 2
L2_ID_tag=M1.mb.tag_get_handle("l2_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# L2ID_tag=M1.mb.tag_get_handle("l2ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
# ni = ID do elemento no nível i
L3_ID_tag=M1.mb.tag_get_handle("l3_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
##########################################################################################
# ni = ID do elemento no nível i
n1=0
n2=0
aux=0
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
print("  ")
print("INICIOU SOLUÇÃO ADM")
tempo0_ADM=time.time()
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = M1.mb.get_entities_by_handle(m1)
        for elem1 in elem_by_L1:
            if elem1 in finos:
                aux=1
                tem_poço_no_vizinho=True
            if elem1 in intermediarios:
                tem_poço_no_vizinho=True
        if aux==1:
            aux=0
            for elem in elem_by_L1:
                n1+=1
                n2+=1

                M1.mb.tag_set_data(L1_ID_tag, elem, n1)
                M1.mb.tag_set_data(L2_ID_tag, elem, n2)
                M1.mb.tag_set_data(L3_ID_tag, elem, 1)
                elem_tags = M1.mb.tag_get_tags_on_entity(elem)
                elem_Global_ID = M1.mb.tag_get_data(elem_tags[0], elem, flat=True)
                finos.append(elem)

    if tem_poço_no_vizinho:
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            n2+=1
            t=1
            for elem in elem_by_L1:
                if elem not in finos:
                    M1.mb.tag_set_data(L1_ID_tag, elem, n1)
                    M1.mb.tag_set_data(L2_ID_tag, elem, n2)
                    M1.mb.tag_set_data(L3_ID_tag, elem, 2)
                    t=0
            n1-=t
            n2-=t
    else:
        n2+=1
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            for elem2 in elem_by_L1:
                elem2_tags = M1.mb.tag_get_tags_on_entity(elem)
                M1.mb.tag_set_data(L2_ID_tag, elem2, n2)
                M1.mb.tag_set_data(L1_ID_tag, elem2, n1)
                M1.mb.tag_set_data(L3_ID_tag, elem2, 3)

# ------------------------------------------------------------------------------
print('Definição da malha ADM: ',time.time()-t0)
t0=time.time()

av=M1.mb.create_meshset()
for v in all_volumes:
    M1.mb.add_entities(av,[v])

# fazendo os ids comecarem de 0 em todos os niveis
tags = [L1_ID_tag, L2_ID_tag]
for tag in tags:
    all_gids = M1.mb.tag_get_data(tag, M1.all_volumes, flat=True)
    minim = min(all_gids)
    all_gids -= minim
    M1.mb.tag_set_data(tag, M1.all_volumes, all_gids)

# volumes da malha grossa primal 1
meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))

# volumes da malha grossa primal 2
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))

n_levels = 2
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
all_meshsets = [meshsets_nv1, meshsets_nv2]
for i in range(n_levels):
    meshsets = all_meshsets[i]
    tag_boundary = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets + str(i+2), 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
    utpy.set_faces_in_boundary_by_meshsets(M1.mb, M1.mtu, meshsets, tag_boundary)


for meshset in meshsets_nv2:
    nc = M1.mb.tag_get_data(primal_id_tag2, meshset, flat=True)[0]
    elems = M1.mb.get_entities_by_handle(meshset)
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))
tmod1=time.time()

# Gera a matriz dos coeficientes
#b=np.zeros((len(M1.all_volumes),1))

internos=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
wirebasket_elems = list(internos) + list(faces) + list(arestas) + list(vertices)

M1.mb.tag_set_data(fine_to_primal1_classic_tag,vertices,np.arange(0,len(vertices)))



for meshset in meshsets_nv1:
    elems = M1.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, vertices)
    nc = M1.mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    M1.mb.tag_set_data(fine_to_primal1_classic_tag, elems, np.repeat(nc, len(elems)))
    M1.mb.tag_set_data(primal_id_tag1, meshset, nc)

ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)
tmod2=time.time()

nni=ni
nnf=nni+nf
nne=nnf+na
nnv=nne+nv
l_elems=[internos,faces,arestas,vertices]
l_ids=[0,nni,nnf,nne,nnv]
for i, elems in enumerate(l_elems):
    M1.mb.tag_set_data(M1.ID_reordenado_tag,elems,np.arange(l_ids[i],l_ids[i+1]))

ln=[]
cn=[]
dn=[]

lines=[]
cols=[]
data=[]
for d in volumes_d:
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,d))
    lines.append(ID_global)
    cols.append(0)
    data.append(press)

    ln.append(ID_global)
    cn.append(0)
    dn.append(vazao)

    #b[ID_global]=press
for n in volumes_n:
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,n))
    lines.append(ID_global)
    cols.append(0)
    data.append(vazao)

    ln.append(ID_global)
    cn.append(0)
    dn.append(vazao)
    #b[ID_global]=vazao
del(b)
b=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),1))
bn=csc_matrix((dn,(ln,cn)),shape=(len(M1.all_volumes),1))

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

print("def As")
ty=time.time()
for f in M1.all_faces:
    adjs = M1.mtu.get_bridge_adjacencies(f, 2, 3)
    if len(adjs)>1:
        Gid_1=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[0]))
        Gid_2=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[1]))

        if Gid_1<ni and Gid_2<ni:
            lii.append(Gid_1)
            cii.append(Gid_2)
            dii.append(1)

            lii.append(Gid_2)
            cii.append(Gid_1)
            dii.append(1)

            lii.append(Gid_1)
            cii.append(Gid_1)
            dii.append(-1)

            lii.append(Gid_2)
            cii.append(Gid_2)
            dii.append(-1)

        elif Gid_1<ni and Gid_2>=ni and Gid_2<ni+nf:
            lif.append(Gid_1)
            cif.append(Gid_2-ni)
            dif.append(1)

            lii.append(Gid_1)
            cii.append(Gid_1)
            dii.append(-1)

        elif Gid_2<ni and Gid_1>=ni and Gid_1<ni+nf:
            lif.append(Gid_2)
            cif.append(Gid_1-ni)
            dif.append(1)

            lii.append(Gid_2)
            cii.append(Gid_2)
            dii.append(-1)

        elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni and Gid_2<ni+nf:
            lff.append(Gid_1-ni)
            cff.append(Gid_2-ni)
            dff.append(1)

            lff.append(Gid_2-ni)
            cff.append(Gid_1-ni)
            dff.append(1)

            lff.append(Gid_1-ni)
            cff.append(Gid_1-ni)
            dff.append(-1)

            lff.append(Gid_2-ni)
            cff.append(Gid_2-ni)
            dff.append(-1)

        elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni+nf and Gid_2<ni+nf+na:
            lfe.append(Gid_1-ni)
            cfe.append(Gid_2-ni-nf)
            dfe.append(1)

            lff.append(Gid_1-ni)
            cff.append(Gid_1-ni)
            dff.append(-1)

        elif Gid_2>=ni and Gid_2<ni+nf and Gid_1>=ni+nf and Gid_1<ni+nf+na:
            lfe.append(Gid_2-ni)
            cfe.append(Gid_1-ni-nf)
            dfe.append(1)

            lff.append(Gid_2-ni)
            cff.append(Gid_2-ni)
            dff.append(-1)

        elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf and Gid_2<ni+nf+na:
            lee.append(Gid_1-ni-nf)
            cee.append(Gid_2-ni-nf)
            dee.append(1)

            lee.append(Gid_2-ni-nf)
            cee.append(Gid_1-ni-nf)
            dee.append(1)

            lee.append(Gid_1-ni-nf)
            cee.append(Gid_1-ni-nf)
            dee.append(-1)

            lee.append(Gid_2-ni-nf)
            cee.append(Gid_2-ni-nf)
            dee.append(-1)

        elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf+na:
            lev.append(Gid_1-ni-nf)
            cev.append(Gid_2-ni-nf-na)
            dev.append(1)

            lee.append(Gid_1-ni-nf)
            cee.append(Gid_1-ni-nf)
            dee.append(-1)

        elif Gid_2>=ni+nf and Gid_2<ni+nf+na and Gid_1>=ni+nf+na:
            lev.append(Gid_2-ni-nf)
            cev.append(Gid_1-ni-nf-na)
            dev.append(1)

            lee.append(Gid_2-ni-nf)
            cee.append(Gid_2-ni-nf)
            dee.append(-1)

        elif Gid_1>=ni+nf+na and Gid_2>=ni+nf+na:
            lvv.append(Gid_1)
            cvv.append(Gid_2)
            dvv.append(1)

            lvv.append(Gid_2)
            cvv.append(Gid_1)
            dvv.append(1)

            lvv.append(Gid_1)
            cvv.append(Gid_1)
            dvv.append(-1)

            lvv.append(Gid_2)
            cvv.append(Gid_2)
            dvv.append(-1)

print("took: ",time.time()-ty)
print("get As")
ty=time.time()

Aii=csc_matrix((dii,(lii,cii)),shape=(ni,ni))
Aif=csc_matrix((dif,(lif,cif)),shape=(ni,nf))
Aff=csc_matrix((dff,(lff,cff)),shape=(nf,nf))
Afe=csc_matrix((dfe,(lfe,cfe)),shape=(nf,na))
Aee=csc_matrix((dee,(lee,cee)),shape=(na,na))
Aev=csc_matrix((dev,(lev,cev)),shape=(na,nv))
Avv=csc_matrix((dvv,(lvv,cvv)),shape=(nv,nv))

Ivv=scipy.sparse.identity(nv)

print("took: ",time.time()-ty)

print("get_OP_AMS")
ty=time.time()

#th=time.time()
#M2=-linalg.inv(Aee)*Aev
#print(time.time()-th,"Direto")

invAee=lu_inv(Aee)
M2=-invAee*Aev
P=vstack([M2,Ivv]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

invAff=lu_inv(Aff)
M3=-invAff*Afe*M2
del(M2)
P=vstack([M3,P])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)

invAii=lu_inv(Aii)
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

ID_global=M1.mb.tag_get_data(M1.ID_reordenado_tag,M1.all_volumes)
nivel=M1.mb.tag_get_data(L3_ID_tag,M1.all_volumes)
ids0_d=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d, flat=True)

lines=[int(ID_global[i]) for i in range(len(ID_global)) if nivel[i]>1 and ID_global[i] in ids0_d]
data=[float(corr_1[i]) for i in range(len(ID_global)) if nivel[i]>1 and ID_global[i] in ids0_d]
cols=np.zeros(len(lines))

corr_adm1_d=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),1))


lines=[i for i in range(len(ID_global)) if (nivel[ID_global[i]]>1 and i not in ids0_d)]
data=[float(corr_1[i]) for i in range(len(ID_global)) if (nivel[ID_global[i]]>1 and i not in ids0_d)]
cols=np.zeros(len(lines))

corr_adm1_sd=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),1))
corr_adm1_sd=csc_matrix(corr_1)


ld=[]
cd=[]
dd=[]

for v in volumes_d:
    ID_global= int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    ld.append(ID_global)
    cd.append(0)
    dd.append(float(corr_1[ID_global]))
corr_1=csc_matrix(corr_1)
corr_d1=csc_matrix((dd,(ld,cd)),shape=(len(M1.all_volumes),1))

l=[]
c=[]
d=[]
for v in M1.all_volumes:
    ID_global= int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    l.append(ID_global)
    c.append(0)
    d.append(float(corr_1.toarray()[ID_global]))

corr_sd1=csc_matrix((d,(l,c)),shape=(len(M1.all_volumes),1))

AMS_TO_ADM={}
for v in vertices:
    ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
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
import pdb; pdb.set_trace()

ty=time.time()
print("iniciou____")
nivel_0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
print("get nivel 1___")

matriz=scipy.sparse.find(P)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)
cont=0
for v in nivel_0:
    ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    lines.append(ID_global)
    cols.append(ID_ADM)
    data.append(1)

    dd=np.where(LIN==ID_global)
    LIN=np.delete(LIN,dd,axis=0)
    COL=np.delete(COL,dd,axis=0)
    DAT=np.delete(DAT,dd,axis=0)

print("set_nivel 0")

print("loop", time.time()-ty)

ID_ADM=[AMS_TO_ADM[str(k)] for k in COL]
lines=np.concatenate([lines,LIN])
cols=np.concatenate([cols,ID_ADM])
data=np.concatenate([data,DAT])
print("op_adm", time.time()-ty)
OP_ADM=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),n1))


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
     elem_Global_ID = int(M1.mb.tag_get_data(M1.ID_reordenado_tag, v, flat=True))
     elem_ID1 = int(M1.mb.tag_get_data(L1_ID_tag, v, flat=True))
     lines.append(elem_ID1)
     cols.append(elem_Global_ID)
     data.append(1)
     #OR_ADM[elem_ID1][elem_Global_ID]=1
OR_ADM=csc_matrix((data,(lines,cols)),shape=(n1,len(M1.all_volumes)))

print(time.time()-tmod3,"Tmod3 _________")

#OR_AMS=np.zeros((nv,len(M1.all_volumes)),dtype=np.int)
lines=[]
cols=[]
data=[]
for v in all_volumes:
     elem_Global_ID = int(M1.mb.tag_get_data(M1.ID_reordenado_tag, v))
     AMS_ID = int(M1.mb.tag_get_data(fine_to_primal1_classic_tag, v))
     lines.append(AMS_ID)
     cols.append(elem_Global_ID)
     data.append(1)
     #OR_AMS[AMS_ID][elem_Global_ID]=1
OR_AMS=csc_matrix((data,(lines,cols)),shape=(nv,len(M1.all_volumes)))

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
for f in M1.all_faces:
    adjs = M1.mtu.get_bridge_adjacencies(f, 2, 3)
    if len(adjs)>1:
        Gid_1=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[0]))
        Gid_2=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[1]))
        lines.append(Gid_1)
        cols.append(Gid_2)
        data.append(1)
        #T[Gid_1][Gid_2]=1
        lines.append(Gid_2)
        cols.append(Gid_1)
        data.append(1)
        #T[Gid_2][Gid_1]=1
        lines.append(Gid_1)
        cols.append(Gid_1)
        data.append(-1)
        #T[Gid_1][Gid_1]-=1
        lines.append(Gid_2)
        cols.append(Gid_2)
        data.append(-1)
        #T[Gid_2][Gid_2]-=1

T=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),len(M1.all_volumes)))

T_orig=T
t_assembly=time.time()-t_ass

#----------------------------------------------------
T_AMS=OR_AMS*T*OP_AMS
T_ADM=OR_ADM*T*OP_ADM

v=M1.mb.create_meshset()
M1.mb.add_entities(v,vertices)
tmod12=time.time()
inte=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

M1.mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

for meshset in meshsets_nv2:
    elems = M1.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, ver)
    nc = M1.mb.tag_get_data(fine_to_primal2_classic_tag, vert, flat=True)[0]
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))
    M1.mb.tag_set_data(primal_id_tag2, meshset, nc)


lines=[]
cols=[]
data=[]

nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)
for i in range(nint):
    v=inte[i]
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(i)
    cols.append(ID_AMS)
    data.append(1)

    #G[i][ID_AMS]=1
i=0
for i in range(nfac):
    v=fac[i]
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+i][ID_AMS]=1
i=0
for i in range(nare):
    v=are[i]
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
    lines.append(nint+nfac+i)
    cols.append(ID_AMS)
    data.append(1)
    #G[nint+nfac+i][ID_AMS]=1
i=0

for i in range(nver):
    v=ver[i]
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
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
    ID_AMS_2=int(M1.mb.tag_get_data(fine_to_primal2_classic_tag,v))
    lines.append(ID_AMS_2)
    cols.append(ID_AMS_1)
    data.append(1)
    i+=1
OR_AMS_2=csc_matrix((data,(lines,cols)),shape=(nver,nv))

W_AMS=G*T_AMS*G.transpose()

MPFA_NO_NIVEL_2=True
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
invAee=lu_inv(Aee)
M2=-csc_matrix(invAee)*Aev
P2=vstack([M2,Ivv])

invAff=lu_inv(Aff)

if MPFA_NO_NIVEL_2:
    M3=-invAff*Afe*M2-invAff*Afv
    P2=vstack([M3,P2])
else:
    Mf=-invAff*Afe*M2
    P2=vstack([Mf,P2])
invAii=lu_inv(Aii)
if MPFA_NO_NIVEL_2:
    M3=invAii*(-Aif*M3+Aie*invAee*Aev-Aiv)
    P2=vstack([M3,P2])
else:
    P2=vstack([-invAii*Aif*Mf,P2])



COL_TO_ADM_2={}
# ver é o meshset dos vértices da malha dual grossa
for i in range(nv):
    v=ver[i]
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal2_classic_tag,v))
    ID_ADM=int(M1.mb.tag_get_data(L2_ID_tag,v))
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
for v in M1.all_volumes:
    ID_global=int(M1.mb.tag_get_data(L1_ID_tag,v))
    if ID_global not in My_IDs_2:
        My_IDs_2.append(ID_global)
        ID_ADM=int(M1.mb.tag_get_data(L2_ID_tag,v))
        nivel=M1.mb.tag_get_data(L3_ID_tag,v)
        d1=M1.mb.tag_get_data(D2_tag,v)
        ID_AMS = int(M1.mb.tag_get_data(fine_to_primal1_classic_tag, v))
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
OP_ADM_2=csc_matrix((data,(lines,cols)),shape=(n1,n2))

#for i in range(P2.shape[0]): print(len(np.where(p2[0]==i)[0]))
#####################################################

lines=[]
cols=[]
data=[]
P2=OP_AMS_2

vm=M1.mb.create_meshset()
M1.mb.add_entities(vm,vertices)
for i in range(len(ver)):
    OP_ams2_tag=M1.mb.tag_get_handle("OP_ams2_tag_"+str(i), 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    vals=OP_AMS_2[:,i].toarray()
    M1.mb.tag_set_data(OP_ams2_tag,vertices,vals)
M1.mb.write_file('delete_me.vtk',[vm])

ty=time.time()
print("iniciou____")
m_vert=M1.mb.create_meshset()
M1.mb.add_entities(m_vert,vertices)
nivel_0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_1=M1.mb.get_entities_by_type_and_tag(m_vert, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
print("get níveis 0 e 1___")

P2=csc_matrix(P2)
matriz=scipy.sparse.find(P2)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)
i=1
for v in nivel_1:
    ID_ADM_1=int(M1.mb.tag_get_data(L1_ID_tag,v))
    ID_ADM_2=int(M1.mb.tag_get_data(L2_ID_tag,v))
    ID_AMS = int(M1.mb.tag_get_data(fine_to_primal1_classic_tag, v))
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
    ID_ADM_1=int(M1.mb.tag_get_data(L1_ID_tag,v))
    ID_ADM_2=int(M1.mb.tag_get_data(L2_ID_tag,v))
    ID_AMS = int(M1.mb.tag_get_data(fine_to_primal1_classic_tag, v))
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

OP_ADM_3=csc_matrix((data,(lines,cols)),shape=(n1,n2))

#OP_ADM_2=OP_ADM_3

####################################################
#OR_ADM_2=np.zeros((n2,len(T_ADM)),dtype=np.int)
lines=[]
cols=[]
data=[]
OR_ADM_2=np.zeros((n2,n1),dtype=np.int)
for v in M1.all_volumes:
    elem_ID2 = int(M1.mb.tag_get_data(L2_ID_tag, v, flat=True))
    elem_Global_ID = int(M1.mb.tag_get_data(L1_ID_tag, v, flat=True))
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

ID_global=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d, flat=True)
ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
ID_ADM_2=M1.mb.tag_get_data(L2_ID_tag,volumes_d, flat=True)
T[ID_global]=scipy.sparse.csc_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global]=np.ones(len(ID_global))

########################## apagar para usar pressão-vazão
# ID_globaln=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n, flat=True)
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
    AMS_ID=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
    ld.append(AMS_ID)
    cd.append(0)
    dd.append(float(corr.toarray()[AMS_ID]))
corr_d=csc_matrix((dd,(ld,cd)),shape=(len(vertices),1))

l=[]
c=[]
d=[]
for v in vertices:
    AMS_ID=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
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

ID_ADM_1=M1.mb.tag_get_data(L1_ID_tag,vertices)
nivel=M1.mb.tag_get_data(L3_ID_tag,vertices)
ids1_d=sorted(set(M1.mb.tag_get_data(fine_to_primal1_classic_tag,volumes_d, flat=True)))

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
pms_adm_nv1_tag  = M1.mb.tag_get_handle('PMS_ADM_NV1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(pms_adm_nv1_tag, wirebasket_elems, SOL_ADM_fina_1)
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

ERRO_tag=M1.mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROams1_tag=M1.mb.tag_get_handle("erroAMS1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROams2_tag=M1.mb.tag_get_handle("erroAMS2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROadm1_tag=M1.mb.tag_get_handle("erroADM1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_TPFA_tag=M1.mb.tag_get_handle("Pressão TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_ADM_tag=M1.mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

for v in M1.all_volumes:
    gid=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    M1.mb.tag_set_data(ERRO_tag,v,erro[gid])
    M1.mb.tag_set_data(ERROams1_tag,v,erroAMS1[gid])
    M1.mb.tag_set_data(ERROams2_tag,v,SOL_AMS_fina_2[gid])
    M1.mb.tag_set_data(ERROadm1_tag,v,erroADM1[gid])
    M1.mb.tag_set_data(Sol_TPFA_tag,v,SOL_TPFA[gid])
    M1.mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])

CORR_tag=M1.mb.tag_get_handle("corr_ams2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
CORR1_tag=M1.mb.tag_get_handle("corr_ams1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
CORR_ADM_2_tag=M1.mb.tag_get_handle("corr_adm2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
CORR_ADM_1_tag=M1.mb.tag_get_handle("corr_adm1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
corr=corr.toarray()
corr_1=corr_1.toarray()
corr_adm2_sd=corr_adm2_sd.toarray()
corr_adm1_sd=corr_adm1_sd.toarray()

i=0
for v in M1.all_volumes:
    gid=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))
    ID_AMS=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
    ID_ADM_1=int(M1.mb.tag_get_data(L1_ID_tag,v))
    M1.mb.tag_set_data(CORR1_tag,v,abs(corr_1[gid]))
    M1.mb.tag_set_data(CORR_ADM_2_tag,v,abs(corr_adm2_sd[ID_ADM_1]))
    M1.mb.tag_set_data(CORR_ADM_1_tag,v,abs(corr_adm1_sd[gid]))
    M1.mb.tag_set_data(CORR_tag,v,abs(corr[ID_AMS]))

teste_tag=M1.mb.tag_get_handle("teste", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
teste=OP_ADM*corr_adm2_sd
for v in M1.all_volumes: M1.mb.tag_set_data(teste_tag,v,abs(teste[int(M1.mb.tag_get_data(M1.ID_reordenado_tag,v))]))

p_tag = Sol_ADM_tag
# name_tag_faces_boundary_meshsets
coarse_flux_nv3_tag = M1.mb.tag_get_handle('Q_nv3', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
# oth1 = oth(M1.mb, M1.mtu)
tag_faces_bound_nv2 = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(3))
all_faces_boundary_nv3 = M1.mb.tag_get_data(tag_faces_bound_nv2, 0, flat=True)[0]
all_faces_boundary_nv3 = M1.mb.get_entities_by_handle(all_faces_boundary_nv3)

for m in meshsets_nv2:
    qtot = 0.0
    elems = M1.mb.get_entities_by_handle(m)
    gids_nv2_adm = np.unique(M1.mb.tag_get_data(L2_ID_tag, elems, flat=True))
    if len(gids_nv2_adm) > 1:
        continue
    faces = M1.mtu.get_bridge_adjacencies(elems, 3, 2)
    faces = rng.intersect(faces, all_faces_boundary_nv3)
    for face in faces:
        keq, s_grav, elems2 = oth.get_kequiv_by_face_quad(M1.mb, M1.mtu, face, M1.perm_tag, M1.area_tag)
        p = M1.mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (p[1] - p[0])*keq
        if elems2[0] in elems:
            qtot += flux
        else:
            qtot -= flux

    M1.mb.tag_set_data(coarse_flux_nv3_tag, elems, np.repeat(qtot, len(elems)))


p_tag = pms_adm_nv1_tag
gids_nv0 = M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
map_global = dict(zip(M1.all_volumes, gids_nv0))
# name_tag_faces_boundary_meshsets
coarse_flux_nv2_tag = M1.mb.tag_get_handle('Q_nv2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
oth1 = oth(M1.mb, M1.mtu)
tag_faces_bound_nv1 = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(2))
all_faces_boundary_nv2 = M1.mb.tag_get_data(tag_faces_bound_nv1, 0, flat=True)[0]
all_faces_boundary_nv2 = M1.mb.get_entities_by_handle(all_faces_boundary_nv2)
M1.mb.tag_set_data(M1.keq_tag, M1.all_faces, np.repeat(1.0, len(M1.all_faces)))
M1.mb.tag_set_data(M1.s_grav_tag, M1.all_faces, np.repeat(0.0, len(M1.all_faces)))
for m in meshsets_nv1:
    qtot = 0.0
    elems = M1.mb.get_entities_by_handle(m)
    gids_nv1_adm = np.unique(M1.mb.tag_get_data(L1_ID_tag, elems, flat=True))
    if len(gids_nv1_adm) > 1:
        continue
    faces = M1.mtu.get_bridge_adjacencies(elems, 3, 2)
    faces = rng.intersect(faces, all_faces_boundary_nv2)
    for face in faces:
        keq, s_grav, elems2 = oth.get_kequiv_by_face_quad(M1.mb, M1.mtu, face, M1.perm_tag, M1.area_tag)
        p = M1.mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (p[1] - p[0])*keq
        if elems2[0] in elems:
            qtot += flux
        else:
            qtot -= flux

    M1.mb.tag_set_data(coarse_flux_nv2_tag, elems, np.repeat(qtot, len(elems)))

M1.mb.write_file('teste_3D_unstructured_18.vtk',[av])
print('New file created')
print(min(erro),max(erro))
import pdb; pdb.set_trace()
# (-T*B1*C2*X1*G+B1*C2 + OP_ADM*C2*OR_ADM)*q
# Ai=OR_ADM*OR_ADM_2*lu_inv(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2)*OR_ADM_2*OP_ADM*(scipy.sparse.identity(len(M1.all_volumes))-T*OP_ADM*corr)
