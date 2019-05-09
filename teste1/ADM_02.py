import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
from matplotlib import pyplot as plt
import sympy
import cython
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, vstack, hstack, linalg, identity, find
from processor import conversao as conv
from processor import def_intermediarios as definter

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')

os.chdir(parent_dir)

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
        self.all_faces = self.mb.get_entities_by_dimension(0, self.dimension-1) #ADJs=np.array([M1.mb.get_adjacencies(face, 3) for face in M1.all_faces])

        self.dirichlet_faces = set()
        self.neumann_faces = set()

        '''self.GLOBAL_ID_tag = self.mb.tag_get_handle(
            "Global_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)'''

        self.create_tags()
        self.set_k_and_phi_structured_spe10()
        #self.set_information("PERM", self.all_volumes, 3)
        self.get_boundary_faces()
        self.gravity = False
        self.gama = 10
        self.mi = 1
        t0=time.time()
        print('set área')
        self.get_kequiv_by_face_quad(self.all_faces)
        '''
        print('set área')
        for f in self.all_faces:
            self.set_area(f)'''
        print("took",time.time()-t0)
        self.get_faces_boundary


    def create_tags(self):
        self.perm_tag = self.mb.tag_get_handle("PERM", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.finos_tag = self.mb.tag_get_handle("finos", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_dirichlet_tag = self.mb.tag_get_handle("WELLS_D", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_neumann_tag = self.mb.tag_get_handle("WELLS_N", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.press_value_tag = self.mb.tag_get_handle("P", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.vazao_value_tag = self.mb.tag_get_handle("Q", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.all_faces_boundary_tag = self.mb.tag_get_handle("FACES_BOUNDARY", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.area_tag = self.mb.tag_get_handle("AREA", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.GLOBAL_ID_tag = self.mb.tag_get_handle("G_ID_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.ID_reordenado_tag = self.mb.tag_get_handle("ID_reord_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.phi_tag = self.mb.tag_get_handle("PHI", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.k_eq_tag = self.mb.tag_get_handle("K_EQ", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)


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
            area = np.cross(n1, n2)/2.0

        #calculo da area para quadrilatero regular
        elif len(points) == 4:
            n = np.array([np.array(points[0] - points[1]), np.array(points[0] - points[2]), np.array(points[0] - points[3])])
            norms = np.array(list(map(np.linalg.norm, n)))
            ind_norm_max = np.where(norms == max(norms))[0]
            n = np.delete(n, ind_norm_max, axis = 0)
            area = np.cross(n[0], n[1])
        self.mb.tag_set_data(self.area_tag, face, area)

    def calc_area(self, face):
        points = self.mtu.get_bridge_adjacencies(face, 2, 0)
        points = [self.mb.get_coords([vert]) for vert in points]
        if len(points) == 3:
            n1 = np.array(points[0] - points[1])
            n2 = np.array(points[0] - points[2])
            area = np.cross(n1, n2)/2.0

        #calculo da area para quadrilatero regular
        elif len(points) == 4:
            n = np.array([np.array(points[0] - points[1]), np.array(points[0] - points[2]), np.array(points[0] - points[3])])
            norms = np.array(list(map(np.linalg.norm, n)))
            ind_norm_max = np.where(norms == max(norms))[0]
            n = np.delete(n, ind_norm_max, axis = 0)
            area = np.cross(n[0], n[1])
        return(area)

    def get_kequiv_by_face_quad(self, conj_faces):
        """
        retorna os valores de k equivalente para colocar na matriz
        a partir da face

        input:
            face: face do elemento
        output:
            kequiv: k equivalente
            elems: elementos vizinhos pela face
            s: termo fonte da gravidade
        """
        k2 = conv.pe_to_m(1.0)
        ADJs=np.array([self.mb.get_adjacencies(face, 3) for face in self.all_faces])

        centroids=np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])

        ADJsv=np.array([self.mb.get_adjacencies(face, 0) for face in self.all_faces])

        ks=self.mb.tag_get_data(self.perm_tag, self.all_volumes)
        #vol_to_pos=dict(zip(M1.all_volumes,range(len(M1.all_volumes))))
        vol_to_pos=dict(zip(self.all_volumes,range(len(self.all_volumes))))
        cont=0
        K_eq=[]
        for f in self.all_faces:
            adjs=ADJs[cont]
            adjsv=ADJsv[cont]
            cont+=1
            if len(adjs)==2:
                v1=adjs[0]
                v2=adjs[1]
                k1 = ks[vol_to_pos[v1]].reshape([3, 3])
                k2 = ks[vol_to_pos[v2]].reshape([3, 3])
                centroid1 = centroids[vol_to_pos[v1]]
                centroid2 = centroids[vol_to_pos[v2]]
                direction = centroid2 - centroid1
                norm=np.linalg.norm(direction)
                uni = np.absolute(direction/norm)
                k1 = np.dot(np.dot(k1,uni), uni)
                k2 = np.dot(np.dot(k2,uni), uni)
                k_harm = (2*k1*k2)/(k1+k2)

                vertex_cent=np.array([self.mb.get_coords([np.uint64(a)]) for a in adjsv])
                dx=max(vertex_cent[:,0])-min(vertex_cent[:,0])
                dy=max(vertex_cent[:,1])-min(vertex_cent[:,1])
                dz=max(vertex_cent[:,2])-min(vertex_cent[:,2])
                if dx<0.001:
                    dx=1
                if dy<0.001:
                    dy=1
                if dz<0.001:
                    dz=1
                area=dx*dy*dz
                #area = self.mb.tag_get_data(self.area_tag, face, flat=True)[0]
                #s_gr = self.gama*keq*(centroid2[2]-centroid1[2])
                keq = k_harm*area/(self.mi*norm)

                K_eq.append(keq)
            else:
                K_eq.append(0.0)
        self.mb.tag_set_data(self.k_eq_tag, self.all_faces, K_eq)

    def set_k_and_phi_structured_spe10(self):
        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']
        k2 = 1.0
        k2 = conv.milidarcy_to_m2(k2)
        ks *= k2

        nx = 60
        ny = 220
        nz = 85
        perms = []
        phis = []

        k = 1.0  #para converter a unidade de permeabilidade
        centroids=np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])
        cont=0
        for v in self.all_volumes:
            centroid = centroids[cont]
            cont+=1
            ijk = np.array([centroid[0]//20.0, centroid[1]//10.0, centroid[2]//2.0])
            e = int(ijk[0] + ijk[1]*nx + ijk[2]*nx*ny)
            # perm = ks[e]*k
            # fi = phi[e]
            perms.append(ks[e]*k)
            phis.append(phi[e])

        self.mb.tag_set_data(self.perm_tag, self.all_volumes, perms)
        self.mb.tag_set_data(self.phi_tag, self.all_volumes, phis)

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
        self.all_faces_boundary=self.mb.get_entities_by_handle(all_faces_boundary_set)
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
            elems = self.mtu.get_bridge_adjacencies(face, 2, 3)
            if len(elems) < 2:
                self.mb.add_entities(all_boundary_faces, [face])

        self.mb.tag_set_data(self.all_faces_boundary_tag, 0, all_boundary_faces)
        self.all_boundary_faces=self.mb.get_entities_by_handle(all_boundary_faces)



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

def get_box(conjunto, all_centroids, limites, return_inds):
    # conjunto-> lista
    # all_centroids->coordenadas dos centroides do conjunto
    # limites-> diagonal que define os volumes objetivo (numpy array com duas coordenadas)
    # Retorna os volumes pertencentes a conjunto cujo centroide está dentro de limites
    inds0 = np.where(all_centroids[:,0] > limites[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > limites[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > limites[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < limites[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < limites[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < limites[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols = list(c1 & c2)
    if return_inds:
        return (rng.Range(np.array(conjunto)[inds_vols]),inds_vols)
    else:
        return rng.Range(np.array(conjunto)[inds_vols])

#--------------Início dos parâmetros de entrada-------------------
# M1= MeshManager('27x27x27.msh')          # Objeto que armazenará as informações da malha
M1= MeshManager('30x30x45.msh')          # Objeto que armazenará as informações da malha
all_volumes=M1.all_volumes

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)

M1.all_centroids=np.array([M1.mtu.get_average_position([v]) for v in all_volumes])
all_centroids = M1.all_centroids

nx=30
ny=30
nz=45

lx=20
ly=10
lz=2

x1=nx*lx
y1=ny*ly
z1=nz*lz
# Distância, em relação ao poço, até onde se usa malha fina
r0 = 4
# Distância, em relação ao poço, até onde se usa malha intermediária
r1 = 1

bvd = np.array([np.array([x1-lx, 0.0, 0.0]), np.array([x1, y1, lz])])
bvn = np.array([np.array([0.0, 0.0, z1-lz]), np.array([lx, y1, z1])])
# bvd = np.array([np.array([0.0, 0.0, 0.0]), np.array([lx, ly, z1])])
# bvn = np.array([np.array([x1-lx, y1-ly, 0.0]), np.array([x1, y1, z1])])
'''
bvd = np.array([np.array([0.0, 0.0, 0.0]), np.array([lx, ly, z1])])
bvn = np.array([np.array([x1-lx, y1-ly, 0.0]), np.array([x1, y1, z1])])'''
#bvd = np.array([np.array([0.0, 0.0, y2]), np.array([y0, y0, y0])])
#bvn = np.array([np.array([0.0, 0.0, 0.0]), np.array([y0, y0, y1])])

bvfd = np.array([np.array([bvd[0][0]-r0*lx, bvd[0][1]-r0*ly, bvd[0][2]-r0*lz]), np.array([bvd[1][0]+r0*lx, bvd[1][1]+r0*ly, bvd[1][2]+r0*lz])])
bvfn = np.array([np.array([bvn[0][0]-r0*lx, bvn[0][1]-r0*ly, bvn[0][2]-r0*lz]), np.array([bvn[1][0]+r0*lx, bvn[1][1]+r0*ly, bvn[1][2]+r0*lz])])

bvid = np.array([np.array([bvd[0][0]-r1, bvd[0][1]-r1, bvd[0][2]-r1]), np.array([bvd[1][0]+r1, bvd[1][1]+r1, bvd[1][2]+r1])])
bvin = np.array([np.array([bvn[0][0]-r1, bvn[0][1]-r1, bvn[0][2]-r1]), np.array([bvn[1][0]+r1, bvn[1][1]+r1, bvn[1][2]+r1])])

# volumes com pressao prescrita

volumes_d, inds_vols_d= get_box(M1.all_volumes, all_centroids, bvd, True)

# volumes com vazao prescrita
volumes_n, inds_vols_n = get_box(M1.all_volumes, all_centroids, bvn, True)

# volumes finos por neumann
volumes_fn = get_box(M1.all_volumes, all_centroids, bvfn, False)

# volumes finos por Dirichlet
volumes_fd = get_box(M1.all_volumes, all_centroids, bvfd, False)

volumes_f=rng.unite(volumes_fn,volumes_fd)

inds_pocos = inds_vols_d + inds_vols_n
Cent_wels = all_centroids[inds_pocos]

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
l1=[3*lx,3*ly,3*lz]
l2=[9*lx,9*ly,9*lz]
# Posição aproximada de cada completação



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
'''
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
'''
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

def lu_inv3(M,lines):
    lines=np.array(lines)
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
        B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
        B=B.toarray()
        inversa=csc_matrix(LU.solve(B,'T')).transpose()
    else:
        c=range(s)
        d=np.repeat(1,s)
        for i in range(n):
            l=lines[s*i:s*(i+1)]
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],s))
            B=B.toarray()
            if i==0:
                inversa=csc_matrix(LU.solve(B,'T')).transpose()
            else:
                inversa=csc_matrix(vstack([inversa,csc_matrix(LU.solve(B,'T')).transpose()]))

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(vstack([inversa,csc_matrix(LU.solve(B,'T')).transpose()]))
    f=find(inversa)
    ll=f[0]
    c=f[1]
    d=f[2]
    pos_to_line=dict(zip(range(len(lines)),lines))
    lg=[pos_to_line[l] for l in ll]
    inversa=csc_matrix((d,(lg,c)),shape=(M.shape[0],M.shape[0]))
    #print(time.time()-tinv,L,"tempo de inversão")
    return inversa

def lu_inv4(M,lines):
    lines=np.array(lines)
    cols=lines
    L=len(lines)
    s=500
    n=int(L/s)
    r=int(L-int(L/s)*s)
    tinv=time.time()
    LU=linalg.splu(M)
    if L<s:
        l=lines
        c=range(len(l))
        d=np.repeat(1,L)
        B=csr_matrix((d,(l,c)),shape=(M.shape[0],L))
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
                inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

        if r>0:
            l=lines[s*n:L]
            c=range(r)
            d=np.repeat(1,r)
            B=csc_matrix((d,(l,c)),shape=(M.shape[0],r))
            B=B.toarray()
            inversa=csc_matrix(hstack([inversa,csc_matrix(LU.solve(B))]))

    tk1=time.time()
    #f=find(inversa.tocsr())
    #l=f[0]
    #cc=f[1]
    #d=f[2]
    #pos_to_col=dict(zip(range(len(cols)),cols))
    #cg=[pos_to_col[c] for c in cc]
    lp=range(len(cols))
    cp=cols
    dp=np.repeat(1,len(cols))
    permut=csc_matrix((dp,(lp,cp)),shape=(len(cols),M.shape[0]))
    inversa=csc_matrix(inversa*permut)

    #inversa1=csc_matrix((d,(l,cg)),shape=(M.shape[0],M.shape[0]))
    #inversa=inversa1
    print(tk1-tinv,L,time.time()-tk1,len(lines),'/',M.shape[0],"tempo de inversão")
    return inversa


all_volumes=M1.all_volumes
print("Volumes:",all_volumes)
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

pocos_meshset=M1.mb.create_meshset()

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
for i in range(int(Lx/l2[0])):    lx2.append(xmin+i*l2[0])
for i in range(int(Ly/l2[1])):    ly2.append(ymin+i*l2[1])
for i in range(int(Lz/l2[2])):    lz2.append(zmin+i*l2[2])
lx2.append(Lx)
ly2.append(Ly)
lz2.append(Lz)
#-------------------------------------------------------------------------------
press = 4000.0
vazao = 10000.0
press = conv.psi_to_Pa(press)
vazao = conv.psi_to_Pa(vazao)
dirichlet_meshset = M1.mb.create_meshset()
neumann_meshset = M1.mb.create_meshset()

M1.gravity = False
Lz2 = conv.pe_to_m(Lz)

if M1.gravity == False:
    pressao = np.repeat(press, len(volumes_d))
    vazao = np.repeat(vazao, len(volumes_n))

# # colocar gravidade
elif M1.gravity == True:
    pressao = []
    z_elems_d = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes_d])
    z_elems_n = -1*np.array([M1.mtu.get_average_position([v])[2] for v in volumes_n])
    delta_z = z_elems_d + Lz2
    pressao = M1.gama*(delta_z) + press
    delta_z = z_elems_n + Lz2
    vazao = M1.gama*(delta_z) + vazao
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

M1.mb.tag_set_data(M1.wells_neumann_tag, 0, neumann_meshset)
M1.mb.tag_set_data(M1.wells_dirichlet_tag, 0, dirichlet_meshset)
M1.mb.tag_set_data(M1.finos_tag, 0, finos_meshset)
M1.mb.tag_set_data(M1.press_value_tag, volumes_d, pressao)
# M1.mb.tag_set_data(M1.press_value_tag, volumes_n, np.repeat(vazao, len(volumes_n)))
M1.mb.tag_set_data(M1.press_value_tag, volumes_n, vazao)

#-------------------------------------------------------------------------------
# Vetor que define a "grade" que separa os volumes da malha fina
# Essa grade é relativa a cada um dos blocos da malha grossa
lx1, ly1, lz1 = [], [], []
for i in range(int(l2[0]/l1[0])):   lx1.append(i*l1[0])
for i in range(int(l2[1]/l1[1])):   ly1.append(i*l1[1])
for i in range(int(l2[2]/l1[2])):   lz1.append(i*l1[2])


D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
nD_x=int((D_x+0.001)/l1[0])
nD_y=int((D_y+0.001)/l1[1])
nD_z=int((D_z+0.001)/l1[2])


lxd1=[xmin+dx0/100]
for i in range(int(Lx/l1[0])-2-nD_x):
    lxd1.append(l1[0]/2+(i+1)*l1[0])
lxd1.append(xmin+Lx-dx0/100)

lyd1=[ymin+dy0/100]
for i in range(int(Ly/l1[1])-2-nD_y):
    lyd1.append(l1[1]/2+(i+1)*l1[1])
lyd1.append(ymin+Ly-dy0/100)

lzd1=[zmin+dz0/100]

for i in range(int(Lz/l1[2])-2-nD_z):
    lzd1.append(l1[2]/2+(i+1)*l1[2])
lzd1.append(xmin+Lz-dz0/100)

#lzd1[-2]=21.5

#lzd1[0]=1.5
#lzd1[-2]=23.5
#lzd1[-3]=20.5
print("definiu planos do nível 1")
lxd2=[lxd1[0]]
for i in range(1,int(len(lxd1)*l1[0]/l2[0])-1):
    lxd2.append(lxd1[int(i*l2[0]/l1[0]+0.0001)+1])
lxd2.append(lxd1[-1])

lyd2=[lyd1[0]]
for i in range(1,int(len(lyd1)*l1[1]/l2[1])-1):
    lyd2.append(lyd1[int(i*l2[1]/l1[1]+0.00001)+1])
lyd2.append(lyd1[-1])

lzd2=[lzd1[0]]
for i in range(1,int(len(lzd1)*l1[2]/l2[2])-1):
    lzd2.append(lzd1[int(i*l2[2]/l1[2]+0.00001)+1])
lzd2.append(lzd1[-1])

print("definiu planos do nível 2")


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
nc2=0

D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])

centroids=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in M1.all_volumes])
sx=0
ref_dual=False
M1.mb.add_entities(AV_meshset,all_volumes)
for i in range(len(lx2)-1):
    t1=time.time()
    if i==len(lx2)-2:
        sx=D_x
    sy=0
    for j in range(len(ly2)-1):
        if j==len(ly2)-2:
            sy=D_y
        sz=0
        for k in range(len(lz2)-1):
            if k==len(lz2)-2:
                sz=D_z
            l2_meshset=M1.mb.create_meshset()
            cont=0
            box_primal2 = np.array([np.array([lx2[i], ly2[j], lz2[k]]), np.array([lx2[i]+l2[0]+sx, ly2[j]+l2[1]+sy, lz2[k]+l2[2]+sz])])
            elem_por_L2 = get_box(M1.all_volumes, centroids, box_primal2, False)
            M1.mb.add_entities(l2_meshset,elem_por_L2)
            centroid_p2=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
            for elem in elem_por_L2:
                centroid=centroid_p2[cont]
                cont+=1
                f1a2v3=0
                if (centroid[0]-lxd2[i])**2<=l1[0]**2/4:
                    f1a2v3+=1
                if (centroid[1]-lyd2[j])**2<=l1[1]**2/4:
                    f1a2v3+=1
                if (centroid[2]-lzd2[k])**2<=l1[2]**2/4:
                    f1a2v3+=1
                M1.mb.tag_set_data(D2_tag, elem, f1a2v3)
                M1.mb.tag_set_data(fine_to_primal2_classic_tag, elem, nc2)

            M1.mb.add_child_meshset(L2_meshset,l2_meshset)
            sg=M1.mb.get_entities_by_handle(l2_meshset)
            print(k, len(sg), time.time()-t1)
            t1=time.time()
            M1.mb.tag_set_data(primal_id_tag2, l2_meshset, nc2)

            centroids_primal2=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L2])
            nc2+=1
            s1x=0
            for m in range(len(lx1)):
                a=int(l2[0]/l1[0])*i+m
                if Lx-D_x==lx2[i]+lx1[m]+l1[0]:# and D_x==Lx-int(Lx/l1[0])*l1[0]:
                    s1x=D_x
                s1y=0
                for n in range(len(ly1)):
                    b=int(l2[1]/l1[1])*j+n
                    if Ly-D_y==ly2[j]+ly1[n]+l1[1]:# and D_y==Ly-int(Ly/l1[1])*l1[1]:
                        s1y=D_y
                    s1z=0

                    for o in range(len(lz1)):
                        c=int(l2[2]/l1[2])*k+o
                        if Lz-D_z==lz2[k]+lz1[o]+l1[2]:
                            s1z=D_z
                        l1_meshset=M1.mb.create_meshset()

                        box_primal1 = np.array([np.array([lx2[i]+lx1[m], ly2[j]+ly1[n], lz2[k]+lz1[o]]), np.array([lx2[i]+lx1[m]+l1[0]+s1x, ly2[j]+ly1[n]+l1[1]+s1y, lz2[k]+lz1[o]+l1[2]+s1z])])
                        elem_por_L1 = get_box(elem_por_L2, centroids_primal2, box_primal1, False)
                        M1.mb.add_entities(l1_meshset,elem_por_L1)
                        #centroid_p1=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in elem_por_L1])
                        cont1=0
                        values_1=[]
                        faces1=[]
                        internos1=[]
                        for e in elem_por_L1:
                            #centroid=centroid_p1[cont1]
                            cont1+=1
                            f1a2v3=0
                            M_M=Min_Max(e)
                            if (M_M[0]<lxd1[a] and M_M[1]>=lxd1[a]):
                                f1a2v3+=1
                            if (M_M[2]<lyd1[b] and M_M[3]>=lyd1[b]):
                                f1a2v3+=1
                            if (M_M[4]<lzd1[c] and M_M[5]>=lzd1[c]):
                                f1a2v3+=1
                            values_1.append(f1a2v3)

                            if ref_dual:
                                if f1a2v3==0:
                                    internos1.append(e)
                                if f1a2v3==1:
                                    faces1.append(e)
                                elif f1a2v3==3:
                                    vertice=e
                        M1.mb.tag_set_data(D1_tag, elem_por_L1,values_1)

                        M1.mb.tag_set_data(fine_to_primal1_classic_tag, elem_por_L1, np.repeat(nc1,len(elem_por_L1)))

                        # Enriquece a malha dual
                        if ref_dual:
                            #viz_vert=rng.unite(rng.Range(vertice),M1.mtu.get_bridge_adjacencies(vertice, 1, 3))
                            viz_vert=M1.mtu.get_bridge_adjacencies(vertice, 1, 3)
                            cent_v=cent=M1.mtu.get_average_position([np.uint64(vertice)])
                            new_vertices=[]
                            perm1=M1.mb.tag_get_data(M1.perm_tag,viz_vert)
                            perm1_x=perm1[:,0]
                            perm1_y=perm1[:,4]
                            perm1_z=perm1[:,8]
                            r=False
                            r_p=0
                            #print(max(perm1_x)/min(perm1_x),max(perm1_y)/min(perm1_y),max(perm1_z)/min(perm1_z))
                            if max(perm1_x)>r_p*min(perm1_x) or max(perm1_y)>r_p*min(perm1_y) or max(perm1_z)>r_p*min(perm1_z):
                                r=True
                            #print(max(perm1_x)/min(perm1_x))
                            #rng.subtract(rng.Range(vertice),viz_vert)
                            for v in viz_vert:
                                cent=M1.mtu.get_average_position([np.uint64(v)])
                                if (cent[2]-cent_v[2])<0.01 and r:# and v in faces1:
                                    new_vertices.append(v)

                            adjs_new_vertices=[M1.mtu.get_bridge_adjacencies(v,2,3) for v in new_vertices]

                            new_faces=[]
                            for conj in adjs_new_vertices:
                                v=rng.intersect(rng.Range(internos1),conj)
                                if len(v)>0:
                                    new_faces.append(np.uint64(v))
                            for f in new_faces:
                                try:
                                    vfd=0
                                    #M1.mb.tag_set_data(D1_tag, f,np.repeat(1,len(f)))
                                except:
                                    import pdb; pdb.set_trace()

                        #M1.mb.tag_set_data(D1_tag, new_vertices,np.repeat(2,len(new_vertices)))
                        M1.mb.tag_set_data(primal_id_tag1, l1_meshset, nc1)
                        nc1+=1
                        M1.mb.add_child_meshset(l2_meshset,l1_meshset)
#-------------------------------------------------------------------------------

print('Criação da árvore de meshsets primais: ',time.time()-t0)

ta=time.time()
all_volumes=M1.all_volumes
vert_meshset=M1.mb.create_meshset()

#################################################################################################
# setando faces de contorno
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# adm_mod_dir = os.path.join(parent_dir, 'ADM_mod_2-master')
# parent_parent_dir = adm_mod_dir
# input_dir = os.path.join(parent_parent_dir, 'input')
# flying_dir = os.path.join(parent_parent_dir, 'flying')
# utils_dir = os.path.join(parent_parent_dir, 'utils')

import importlib.machinery
# loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
# utpy = loader.load_module('pymoab_utils')
from utils import pymoab_utils as utpy

meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))


n_levels = 2
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
all_meshsets = [meshsets_nv1, meshsets_nv2]
t0 = time.time()

for i in range(n_levels):
    meshsets = all_meshsets[i]
    # names_tags_criadas_aqui.append(name_tag_faces_boundary_meshsets + str(i+2))
    tag_boundary = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets + str(i+2), 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
    utpy.set_faces_in_boundary_by_meshsets(M1.mb, M1.mtu, meshsets, tag_boundary)
t1 = time.time()
print('tempo faces contorno')
print(t1-t0)
###################################################################################################

#for e in all_volumes:
#    d1_tag = int(M1.mb.tag_get_data(D1_tag, e, flat=True))
#    if d1_tag==3:
#        M1.mb.add_entities(vert_meshset,[e])
all_vertex_d1=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
all_vertex_d1=np.uint64(np.array(rng.unite(all_vertex_d1,M1.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))))
mm=0

vertex_centroids=np.array([M1.mtu.get_average_position([np.uint64(v)]) for v in all_vertex_d1])

'''
for x in lxd1:
    for y in lyd1:
        for z in lzd1:
            bv_vert = np.array([np.array([x-dx0, y-dy0, z-dz0]), np.array([x+dx0, y+dy0, z+dz0])])
            volumes_c = get_box(all_vertex_d1, vertex_centroids, bv_vert, False)
            v1 = all_vertex_d1[0]
            c=M1.mtu.get_average_position([np.uint(v1)])
            d=(c[0]-x)**2+(c[1]-y)**2+(c[2]-z)**2
            for e in volumes_c:
                c=M1.mtu.get_average_position([e])
                dist=(c[0]-x)**2+(c[1]-y)**2+(c[2]-z)**2
                if dist<d:
                    d=dist
                    v1=e
            M1.mb.tag_set_data(D1_tag, np.uint(v1), 4)

all_vertex_d1=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
all_vertex_d1=np.uint64(np.array(rng.unite(all_vertex_d1,M1.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))))
for e in all_vertex_d1:

    M1.mb.tag_set_data(D1_tag, np.uint64(e), 2)
all_vertex_d1=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([4]))
all_vertex_d1=np.uint64(np.array(rng.unite(all_vertex_d1,M1.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([4])))))

for e in all_vertex_d1: M1.mb.tag_set_data(D1_tag, np.uint(e), 3)
print(time.time()-ta,"correção")
t0=time.time()'''


# --------------Atribuição dos IDs de cada nível em cada volume-----------------
# Esse bloco é executado uma vez a cada iteração em um problema bifásico,
# sua eficiência é criticamente importante.

##########################################################################################
vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
vertices=rng.unite(vertices,M1.mb.get_entities_by_type_and_tag(0, types.MBTET, np.array([D1_tag]), np.array([3])))
all_vertex_centroids=np.array([M1.mtu.get_average_position([v]) for v in vertices])

# volumes intermediarios por neumann
volumes_in = get_box(vertices, all_vertex_centroids, bvin, False)

# volumes intermediarios por Dirichlet
volumes_id = get_box(vertices, all_vertex_centroids, bvid, False)
intermediarios=rng.unite(volumes_id,volumes_in)

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

# volumes da malha grossa primal 1
meshsets_nv1 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag1]), np.array([None]))

# volumes da malha grossa primal 2
meshsets_nv2 = M1.mb.get_entities_by_type_and_tag(
        M1.root_set, types.MBENTITYSET, np.array([primal_id_tag2]), np.array([None]))


for meshset in meshsets_nv2:
    nc = M1.mb.tag_get_data(primal_id_tag2, meshset, flat=True)[0]
    elems = M1.mb.get_entities_by_handle(meshset)
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))


# Gera a matriz dos coeficientes

internos=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([0]))
faces=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([1]))
arestas=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([2]))
vertices=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([D1_tag]), np.array([3]))
wirebasket_elems_0 = np.array([list(internos) + list(faces) + list(arestas) + list(vertices)])
M1.mb.tag_set_data(fine_to_primal1_classic_tag,vertices,np.arange(0,len(vertices)))

ni=len(internos)
nf=len(faces)
na=len(arestas)
nv=len(vertices)


nni=ni
nnf=nni+nf
nne=nnf+na
nnv=nne+nv
l_elems=[internos,faces,arestas,vertices]
l_ids=[0,nni,nnf,nne,nnv]
for i, elems in enumerate(l_elems):
    M1.mb.tag_set_data(M1.ID_reordenado_tag,elems,np.arange(l_ids[i],l_ids[i+1]))

def add_topology(conj_vols,tag_local,lista):
    all_fac=np.uint64(M1.mtu.get_bridge_adjacencies(conj_vols, 2 ,2))
    all_int_fac=np.uint64([face for face in all_fac if len(M1.mb.get_adjacencies(face, 3))==2])
    adjs=np.array([M1.mb.get_adjacencies(face, 3) for face in all_int_fac])
    adjs1=M1.mb.tag_get_data(tag_local,np.array(adjs[:,0]),flat=True)
    adjs2=M1.mb.tag_get_data(tag_local,np.array(adjs[:,1]),flat=True)
    adjsg1=M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(adjs[:,0]),flat=True)
    adjsg2=M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(adjs[:,1]),flat=True)
    Gids=M1.mb.tag_get_data(M1.ID_reordenado_tag,conj_vols,flat=True)
    lista.append(Gids)
    lista.append(all_int_fac)
    lista.append(adjs1)
    lista.append(adjs2)
    lista.append(adjsg1)
    lista.append(adjsg2)


local_id_int_tag = M1.mb.tag_get_handle("local_id_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
local_id_fac_tag = M1.mb.tag_get_handle("local_fac_internos", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(local_id_int_tag, M1.all_volumes,np.repeat(len(M1.all_volumes)+1,len(M1.all_volumes)))
M1.mb.tag_set_data(local_id_fac_tag, M1.all_volumes,np.repeat(len(M1.all_volumes)+1,len(M1.all_volumes)))
sgids=0
li=[]
ci=[]
di=[]
cont=0
intern_adjs_by_dual=[]
faces_adjs_by_dual=[]
dual_1_meshset=M1.mb.create_meshset()

D_x=max(Lx-int(Lx/l1[0])*l1[0],Lx-int(Lx/l2[0])*l2[0])
D_y=max(Ly-int(Ly/l1[1])*l1[1],Ly-int(Ly/l2[1])*l2[1])
D_z=max(Lz-int(Lz/l1[2])*l1[2],Lz-int(Lz/l2[2])*l2[2])
for i in range(len(lxd1)-1):
    x0=lxd1[i]
    x1=lxd1[i+1]
    box_x=np.array([[x0-0.01,ymin,zmin],[x1+0.01,ymax,zmax]])
    vols_x=get_box(M1.all_volumes, all_centroids, box_x, False)
    x_centroids=np.array([M1.mtu.get_average_position([v]) for v in vols_x])
    for j in range(len(lyd1)-1):
        y0=lyd1[j]
        y1=lyd1[j+1]
        box_y=np.array([[x0-0.01,y0-0.01,zmin],[x1+0.01,y1+0.01,zmax]])
        vols_y=get_box(vols_x, x_centroids, box_y, False)
        y_centroids=np.array([M1.mtu.get_average_position([v]) for v in vols_y])
        for k in range(len(lzd1)-1):
            z0=lzd1[k]
            z1=lzd1[k+1]
            tb=time.time()
            box_dual_1=np.array([[x0-0.01,y0-0.01,z0-0.01],[x1+0.01,y1+0.01,z1+0.01]])
            vols=get_box(vols_y, y_centroids, box_dual_1, False)
            tipo=M1.mb.tag_get_data(D1_tag,vols,flat=True)
            inter=rng.Range(np.array(vols)[np.where(tipo==0)[0]])

            M1.mb.tag_set_data(local_id_int_tag,inter,range(len(inter)))
            add_topology(inter,local_id_int_tag,intern_adjs_by_dual)


            fac=rng.Range(np.array(vols)[np.where(tipo==1)[0]])
            fac_centroids=np.array([M1.mtu.get_average_position([f]) for f in fac])

            box_faces_x=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x0+lx/2,y1+ly/2,z1+lz/2]])
            box_faces_y=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y0+ly/2,z1+lz/2]])
            box_faces_z=np.array([[x0-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z0+lz/2]])

            faces_x=get_box(fac, fac_centroids, box_faces_x, False)

            faces_y=get_box(fac, fac_centroids, box_faces_y, False)
            f1=rng.unite(faces_x,faces_y)

            faces_z=get_box(fac, fac_centroids, box_faces_z, False)
            f1=rng.unite(f1,faces_z)

            if i==len(lxd1)-2:
                box_faces_x2=np.array([[x1-lx/2,y0-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_x2=get_box(fac, fac_centroids, box_faces_x2, False)
                f1=rng.unite(f1,faces_x2)

            if j==len(lyd1)-2:
                box_faces_y2=np.array([[x0-lx/2,y1-ly/2,z0-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_y2=get_box(fac, fac_centroids, box_faces_y2, False)
                f1=rng.unite(f1,faces_y2)

            if k==len(lzd1)-2:
                box_faces_z2=np.array([[x0-lx/2,y0-ly/2,z1-lz/2],[x1+lx/2,y1+ly/2,z1+lz/2]])
                faces_z2=get_box(fac, fac_centroids, box_faces_z2, False)
                f1=rng.unite(f1,faces_z2)

            sgids+=len(f1)
            M1.mb.tag_set_data(local_id_fac_tag,f1,range(len(f1)))
            add_topology(f1,local_id_fac_tag,faces_adjs_by_dual)

print(time.time()-t1,"criou meshset")


t_invaii=time.time()
meshsets_duais=M1.mb.get_child_meshsets(dual_1_meshset)

def solve_block_matrix(topology,pos_0):
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
        ks_all=np.array(M1.mb.tag_get_data(M1.k_eq_tag,np.array(all_faces_topo),flat=True))
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
    return(lgp,cgp,dgp,fl,fc,fd,t_invaii,st,ts,ta,tc)

lgp,cgp,dgp,fl,fc,fd,t_invaii,st,ts,ta,tc=solve_block_matrix(intern_adjs_by_dual,0)
fl=np.concatenate(np.array(fl))
fc=np.concatenate(np.array(fc))
fd=np.concatenate(np.array(fd))

m_loc=csc_matrix((fd,(fl,fc)),shape=(ni,ni))
lgp=np.concatenate(np.array(lgp))
cgp=range(ni)
dgp=np.ones(len(lgp))
permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(ni,ni))
invbAii=permut_g*m_loc*permut_g.transpose()
print("inversão de Aii",time.time()-t_invaii,st,ts,ta,tc)

t_invaff=time.time()
lgp,cgp,dgp,fl,fc,fd,t_invaii,st,ts,ta,tc=solve_block_matrix(faces_adjs_by_dual,ni)
fl=np.concatenate(np.array(fl))
fc=np.concatenate(np.array(fc))
fd=np.concatenate(np.array(fd))

m_loc=csc_matrix((fd,(fl,fc)),shape=(nf,nf))
lgp=np.concatenate(np.array(lgp))
cgp=range(nf)
dgp=np.ones(len(lgp))
permut_g=csc_matrix((dgp,(lgp,cgp)),shape=(nf,nf))

invbAff=permut_g*m_loc*permut_g.transpose()

print("inversão de Aff",time.time()-t_invaff,st,ts,ta,tc)

t0=time.time()
for meshset in meshsets_nv1:
    elems = M1.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, vertices)
    try:
        nc = M1.mb.tag_get_data(fine_to_primal1_classic_tag, vert, flat=True)[0]
    except:
        import pdb; pdb.set_trace()
    M1.mb.tag_set_data(fine_to_primal1_classic_tag, elems, np.repeat(nc, len(elems)))
    M1.mb.tag_set_data(primal_id_tag1, meshset, nc)

first=True
try:
    SOL_ADM_f = np.load('SOL_ADM_fina.npy')
    res=np.load('residuo.npy')
    first=False
    if len(SOL_ADM_f)!=len(M1.all_volumes):
        print("criará o vetor para refinamento")
        SOL_ADM_f = np.repeat(1,len(M1.all_volumes))
    else:
        print("leu o vetor criado")
except:
    print("criará o vetor para refinamento")
    SOL_ADM_f = np.repeat(1,len(M1.all_volumes))
    res = np.repeat(1,len(M1.all_volumes))

gids_p_d=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d,flat=True)
gids_p_n=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n,flat=True)
press_d=SOL_ADM_f[gids_p_d]
press_n=SOL_ADM_f[gids_p_n]

dist_res=np.linalg.norm(bvd-bvn,axis=1).max()
delta_p_res=(abs(sum(press_d)/len(press_d)-sum(press_n)/len(press_n)))
grad_p_res=abs(delta_p_res/dist_res)

med_perm_by_primal_1=[]
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = M1.mb.get_entities_by_handle(m1)
        perm1=M1.mb.tag_get_data(M1.perm_tag,elem_by_L1).reshape([len(elem_by_L1),9])
        med1_x=sum(perm1[:,0])/len(perm1[:,0])
        med_perm_by_primal_1.append(med1_x)
med_perm_by_primal_1=np.sort(med_perm_by_primal_1)
s=1.0    #Parâmetro da secante
lg=np.log(med_perm_by_primal_1)
ordem=11
print("fit")
fit=np.polyfit(range(len(lg)),lg,ordem)
x=sympy.Symbol('x',real=True,positive=True)
func=0
for i in range(ordem+1):
    func+=fit[i]*x**(ordem-i)
print("deriv")
derivada=sympy.diff(func,x)
inc_secante=(lg[-1]-lg[0])/len(lg)
print("solve")
equa=sympy.Eq(derivada,2*inc_secante)
#real_roots=sympy.solve(equa)
ind_inferior=int(sympy.nsolve(equa,0.1*len(lg),verify=False))
if ind_inferior<0:
    ind_inferior=0
ind_superior=int(sympy.nsolve(equa,0.9*len(lg),verify=False))
if ind_superior>len(lg)-1:
    ind_superior=len(lg)-1

new_inc_secante=(lg[ind_superior]-lg[ind_inferior])/(ind_superior-ind_inferior)
eq2=sympy.Eq(derivada,new_inc_secante)
new_ind_inferior=int(sympy.nsolve(eq2,ind_inferior, verify=False))
if new_ind_inferior<ind_inferior:
    new_ind_inferior=ind_inferior
new_ind_superior=int(sympy.nsolve(eq2,ind_superior, verify=False))
if new_ind_superior>ind_superior:
    new_ind_superior=ind_superior

ind_inferior=new_ind_inferior
ind_superior=new_ind_superior
vt=med_perm_by_primal_1[ind_inferior]

if first:
    val_barreira=med_perm_by_primal_1[ind_inferior]
    val_canal=med_perm_by_primal_1[ind_superior]
    raz_lim=100000000000
else:
    val_barreira=min(med_perm_by_primal_1)
    val_canal=max(med_perm_by_primal_1)
    raz_lim=100000000000
all_boundary_faces=M1.all_boundary_faces
def get_max_grad(meshset):
    vols_meshset = M1.mb.get_entities_by_handle(meshset)
    faces_meshset = M1.mtu.get_bridge_adjacencies(vols_meshset,2,2)
    faces_meshset=rng.subtract(faces_meshset,all_boundary_faces)
    adjs=[M1.mb.get_adjacencies(fac,3) for fac in faces_meshset]
    GIDs0=M1.mb.tag_get_data(M1.ID_reordenado_tag,np.uint64(adjs)[:,0])
    GIDs1=M1.mb.tag_get_data(M1.ID_reordenado_tag,np.uint64(adjs)[:,1])
    s0=SOL_ADM_f[GIDs0]
    s1=SOL_ADM_f[GIDs1]
    c0=np.array([M1.mtu.get_average_position([adj]) for adj in np.uint64(adjs)[:,0]])
    c1=np.array([M1.mtu.get_average_position([adj]) for adj in np.uint64(adjs)[:,1]])
    dif=c0-c1
    dist=np.linalg.norm(dif,axis=1)
    grads=np.transpose(abs(s1-s0))[0]/dist
    return(max(grads))

################################################################################
ares_tag=M1.mb.tag_get_handle("ares", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
for m2 in meshset_by_L2:
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        ver_1=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
        k_vert=M1.mb.tag_get_data(M1.perm_tag,ver_1)[:,0]
        ares=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([2]))
        viz_ares=M1.mtu.get_bridge_adjacencies(ares,2,3)
        ares_m=M1.mb.create_meshset()
        M1.mb.add_entities(ares_m,viz_ares)
        viz_ares_ares=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
        ares=viz_ares_ares

        viz_ares=M1.mtu.get_bridge_adjacencies(ares,2,3)
        ares_m=M1.mb.create_meshset()
        M1.mb.add_entities(ares_m,viz_ares)
        viz_ares_ares=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
        viz_ares_ver=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
        viz_ares_fac=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([1]))
        viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_ver)
        viz_ares_ares=rng.unite(viz_ares_ares,ver_1)
        viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_fac)
        ares=viz_ares_ares
        k_ares_max=M1.mb.tag_get_data(M1.perm_tag,ares)[:,0].max()
        k_ares_min=M1.mb.tag_get_data(M1.perm_tag,ares)[:,0].min()
        try:
            r_ver=M1.mb.tag_get_data(ares_tag,ver_1)
            r_k_are_ver=float(max((k_ares_max-k_ares_min)/k_vert,r_ver))
        except:
            r_k_are_ver=float((k_ares_max-k_ares_min)/k_vert)
        M1.mb.tag_set_data(ares_tag, ares, np.repeat(float((k_ares_max-k_ares_min)/k_vert),len(ares)))
        M1.mb.tag_set_data(ares_tag, ver_1,r_k_are_ver)
################################################################################

################################################################################
ares2_tag=M1.mb.tag_get_handle("ares_2", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
max1=0
for m2 in meshset_by_L2:
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        ver_1=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
        k_vert=M1.mb.tag_get_data(M1.perm_tag,ver_1)[:,0]
        facs_ver1=M1.mtu.get_bridge_adjacencies(ver_1,2,2)
        for f in facs_ver1:
            viz_facs=M1.mtu.get_bridge_adjacencies(f,2,3)
            ares_m=M1.mb.create_meshset()
            M1.mb.add_entities(ares_m,viz_facs)
            ares=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
            if len(ares)>0:
                viz_ares=M1.mtu.get_bridge_adjacencies(ares,2,3)
                ares_m=M1.mb.create_meshset()
                M1.mb.add_entities(ares_m,viz_ares)
                ares_com_novas=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                novas_ares=rng.subtract(ares_com_novas,ares)
                ares=rng.unite(ares,novas_ares)
                for i in range(20):
                    try:
                        viz_ares=M1.mtu.get_bridge_adjacencies(novas_ares,2,3)
                    except:
                        import pdb; pdb.set_trace()
                    ares_m=M1.mb.create_meshset()
                    M1.mb.add_entities(ares_m,viz_ares)
                    ares_com_novas=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
                    novas_ares=rng.subtract(ares_com_novas,ares)
                    ares=rng.unite(ares,novas_ares)

                    if len(ares)>max1:
                        print(len(ares),len(novas_ares))
                        max1=len(ares)
                    if len(novas_ares)==0:
                        break
                ares=M1.mtu.get_bridge_adjacencies(ares,2,3)
                ares_m=M1.mb.create_meshset()
                M1.mb.add_entities(ares_m,ares)
                verts=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
                k_ares_max=M1.mb.tag_get_data(M1.perm_tag,ares)[:,0].max()
                k_ares_min=M1.mb.tag_get_data(M1.perm_tag,ares)[:,0].min()
                ver_2=rng.subtract(verts,ver_1)
                try:
                    r_ver=M1.mb.tag_get_data(ares2_tag,ver_1)
                    r_k_are_ver=float(max((k_ares_max-k_ares_min)/k_vert,r_ver))
                except:
                    r_k_are_ver=float((k_ares_max-k_ares_min)/k_vert)
                #M1.mb.tag_set_data(ares2_tag, ares, np.repeat(float((k_ares_max-k_ares_min)/k_vert),len(ares)))
                M1.mb.tag_set_data(ares2_tag, ver_1,r_k_are_ver)
                #k_ver2=M1.mb.tag_get_data(M1.perm_tag,ver_2)[:,0]
                #try:
                #    r_ver=M1.mb.tag_get_data(ares2_tag,ver_2)
                #    r_k_are_ver=float(max((k_ares_max-k_ares_min)/k_ver2,r_ver))
                #except:
                #    r_k_are_ver=float((k_ares_max-k_ares_min)/k_ver2)
                #M1.mb.tag_set_data(ares2_tag, ver_2,r_k_are_ver)


################################################################################
n1=0
n2=0
aux=0
meshset_by_L2 = M1.mb.get_child_meshsets(L2_meshset)
for m2 in meshset_by_L2:
    tem_poço_no_vizinho=False
    meshset_by_L1=M1.mb.get_child_meshsets(m2)
    for m1 in meshset_by_L1:
        elem_by_L1 = M1.mb.get_entities_by_handle(m1)
        ver_1=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
        ver_1=rng.unite(ver_1,M1.mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))

        if ver_1[0] in finos:
            aux=1
            tem_poço_no_vizinho=True
        else:
            viz_vertice=M1.mtu.get_bridge_adjacencies(ver_1,2,3)
            k_vert=M1.mb.tag_get_data(M1.perm_tag,ver_1)[:,0]
            k_viz=M1.mb.tag_get_data(M1.perm_tag,viz_vertice)[:,0]
            raz=float((max(k_viz)-min(k_viz))/k_vert)
            perm1=M1.mb.tag_get_data(M1.perm_tag,elem_by_L1)
            med1_x=sum(perm1[:,0])/len(perm1[:,0])
            med1_y=sum(perm1[:,4])/len(perm1[:,4])
            med1_z=sum(perm1[:,8])/len(perm1[:,8])
            gids_primal=M1.mb.tag_get_data(M1.ID_reordenado_tag,elem_by_L1)
            press_primal=SOL_ADM_f[gids_primal]
            ares=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([2]))

            viz_ares=M1.mtu.get_bridge_adjacencies(ares,2,3)
            ares_m=M1.mb.create_meshset()
            M1.mb.add_entities(ares_m,viz_ares)
            viz_ares_ares=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
            ares=viz_ares_ares

            viz_ares=M1.mtu.get_bridge_adjacencies(ares,2,3)
            ares_m=M1.mb.create_meshset()
            M1.mb.add_entities(ares_m,viz_ares)
            viz_ares_ares=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([2]))
            viz_ares_ver=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([3]))
            viz_ares_fac=M1.mb.get_entities_by_type_and_tag(ares_m, types.MBHEX, np.array([D1_tag]), np.array([1]))
            viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_ver)
            viz_ares_ares=rng.unite(viz_ares_ares,ver_1)
            viz_ares_ares=rng.unite(viz_ares_ares,viz_ares_fac)
            ares=viz_ares_ares

            k_ares_max=M1.mb.tag_get_data(M1.perm_tag,ares)[:,0].max()
            k_ares_min=M1.mb.tag_get_data(M1.perm_tag,ares)[:,0].min()
            r_k_are_ver=float((k_ares_max-k_ares_min)/k_vert)
            #M1.mb.tag_set_data(ares_tag, ares, np.repeat(r_k_are_ver,len(ares)))
            r_k_are_ver=float(M1.mb.tag_get_data(ares2_tag,ver_1))
            if first:
                max_grad=0
            else:
                max_grad=get_max_grad(m1)
            if max_grad>100*grad_p_res or med1_x<val_barreira or med1_x>val_canal or raz>raz_lim or (max_grad>10*grad_p_res and (r_k_are_ver>40 or k_ares_max/k_vert>100 or k_vert/k_ares_min>100 or k_vert<vt)) or r_k_are_ver>2000 or (max_grad>5*grad_p_res and r_k_are_ver>100) or (max_grad>1*grad_p_res and r_k_are_ver>100):
                aux=1
                tem_poço_no_vizinho=True
        if ver_1[0] in intermediarios:
            tem_poço_no_vizinho=True
        if aux==1:
            aux=0
            for elem in elem_by_L1:
                n1+=1
                n2+=1
                M1.mb.tag_set_data(L1_ID_tag, elem, n1)
                M1.mb.tag_set_data(L2_ID_tag, elem, n2)
                M1.mb.tag_set_data(L3_ID_tag, elem, 1)
                finos.append(elem)
    if tem_poço_no_vizinho==False:
        elem_by_L2 = M1.mb.get_entities_by_handle(m2)
        perm2=M1.mb.tag_get_data(M1.perm_tag,elem_by_L2).reshape([len(elem_by_L2),9])

        med2_x=sum(perm2[:,0])/len(perm2[:,0])
        var=sum((x - med2_x)**2 for x in perm2[:,0])/len(perm2[:,0])
        desv_pad_x2=sqrt(var)

        med2_y=sum(perm2[:,4])/len(perm2[:,4])
        var=sum((x - med2_y)**2 for x in perm2[:,4])/len(perm2[:,4])
        desv_pad_y2=sqrt(var)

        med2_z=sum(perm2[:,8])/len(perm2[:,8])
        var=sum((x - med2_z)**2 for x in perm2[:,8])/len(perm2[:,8])
        desv_pad_z2=sqrt(var)

        ref=max([desv_pad_x2/med2_x,desv_pad_y2/med2_y,desv_pad_z2/med2_z])
        desv_pad=max([desv_pad_x2, desv_pad_y2,desv_pad_z2])


        desv_pad=sqrt(var)
        #print("Desvio padrão",desv_pad)
        if desv_pad>50000000 or ref>99993.6:
            tem_poço_no_vizinho=True
    if tem_poço_no_vizinho:
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            n2+=1
            t=1
            ver_1=M1.mb.get_entities_by_type_and_tag(m1, types.MBHEX, np.array([D1_tag]), np.array([3]))
            ver_1=rng.unite(ver_1,M1.mb.get_entities_by_type_and_tag(m1, types.MBTET, np.array([D1_tag]), np.array([3])))
            if ver_1[0] not in finos:
                M1.mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
                M1.mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
                M1.mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(2,len(elem_by_L1)))
                t=0
            n1-=t
            n2-=t
    else:
        n2+=1
        for m1 in meshset_by_L1:
            elem_by_L1 = M1.mb.get_entities_by_handle(m1)
            n1+=1
            M1.mb.tag_set_data(L2_ID_tag, elem_by_L1, np.repeat(n2,len(elem_by_L1)))
            M1.mb.tag_set_data(L1_ID_tag, elem_by_L1, np.repeat(n1,len(elem_by_L1)))
            M1.mb.tag_set_data(L3_ID_tag, elem_by_L1, np.repeat(3,len(elem_by_L1)))

# ------------------------------------------------------------------------------
print('Definição da malha ADM: ',time.time()-t0)
t0=time.time()


av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)
#M1.mb.write_file('teste_3D_unstructured_18.vtk',[av])
#print("new file!!")


# fazendo os ids comecarem de 0 em todos os niveis

tags = [L1_ID_tag, L2_ID_tag]
for tag in tags:
    all_gids = M1.mb.tag_get_data(tag, M1.all_volumes, flat=True)
    minim = min(all_gids)
    all_gids -= minim
    M1.mb.tag_set_data(tag, M1.all_volumes, all_gids)

ln=[]
cn=[]
dn=[]

lines=[]
cols=[]
data=[]

IDs_globais_d=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d,flat=True)
lines_d=IDs_globais_d
cols_d=np.zeros((1,len(lines_d)),dtype=np.int32)[0]
# data_d=np.repeat(press,len(lines_d))
data_d=M1.mb.tag_get_data(M1.press_value_tag, volumes_d, flat=True)
#for d in volumes_d:
#    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,d))
#    lines.append(ID_global)
#    cols.append(0)
#    data.append(press)


IDs_globais_n=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n,flat=True)
lines_n=IDs_globais_n
cols_n=np.zeros((1,len(lines_n)),dtype=np.int32)[0]
# data_n=np.repeat(vazao,len(lines_n))
data_n=M1.mb.tag_get_data(M1.press_value_tag, volumes_n, flat=True)
    #b[ID_global]=press
#for n in volumes_n:
#    ID_global=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,n))
#    lines.append(ID_global)
#    cols.append(0)
#    data.append(vazao)

    #b[ID_global]=vazao
lines=np.concatenate([lines_d,lines_n])
cols=np.concatenate([cols_d,cols_n])
data=np.concatenate([data_d,data_n])

b=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),1))
# b = b2.copy()

all_intern_faces=[face for face in M1.all_faces if len(M1.mb.get_adjacencies(face, 3))==2]
all_intern_adjacencies=np.array([M1.mb.get_adjacencies(face, 3) for face in all_intern_faces])
all_adjacent_volumes=[]
all_adjacent_volumes.append(M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(all_intern_adjacencies[:,0]),flat=True))
all_adjacent_volumes.append(M1.mb.tag_get_data(M1.ID_reordenado_tag,np.array(all_intern_adjacencies[:,1]),flat=True))

print("TEMPO TOTAL DE PRÉ PROCESSAMENTO:",time.time()-tempo0_pre)
print(" ")
print(n1,n2)
print("  ")
print("INICIOU SOLUÇÃO ADM")
#M1.mb.write_file('teste_3D_unstructured_18_2.vtk')

tempo0_ADM=time.time()
lines=[]
data=[]
cols=[]

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



#ADJs=np.array([M1.mb.get_adjacencies(face, 3) for face in M1.all_faces])
ADJs1=all_adjacent_volumes[0]
ADJs2=all_adjacent_volumes[1]
ks=M1.mb.tag_get_data(M1.k_eq_tag,all_intern_faces,flat=True)
c2=0
cont=0
for f in all_intern_faces:
    k_eq=ks[cont]
    #Gid_1=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[0]))
    #Gid_2=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[1]))
    Gid_1=ADJs1[c2]
    Gid_2=ADJs2[c2]
    #print(Gid_1,Gid_2,ADJs1[c2],ADJs2[c2])
    c2+=1

    #lines.append(Gid_1)
    #cols.append(Gid_2)
    #data.append(k_eq)
    ##T[Gid_1][Gid_2]=1
    #lines.append(Gid_2)
    #cols.append(Gid_1)
    #data.append(k_eq)
    ##T[Gid_2][Gid_1]=1
    #lines.append(Gid_1)
    #cols.append(Gid_1)
    #data.append(-k_eq)
    ##T[Gid_1][Gid_1]-=1
    #lines.append(Gid_2)
    #cols.append(Gid_2)
    #data.append(-k_eq)
    ##T[Gid_2][Gid_2]-=1

    if Gid_1<ni and Gid_2<ni:
        lii.append(Gid_1)
        cii.append(Gid_2)
        dii.append(k_eq)

        lii.append(Gid_2)
        cii.append(Gid_1)
        dii.append(k_eq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-k_eq)

        lii.append(Gid_2)
        cii.append(Gid_2)
        dii.append(-k_eq)

    elif Gid_1<ni and Gid_2>=ni and Gid_2<ni+nf:
        lif.append(Gid_1)
        cif.append(Gid_2-ni)
        dif.append(k_eq)

        lii.append(Gid_1)
        cii.append(Gid_1)
        dii.append(-k_eq)

    elif Gid_2<ni and Gid_1>=ni and Gid_1<ni+nf:
        lif.append(Gid_2)
        cif.append(Gid_1-ni)
        dif.append(k_eq)

        lii.append(Gid_2)
        cii.append(Gid_2)
        dii.append(-k_eq)

    elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni and Gid_2<ni+nf:
        lff.append(Gid_1-ni)
        cff.append(Gid_2-ni)
        dff.append(k_eq)

        lff.append(Gid_2-ni)
        cff.append(Gid_1-ni)
        dff.append(k_eq)

        lff.append(Gid_1-ni)
        cff.append(Gid_1-ni)
        dff.append(-k_eq)

        lff.append(Gid_2-ni)
        cff.append(Gid_2-ni)
        dff.append(-k_eq)

    elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni+nf and Gid_2<ni+nf+na:
        lfe.append(Gid_1-ni)
        cfe.append(Gid_2-ni-nf)
        dfe.append(k_eq)

        lff.append(Gid_1-ni)
        cff.append(Gid_1-ni)
        dff.append(-k_eq)

    elif Gid_2>=ni and Gid_2<ni+nf and Gid_1>=ni+nf and Gid_1<ni+nf+na:
        lfe.append(Gid_2-ni)
        cfe.append(Gid_1-ni-nf)
        dfe.append(k_eq)

        lff.append(Gid_2-ni)
        cff.append(Gid_2-ni)
        dff.append(-k_eq)

    elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf and Gid_2<ni+nf+na:
        lee.append(Gid_1-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(k_eq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(k_eq)

        lee.append(Gid_1-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(-k_eq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(-k_eq)

    elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf+na:
        lev.append(Gid_1-ni-nf)
        cev.append(Gid_2-ni-nf-na)
        dev.append(k_eq)

        lee.append(Gid_1-ni-nf)
        cee.append(Gid_1-ni-nf)
        dee.append(-k_eq)

    elif Gid_2>=ni+nf and Gid_2<ni+nf+na and Gid_1>=ni+nf+na:
        lev.append(Gid_2-ni-nf)
        cev.append(Gid_1-ni-nf-na)
        dev.append(k_eq)

        lee.append(Gid_2-ni-nf)
        cee.append(Gid_2-ni-nf)
        dee.append(-k_eq)

    elif Gid_1>=ni+nf+na and Gid_2>=ni+nf+na:
        lvv.append(Gid_1)
        cvv.append(Gid_2)
        dvv.append(k_eq)

        lvv.append(Gid_2)
        cvv.append(Gid_1)
        dvv.append(k_eq)

        lvv.append(Gid_1)
        cvv.append(Gid_1)
        dvv.append(-k_eq)

        lvv.append(Gid_2)
        cvv.append(Gid_2)
        dvv.append(-k_eq)
    cont+=1

Gid_1=ADJs1
Gid_2=ADJs2

lines=Gid_1
cols=Gid_2
data=ks
#T[Gid_1][Gid_2]=1
lines=np.concatenate([lines,Gid_2])
cols=np.concatenate([cols,Gid_1])
data=np.concatenate([data,ks])
#T[Gid_2][Gid_1]=1
lines=np.concatenate([lines,Gid_1])
cols=np.concatenate([cols,Gid_1])
data=np.concatenate([data,-ks])
#T[Gid_1][Gid_1]-=1
lines=np.concatenate([lines,Gid_2])
cols=np.concatenate([cols,Gid_2])
data=np.concatenate([data,-ks])
#T[Gid_2][Gid_2]-=1



T=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
# Tfina = T.copy()
Aii=csc_matrix((dii,(lii,cii)),shape=(ni,ni))
Aif=csc_matrix((dif,(lif,cif)),shape=(ni,nf))
Aff=csc_matrix((dff,(lff,cff)),shape=(nf,nf))
Afe=csc_matrix((dfe,(lfe,cfe)),shape=(nf,na))
Aee=csc_matrix((dee,(lee,cee)),shape=(na,na))
Aev=csc_matrix((dev,(lev,cev)),shape=(na,nv))
Avv=csc_matrix((dvv,(lvv,cvv)),shape=(nv,nv))

print("def as",time.time()-tempo0_ADM)
Ivv=scipy.sparse.identity(nv)
ty=time.time()

'''
#############
# obter termo fonte de gravidade
definter.cent(M1.mb, M1.mtu, all_volumes)
cent_tag = M1.mb.tag_get_handle('CENT')
boundary_faces = M1.all_boundary_faces
faces_in = rng.subtract(M1.all_faces, boundary_faces)
ids_volumes_tag = M1.mb.tag_get_handle('IDS_VOLUMES', 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(ids_volumes_tag, all_volumes, np.arange(len(all_volumes)))
gamaf_tag = M1.mb.tag_get_handle('GAMAF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
M1.mb.tag_set_data(gamaf_tag, M1.all_faces, np.repeat(float(M1.gama), len(M1.all_faces)))
sgravf_tag = M1.mb.tag_get_handle('SGRAVF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Adjs = np.array([np.array(M1.mb.get_adjacencies(f, 3)) for f in faces_in])
all_centroids = M1.mb.tag_get_data(cent_tag, all_volumes)
ids_0 = np.array([M1.mb.tag_get_data(ids_volumes_tag, int(elem), flat=True)[0] for elem in Adjs[:,0]])
ids_1 = np.array([M1.mb.tag_get_data(ids_volumes_tag, int(elem), flat=True)[0] for elem in Adjs[:,1]])
s_gravsf = definter.set_s_grav_faces(M1.mb, M1.k_eq_tag, all_volumes, all_centroids, faces_in, gamaf_tag, sgravf_tag, ids_0, ids_1)
idsrr0 = M1.mb.tag_get_data(M1.ID_reordenado_tag, np.array(Adjs[:,0]), flat=True)
idsrr1 = M1.mb.tag_get_data(M1.ID_reordenado_tag, np.array(Adjs[:,1]), flat=True)
ids_reord_elems0 = idsrr0
ids_reord_elems1 = idsrr1
# ids_reord_elems0 = np.array([M1.mb.tag_get_data(M1.ID_reordenado_tag, int(elem), flat=True)[0] for elem in Adjs[:,0]])
# ids_reord_elems1 = np.array([M1.mb.tag_get_data(M1.ID_reordenado_tag, int(elem), flat=True)[0] for elem in Adjs[:,1]])
fonte_grav = np.zeros(len(all_volumes))
fonte_grav[ids_reord_elems0] -= s_gravsf
fonte_grav[ids_reord_elems1] += s_gravsf
b2 = fonte_grav.copy()

ids_volumes_d = M1.mb.tag_get_data(M1.ID_reordenado_tag, volumes_d, flat=True)
values_d = M1.mb.tag_get_data(M1.press_value_tag, volumes_d, flat=True)
Tfina = Tfina.tolil()
Tfina[ids_volumes_d] = sp.lil_matrix((len(ids_volumes_d), Tfina.shape[0]))
Tfina[ids_volumes_d, ids_volumes_d] = np.ones(len(ids_volumes_d))
b2[ids_volumes_d] = values_d

ids_volumes_n = M1.mb.tag_get_data(M1.ID_reordenado_tag, volumes_n, flat=True)
# values_n = M1.mb.tag_get_data(M1.vazao_value_tag, volumes_n, flat=True)
values_n = M1.mb.tag_get_data(M1.press_value_tag, volumes_n, flat=True)
Tfina[ids_volumes_n] = sp.lil_matrix((len(ids_volumes_n), Tfina.shape[0]))
Tfina[ids_volumes_n, ids_volumes_n] = np.ones(len(ids_volumes_n))
b2[ids_volumes_n] = values_n

pf2_tag = M1.mb.tag_get_handle('Pfino2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
pf2 = linalg.spsolve(Tfina.tocsc(), b2)
M1.mb.tag_set_data(pf2_tag, wirebasket_elems_0, pf2)

# b = b2.copy()
'''
#th=time.time()
#M2=-linalg.inv(Aee)*Aev
#print(time.time()-th,"Direto")
'''
invAee=lu_inv2(Aee)
M2=-invAee*Aev
del(invAee)
P=vstack([M2,Ivv]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

invAff=lu_inv2(Aff)
M3=-invAff*Afe*M2
del(M2)
del(invAff)
P=vstack([M3,P])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)
invAii=lu_inv2(Aii)
P=vstack([-invAii*Aif*M3,P]) ##P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)
del(M3)
'''

ta1=time.time()

arestas_meshset=M1.mb.create_meshset()
M1.mb.add_entities(arestas_meshset,arestas)
faces_meshset=M1.mb.create_meshset()
M1.mb.add_entities(faces_meshset,faces)
internos_meshset=M1.mb.create_meshset()
M1.mb.add_entities(internos_meshset,internos)

nivel_0_arestas=M1.mb.get_entities_by_type_and_tag(arestas_meshset, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_0_faces=M1.mb.get_entities_by_type_and_tag(faces_meshset, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_0_internos=M1.mb.get_entities_by_type_and_tag(internos_meshset, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))

IDs_arestas_0=M1.mb.tag_get_data(M1.ID_reordenado_tag,nivel_0_arestas,flat=True)
IDs_faces_0=M1.mb.tag_get_data(M1.ID_reordenado_tag,nivel_0_faces,flat=True)
IDs_internos_0=M1.mb.tag_get_data(M1.ID_reordenado_tag,nivel_0_internos,flat=True)

IDs_arestas_0_locais=np.subtract(IDs_arestas_0,ni+nf)
IDs_faces_0_locais=np.subtract(IDs_faces_0,ni)
IDs_internos_0_locais=IDs_internos_0

IDs_arestas_1_locais=np.setdiff1d(range(na),IDs_arestas_0_locais)

IDs_faces_1_locais=np.setdiff1d(range(nf),IDs_faces_0_locais)
IDs_internos_1_locais=np.setdiff1d(range(ni),IDs_internos_0_locais)

ids_arestas=np.where(Aev.sum(axis=1)==0)[0]
ids_arestas_slin_m0=np.setdiff1d(range(na),ids_arestas)

ids_faces=np.where(Afe.sum(axis=1)==0)[0]
ids_faces_slin_m0=np.setdiff1d(range(nf),ids_faces)

ids_internos=np.where(Aif.sum(axis=1)==0)[0]
ids_internos_slin_m0=np.setdiff1d(range(ni),ids_internos)

invAee=lu_inv4(Aee,ids_arestas_slin_m0)
M2=-invAee*Aev
PAD=vstack([M2,Ivv]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

#invAff=lu_inv4(Aff,ids_faces_slin_m0)
invAff=invbAff
M3=-invAff*(Afe*M2)

#ids_internos_slin_m0=np.setdiff1d(ids_internos_slin_m0,IDs_internos_0_locais)
PAD=vstack([M3,PAD])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)
#invAii=lu_inv4(Aii,ids_internos_slin_m0)
invAii=invbAii
PAD=vstack([-invAii*(Aif*M3),PAD]) ##P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)
print("get_OP_AMS", time.time()-ta1)

del(M3)

ids_1=M1.mb.tag_get_data(L1_ID_tag,vertices,flat=True)
ids_class=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
t0=time.time()

AMS_TO_ADM=dict(zip(ids_class,ids_1))
ty=time.time()
vm=M1.mb.create_meshset()
M1.mb.add_entities(vm,vertices)

tm=time.time()
PAD=csc_matrix(PAD)
OP3=PAD.copy()
nivel_0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
tor=time.time()
ID_global1=M1.mb.tag_get_data(M1.ID_reordenado_tag,nivel_0, flat=True)
IDs_ADM1=M1.mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
IDs_AMS1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,nivel_0, flat=True)
OP3[ID_global1]=csc_matrix((1,OP3.shape[1]))
IDs_ADM1=M1.mb.tag_get_data(L1_ID_tag,nivel_0, flat=True)
IDs_ADM_1=M1.mb.tag_get_data(L1_ID_tag,vertices, flat=True)
IDs_AMS_1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices, flat=True)
lp=IDs_AMS_1
cp=IDs_ADM_1
dp=np.repeat(1,len(lp))
permut=csc_matrix((dp,(lp,cp)),shape=(len(vertices),n1))
opad3=OP3*permut
m=find(opad3)
l1=m[0]
c1=m[1]
d1=m[2]
l1=np.concatenate([l1,ID_global1])
c1=np.concatenate([c1,IDs_ADM1])
d1=np.concatenate([d1,np.ones(len(nivel_0))])
opad3=csc_matrix((d1,(l1,c1)),shape=(len(M1.all_volumes),n1))
print("opad1",tor-time.time(),time.time()-ta1, time.time()-tempo0_ADM)
OP_ADM=csc_matrix(opad3)

print("obteve OP_ADM_1",time.time()-tempo0_ADM)

l1=M1.mb.tag_get_data(L1_ID_tag, M1.all_volumes, flat=True)
c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
d1=np.ones((1,len(l1)),dtype=np.int)[0]
OR_ADM=csc_matrix((d1,(l1,c1)),shape=(n1,len(M1.all_volumes)))

l1=M1.mb.tag_get_data(fine_to_primal1_classic_tag, M1.all_volumes, flat=True)
c1=M1.mb.tag_get_data(M1.ID_reordenado_tag, M1.all_volumes, flat=True)
d1=np.ones((1,len(l1)),dtype=np.int)[0]
OR_AMS=csc_matrix((d1,(l1,c1)),shape=(nv,len(M1.all_volumes)))

OP_AMS=PAD

v=M1.mb.create_meshset()
M1.mb.add_entities(v,vertices)

inte=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
fac=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
are=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
ver=M1.mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

M1.mb.tag_set_data(fine_to_primal2_classic_tag, ver, np.arange(len(ver)))

for meshset in meshsets_nv2: #print(rng.intersect(M1.mb.get_entities_by_handle(meshset), ver))
    elems = M1.mb.get_entities_by_handle(meshset)
    vert = rng.intersect(elems, ver)
    try:
        nc = M1.mb.tag_get_data(fine_to_primal2_classic_tag, vert, flat=True)[0]
    except:
        import pdb; pdb.set_trace()
    M1.mb.tag_set_data(fine_to_primal2_classic_tag, elems, np.repeat(nc, len(elems)))
    M1.mb.tag_set_data(primal_id_tag2, meshset, nc)

lines=[]
cols=[]
data=[]

nint=len(inte)
nfac=len(fac)
nare=len(are)
nver=len(ver)
tu=time.time()
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

T_AMS=OR_AMS*T*OP_AMS
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
if MPFA_NO_NIVEL_2==False:
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
invAee=lu_inv2(Aee)
M2=-csc_matrix(invAee)*Aev
P2=vstack([M2,Ivv])

invAff=lu_inv2(Aff)
if MPFA_NO_NIVEL_2:
    M3=-invAff*Afe*M2-invAff*Afv
    P2=vstack([M3,P2])
else:
    Mf=-invAff*Afe*M2
    P2=vstack([Mf,P2])
invAii=lu_inv2(Aii)
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


ID_AMS_1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices,flat=True)
ID_AMS_2=M1.mb.tag_get_data(fine_to_primal2_classic_tag,vertices,flat=True)

OR_AMS_2=csc_matrix((np.repeat(1,len(vertices)),(ID_AMS_2,ID_AMS_1)),shape=(len(ver),len(vertices)))
T_AMS_2=OR_AMS_2*T_AMS*OP_AMS_2

P2=P2.toarray()
#OP_ADM_2=np.zeros((len(T_ADM),n2))
'''
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
'''
#for i in range(P2.shape[0]): print(len(np.where(p2[0]==i)[0]))
#####################################################

lines=[]
cols=[]
data=[]
P2=OP_AMS_2
'''
vm=M1.mb.create_meshset()
M1.mb.add_entities(vm,vertices)
for i in range(len(ver)):
    OP_ams2_tag=M1.mb.tag_get_handle("OP_ams2_tag_"+str(i), 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    vals=OP_AMS_2[:,i].toarray()
    M1.mb.tag_set_data(OP_ams2_tag,vertices,vals)
M1.mb.write_file('delete_me.vtk',[vm])
'''

##################################################
tu=time.time()
PAD=csc_matrix(P2)
OP3=PAD.copy()

m_vert=M1.mb.create_meshset()
M1.mb.add_entities(m_vert,vertices)
nivel_0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_1=M1.mb.get_entities_by_type_and_tag(m_vert, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
nivel_0_e_1=rng.unite(nivel_0,nivel_1)

nivel_0v=M1.mb.get_entities_by_type_and_tag(m_vert, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_0_e_1_v=rng.unite(nivel_0v,nivel_1)
IDs_AMS1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,nivel_0_e_1_v, flat=True)
OP3[IDs_AMS1]=csc_matrix((1,OP3.shape[1]))

IDs_ADM_2=M1.mb.tag_get_data(L2_ID_tag,ver, flat=True)
IDs_AMS_2=M1.mb.tag_get_data(fine_to_primal2_classic_tag,ver, flat=True)
lp=IDs_AMS_2
cp=IDs_ADM_2
dp=np.repeat(1,len(lp))
permutc=csc_matrix((dp,(lp,cp)),shape=(len(ver),n2))
opad3=OP3*permutc

IDs_ADM_1=M1.mb.tag_get_data(L1_ID_tag,vertices, flat=True)
IDs_AMS_1=M1.mb.tag_get_data(fine_to_primal1_classic_tag,vertices, flat=True)

lp=IDs_ADM_1
cp=IDs_AMS_1
dp=np.repeat(1,len(lp))
permutl=csc_matrix((dp,(lp,cp)),shape=(n1,len(vertices)))
opad3=permutl*opad3

m=find(opad3)
l1=m[0]
c1=m[1]
d1=m[2]

ID_global1=M1.mb.tag_get_data(L1_ID_tag,nivel_0_e_1, flat=True)
IDs_ADM1=M1.mb.tag_get_data(L2_ID_tag,nivel_0_e_1, flat=True)

l1=np.concatenate([l1,ID_global1])
c1=np.concatenate([c1,IDs_ADM1])
d1=np.concatenate([d1,np.ones(len(nivel_0_e_1))])
opad3=csc_matrix((d1,(l1,c1)),shape=(n1,n2))
####################################################
'''
P2=csc_matrix(P2)

matriz=scipy.sparse.find(P2)
LIN=matriz[0]
COL=matriz[1]
DAT=matriz[2]
del(matriz)
IDs_ADM_1=M1.mb.tag_get_data(L1_ID_tag,nivel_1,flat=True)
IDs_ADM_2=M1.mb.tag_get_data(L2_ID_tag,nivel_1,flat=True)
IDs_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag, nivel_1,flat=True)
dd=np.array([])
for i in range(len(nivel_1)):
    ID_AMS = IDs_AMS[i]
    dd=np.concatenate([dd,np.array(np.where(LIN==ID_AMS))[0]])
LIN=np.delete(LIN,dd,axis=0)
COL=np.delete(COL,dd,axis=0)
DAT=np.delete(DAT,dd,axis=0)
lines=IDs_ADM_1
cols=IDs_ADM_2
data=np.ones((1,len(lines)),dtype=np.int32)[0]
IDs_ADM_1=M1.mb.tag_get_data(L1_ID_tag,nivel_0,flat=True)
IDs_ADM_2=M1.mb.tag_get_data(L2_ID_tag,nivel_0,flat=True)
IDs_AMS=M1.mb.tag_get_data(fine_to_primal1_classic_tag, nivel_0,flat=True)
tu=time.time()
dd=np.array([])
for i in range(len(nivel_0)):
    ID_AMS = IDs_AMS[i]
    dd=np.concatenate([dd,np.array(np.where(LIN==ID_AMS))[0]])
    dd=np.unique(dd)
LIN=np.delete(LIN,dd,axis=0)
COL=np.delete(COL,dd,axis=0)
DAT=np.delete(DAT,dd,axis=0)
print("fss",time.time()-tu)
lines=np.concatenate([lines,IDs_ADM_1])
cols=np.concatenate([cols,IDs_ADM_2])
data=np.concatenate([data,np.ones((1,len(IDs_ADM_1)),dtype=np.int32)[0]])
LIN_ADM=[AMS_TO_ADM[k] for k in LIN]
COL_ADM=[COL_TO_ADM_2[str(k)] for k in COL]
lines=np.concatenate([lines,LIN_ADM])
cols=np.concatenate([cols,COL_ADM])
data=np.concatenate([data,DAT])
OP_ADM_2=csc_matrix((data,(lines,cols)),shape=(n1,n2))'''
print("fss",time.time()-tu)
OP_ADM_2=opad3
##########################################
'''OP2=P2.copy()
ta2=time.time()
vm=M1.mb.create_meshset()
M1.mb.add_entities(vm,vertices)
v_nivel_0=M1.mb.get_entities_by_type_and_tag(vm, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
v_nivel_1=M1.mb.get_entities_by_type_and_tag(vm, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
v_n0e1=rng.unite(v_nivel_0,v_nivel_1)
ID_class=M1.mb.tag_get_data(fine_to_primal1_classic_tag,v_n0e1, flat=True)
OP2[ID_class]=csc_matrix((1,OP2.shape[1]))

nivel_0=M1.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([L3_ID_tag]), np.array([1]))
nivel_1=M1.mb.get_entities_by_type_and_tag(vm, types.MBHEX, np.array([L3_ID_tag]), np.array([2]))
n0e1=rng.unite(nivel_0,nivel_1)
IDs_ADM1=M1.mb.tag_get_data(L1_ID_tag,n0e1, flat=True)
IDs_ADM2=M1.mb.tag_get_data(L2_ID_tag,n0e1, flat=True)

lines=IDs_ADM1
cols=IDs_ADM2
data=np.repeat(1,len(lines))
IDs_AMS2=M1.mb.tag_get_data(fine_to_primal2_classic_tag,ver, flat=True)
IDs_ADM2_ver=M1.mb.tag_get_data(L2_ID_tag,ver, flat=True)
AMS2_TO_ADM2=dict(zip(IDs_AMS2,IDs_ADM2_ver))

m=find(OP2)
l1=m[0]
c1=m[1]
d1=m[2]
ID_ADM2=[AMS2_TO_ADM2[k] for k in c1]
lines=np.concatenate([lines,l1])
cols=np.concatenate([cols,ID_ADM2])
data=np.concatenate([data,d1])

opad2=csc_matrix((data,(lines,cols)),shape=(n1,n2))
print("opad2",time.time()-ta2)

#OP_ADM_2=opad2'''

###################################
l2=M1.mb.tag_get_data(L2_ID_tag, M1.all_volumes, flat=True)
c2=M1.mb.tag_get_data(L1_ID_tag, M1.all_volumes, flat=True)
d2=np.ones((1,len(l2)),dtype=np.int)[0]
OR_ADM_2=csc_matrix((d2,(l2,c2)),shape=(n2,n1))

r2=find(OR_ADM_2)
lin=r2[0]
col=r2[1]
dat=np.ones((1,len(lin)),dtype=np.int)[0]
'''
dat=r2[2]
for i in range(len(dat)):
    if dat[i]>1:
        dat[i]=1
'''
OR_ADM_2=csc_matrix((dat,(lin,col)),shape=(n2,n1))


ID_global=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_d, flat=True)
ID_ADM=int(M1.mb.tag_get_data(L1_ID_tag,v))
ID_ADM_2=M1.mb.tag_get_data(L2_ID_tag,volumes_d, flat=True)
T[ID_global]=scipy.sparse.csc_matrix((len(ID_global),T.shape[0]))
T[ID_global,ID_global]=np.ones(len(ID_global))


########################## apagar para usar pressão-vazão
ID_globaln=M1.mb.tag_get_data(M1.ID_reordenado_tag,volumes_n, flat=True)
T[ID_globaln]=scipy.sparse.csc_matrix((len(ID_globaln),T.shape[0]))
T[ID_globaln,ID_globaln]=np.ones(len(ID_globaln))
########################## fim de apagar
b_2=OR_ADM_2*OR_ADM*b
T_2=OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2
try:
    GIDs_vertices=M1.mb.tag_get_data(M1.ID_reordenado_tag,vertices,flat=True)
    nv2_IDs_vertices=M1.mb.tag_get_data(L2_ID_tag,vertices,flat=True)
    nivel_id_vertices=M1.mb.tag_get_data(L3_ID_tag,vertices,flat=True)
    inds_vertices_nv2=np.where(nivel_id_vertices==2)[0]
    #GIDs_dir=GIDs_vertices[GIDs_vertices]
    #T_2[GIDs_dir]=scipy.sparse.csc_matrix((len(GIDs_dir),T_2.shape[0]))
    #T_2[GIDs_dir,GIDs_dir]=np.ones(len(GIDs_dir))
    press_dir=SOL_ADM_f[GIDs_vertices]
    #b_2[GIDs_dir]=csc_matrix((press_dir)).transpose()
except:
    print("Mesmo T")
    import pdb; pdb.set_trace()

t0=time.time()
#SOL_ADM_2=linalg.spsolve(T_2,b_2)
SOL_ADM_2=linalg.spsolve(OR_ADM_2*OR_ADM*T*OP_ADM*OP_ADM_2,OR_ADM_2*OR_ADM*b) #+OR_ADM_2*T1*corr_adm2_sd    -OR_ADM_2*T1*corr_adm2_sd
#SOL_ADM_2[nv2_IDs_vertices]=press_dir


SOL_ADM_fina=OP_ADM*OP_ADM_2*SOL_ADM_2#+OP_ADM*corr_adm2_sd#.transpose().toarray()[0] #+corr_adm1_sd.transpose().toarray()[0]
print("Solução do sistema ADM",time.time()-t0)
if first:
    np.save('SOL_ADM_fina.npy',x1)
#teste=OP_ADM*(-OP_ADM_2*OR_ADM_2*T1*corr_adm2_sd+corr_adm2_sd)

print("TEMPO TOTAL PARA SOLUÇÃO ADM:", time.time()-tempo0_ADM)
print("")
#M1.mb.write_file('teste_3D_unstructured_18_2.vtk',[av])
Sol_ADM_tag=M1.mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

perm_xx_tag=M1.mb.tag_get_handle("Perm_xx", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
perm_zz_tag=M1.mb.tag_get_handle("Perm_zz", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
GIDs=M1.mb.tag_get_data(M1.ID_reordenado_tag,M1.all_volumes,flat=True)
perms_xx=M1.mb.tag_get_data(M1.perm_tag,M1.all_volumes)[:,0]
perms_zz=M1.mb.tag_get_data(M1.perm_tag,M1.all_volumes)[:,8]
residuo_tag=M1.mb.tag_get_handle("Residuo ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
res=T*(SOL_ADM_fina.transpose()-b.transpose()).transpose()
res=abs(np.array(res))
np.save('residuo.npy',res)
cont=0
for v in M1.all_volumes:
    gid=GIDs[cont]
    M1.mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])
    M1.mb.tag_set_data(perm_xx_tag,v,perms_xx[cont])
    M1.mb.tag_set_data(perm_zz_tag,v,perms_zz[cont])
    M1.mb.tag_set_data(residuo_tag,v,res[gid])
    cont+=1
'''
av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)

M1.mb.write_file('teste_3D_unstructured_18_2.vtk',[av])
print('New file created')
import pdb; pdb.set_trace()'''
'''##############  APENAS PARA CÁLCULO DO TEMPO DE ASSEMBLY ##################'''
t_ass=time.time()

#T=np.zeros((len(M1.all_volumes),len(M1.all_volumes)))
lines=[]
cols=[]
data=[]
kst=M1.mb.tag_get_data(M1.k_eq_tag,M1.all_faces)
ADJst=np.array([M1.mb.get_adjacencies(face, 3) for face in M1.all_faces])

cont=0
for f in M1.all_faces:
    adjs = ADJst[cont]
    if len(adjs)>1:
        k_eq=float(kst[cont])
        Gid_1=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[0]))
        Gid_2=int(M1.mb.tag_get_data(M1.ID_reordenado_tag,adjs[1]))

        lines.append(Gid_1)
        cols.append(Gid_2)
        data.append(k_eq)
        #T[Gid_1][Gid_2]=1
        lines.append(Gid_2)
        cols.append(Gid_1)
        data.append(k_eq)
        #T[Gid_2][Gid_1]=1
        lines.append(Gid_1)
        cols.append(Gid_1)
        data.append(-k_eq)
        #T[Gid_1][Gid_1]-=1
        lines.append(Gid_2)
        cols.append(Gid_2)
        data.append(-k_eq)
        #T[Gid_2][Gid_2]-=1


T_teste=csc_matrix((data,(lines,cols)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
t_assembly=time.time()-t_ass
del(T_teste)
'''#####################################################'''

SOL_ADM_1=linalg.spsolve(OR_ADM*T*OP_ADM,OR_ADM*b)    #-OR_ADM*T*corr_adm1_sd   +OR_ADM*T*corr_adm1_sd



SOL_ADM_fina_1=OP_ADM*SOL_ADM_1#-corr_adm1_sd.transpose()[0]

if first:
    print("resolvendo TPFA")
    t0=time.time()
    SOL_TPFA=linalg.spsolve(T,b)
    print("resolveu TPFA: ",time.time()-t0+t_assembly,t_assembly)
    np.save('SOL_TPFA.npy', SOL_TPFA)
else:
    SOL_TPFA = np.load('SOL_TPFA.npy')


erro=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)):
    erro[i]=100*abs((SOL_TPFA[i]-SOL_ADM_fina[i])/SOL_TPFA[i])

erroADM1=np.zeros(len(SOL_TPFA))
for i in range(len(SOL_TPFA)): erroADM1[i]=100*abs((SOL_TPFA[i]-SOL_ADM_fina_1[i])/SOL_TPFA[i])

ERRO_tag=M1.mb.tag_get_handle("erro", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
ERROadm1_tag=M1.mb.tag_get_handle("erroADM1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_TPFA_tag=M1.mb.tag_get_handle("Pressão TPFA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
Sol_ADM_tag=M1.mb.tag_get_handle("Pressão ADM", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

perm_xx_tag=M1.mb.tag_get_handle("Perm_xx", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
GIDs=M1.mb.tag_get_data(M1.ID_reordenado_tag,M1.all_volumes,flat=True)
perms_xx=M1.mb.tag_get_data(M1.perm_tag,M1.all_volumes)[:,0]
cont=0
for v in M1.all_volumes:
    gid=GIDs[cont]
    M1.mb.tag_set_data(ERRO_tag,v,erro[gid])
    M1.mb.tag_set_data(ERROadm1_tag,v,erroADM1[gid])
    M1.mb.tag_set_data(Sol_TPFA_tag,v,SOL_TPFA[gid])
    M1.mb.tag_set_data(Sol_ADM_tag,v,SOL_ADM_fina[gid])
    M1.mb.tag_set_data(perm_xx_tag,v,perms_xx[cont])
    cont+=1


i=0
#for v in M1.all_volumes:
#    perm_x=M1.mb.tag_get_data(M1.perm_tag,v)[0][0]
#    M1.mb.tag_set_data(perm_xx_tag,v,perm_x)

########################################################################
# calculando o fluxo coarse

################################################################################
p_tag = Sol_ADM_tag
gids_nv0 = M1.mb.tag_get_data(M1.ID_reordenado_tag, all_volumes, flat=True)
map_global = dict(zip(all_volumes, gids_nv0))
# name_tag_faces_boundary_meshsets
coarse_flux_nv2_tag = M1.mb.tag_get_handle('Q_nv2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
# oth1 = oth(M1.mb, mtu)
tag_faces_bound_nv2 = M1.mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(2))
all_faces_boundary_nv2 = M1.mb.tag_get_data(tag_faces_bound_nv2, 0, flat=True)[0]
all_faces_boundary_nv2 = M1.mb.get_entities_by_handle(all_faces_boundary_nv2)
for m in meshsets_nv1:
    qtot = 0.0
    elems = M1.mb.get_entities_by_handle(m)
    l3 = np.unique(M1.mb.tag_get_data(L3_ID_tag, elems, flat=True))
    if l3[0] > 2:
        continue
    faces = M1.mtu.get_bridge_adjacencies(elems, 3, 2)
    b_faces = rng.intersect(faces, all_faces_boundary_nv2)
    for face in b_faces:
        # keq = map_all_keqs[face]
        keq = M1.mb.tag_get_data(M1.k_eq_tag, face, flat=True)[0]
        elems2 = M1.mb.get_adjacencies(face, 3)
        # keq, s_grav, elems2 = oth.get_kequiv_by_face_quad(M1.mb, mtu, face, dict_tags['PERM'], dict_tags['AREA'])
        p = M1.mb.tag_get_data(p_tag, elems2, flat=True)
        flux = (p[1] - p[0])*keq
        if elems2[0] in elems:
            qtot += flux
        else:
            qtot -= flux
    qtot = abs(qtot)

    M1.mb.tag_set_data(coarse_flux_nv2_tag, elems, np.repeat(qtot, len(elems)))
    # mb.tag_set_data(coarse_flux_nv2_tag, elems, np.repeat(res, len(elems)))
#############################################################################

# p_tag = Sol_ADM_tag
# # name_tag_faces_boundary_meshsets
# coarse_flux_nv3_tag = mb.tag_get_handle('Q_nv3', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
# # oth1 = oth(M1.mb, M1.mtu)
# tag_faces_bound_nv3 = mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(3))
# all_faces_boundary_nv3 = mb.tag_get_data(tag_faces_bound_nv3, 0, flat=True)[0]
# all_faces_boundary_nv3 = mb.get_entities_by_handle(all_faces_boundary_nv3)
#
# for m in meshsets_nv2:
#     qtot = 0.0
#     elems = mb.get_entities_by_handle(m)
#     gids_nv2_adm = np.unique(M1.mb.tag_get_data(dict_tags['l2_ID'], elems, flat=True))
#     if len(gids_nv2_adm) > 1:
#         continue
#     faces = mtu.get_bridge_adjacencies(elems, 3, 2)
#     faces = rng.intersect(faces, all_faces_boundary_nv3)
#     for face in faces:
#         keq = map_all_keqs[face]
#         s_grav, elems2 = oth.get_sgrav_adjs_by_face(M1.mb, mtu, face, keq)
#         # keq, s_grav, elems2 = oth.get_kequiv_by_face_quad(M1.mb, mtu, face, dict_tags['PERM'], dict_tags['AREA'])
#         p = mb.tag_get_data(p_tag, elems2, flat=True)
#         flux = (p[1] - p[0])*keq
#         if oth.gravity == True:
#             flux += s_grav
#         if elems2[0] in elems:
#             qtot += flux
#         else:
#             qtot -= flux
#     qtot = abs(qtot)
#
#     mb.tag_set_data(coarse_flux_nv3_tag, elems, np.repeat(qtot, len(elems)))

#############################################################################

av=M1.mb.create_meshset()
M1.mb.add_entities(av,M1.all_volumes)

M1.mb.write_file('teste_3D_unstructured_18_2.vtk',[av])
os.chdir(flying_dir)
M1.mb.write_file('30x30x45_malha_adm')
np.save('faces_adjs_by_dual', faces_adjs_by_dual)
np.save('intern_adjs_by_dual', intern_adjs_by_dual)
print('New file created')
print(min(erro),max(erro))
print("razão norma L2",np.linalg.norm(erro)/np.linalg.norm(SOL_TPFA))
print('norma L2:', np.linalg.norm(erro))



x0=SOL_ADM_fina
ran=range(len(M1.all_volumes))
D=T[ran,ran]
ff=find(D)
l_inv=range(len(M1.all_volumes))
data_inv=np.repeat(1,len(ff[2]))/ff[2]
D_inv=csc_matrix((data_inv,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))
aux=D_inv*T.copy()

D1=D_inv.copy()#csc_matrix((D1.toarray()[0],(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))

D=aux[ran,ran]
ff=find(D)
l_inv=range(len(M1.all_volumes))
data_inv=np.repeat(1,len(ff[2]))/ff[2]
D_inv=csc_matrix((data_inv,(l_inv,l_inv)),shape=(len(M1.all_volumes),len(M1.all_volumes)))

aux[ran,ran]=0
LU=aux

ei=[]
l2i=[]
l2i.append(100*np.linalg.norm(erro)/np.linalg.norm(SOL_TPFA))
ei.append(erro.max())
x1=csc_matrix(x0.copy()).transpose()

k=D1*b
k2=D_inv*LU
max_iter=3000
tol=1
for i in range(int(max_iter/500)):
    if x1.max()<10000+tol and x1.min()>4000-tol:
        print(x1.max(),x1.min(),i*(j+1),"iterações")
        break
    for j in range(500):
        x1=k-k2*x1


#for i in range(2000): x1=k-k2*x1
x1=(x1.toarray()).transpose()[0]
if first:
    np.save('SOL_ADM_fina.npy', SOL_ADM_fina)
