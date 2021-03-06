import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
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
        self.mi = 1.0

        '''self.GLOBAL_ID_tag = self.mb.tag_get_handle(
            "Global_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)'''

        self.create_tags()
        self.mb.tag_set_data(self.ids_volumes_tag, self.all_volumes, np.arange(len(self.all_volumes)))
        self.mb.tag_set_data(self.ids_faces_tag, self.all_faces, np.arange(len(self.all_faces)))
        self.set_k()
        self.set_phi()
        # self.set_k_and_phi_structured_spe10()
        self.set_volumes()
        #self.set_information("PERM", self.all_volumes, 3)
        self.get_boundary_faces()
        self.set_vols_centroids()
        self.gravity = False
        self.gama = 10
        # self.set_keq_simple()
        self.set_keq_structured()

    def create_tags(self):
        self.perm_tag = self.mb.tag_get_handle("PERM", 9, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.phi_tag = self.mb.tag_get_handle("PHI", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.cent_tag = self.mb.tag_get_handle("CENT", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.finos_tag = self.mb.tag_get_handle("finos", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.intermediarios_tag = self.mb.tag_get_handle("intermediarios", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.press_value_tag = self.mb.tag_get_handle("P", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.vazao_value_tag = self.mb.tag_get_handle("Q", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.all_faces_boundary_tag = self.mb.tag_get_handle("FACES_BOUNDARY", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.area_tag = self.mb.tag_get_handle("AREA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.GLOBAL_ID_tag = self.mb.tag_get_handle("G_ID_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.ID_reordenado_tag = self.mb.tag_get_handle("ID_reord_tag", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.keq_tag = self.mb.tag_get_handle("K_EQ", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.s_grav_tag = self.mb.tag_get_handle("S_GRAV", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.volume_tag = self.mb.tag_get_handle("VOLUME", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.ltot_tag = self.mb.tag_get_handle("L_TOT", 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.r0_tag = self.mb.tag_get_handle("R0", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.r1_tag = self.mb.tag_get_handle("R1", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.gama_tag = self.mb.tag_get_handle("GAMA", 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.gravity_tag = self.mb.tag_get_handle('GRAVITY', 1, types.MB_TYPE_BIT, types.MB_TAG_SPARSE, True)
        self.bifasico_tag = self.mb.tag_get_handle('BIFASICO', 1, types.MB_TYPE_BIT, types.MB_TAG_SPARSE, True)
        self.mpfa_tag = self.mb.tag_get_handle('MPFA', 1, types.MB_TYPE_BIT, types.MB_TAG_SPARSE, True)
        self.mi_tag = self.mb.tag_get_handle('MI', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.tz_tag = self.mb.tag_get_handle('TZ', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.ids_volumes_tag = self.mb.tag_get_handle('IDS_VOLUMES', 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        self.ids_faces_tag = self.mb.tag_get_handle('IDS_FACES', 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        #deleteme
        self.wells_dirichlet_tag = self.mb.tag_get_handle("WELLS_D", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_neumann_tag = self.mb.tag_get_handle("WELLS_N", 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)

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

    def set_phi(self):
        self.mb.tag_set_data(self.phi_tag, self.all_volumes, np.repeat(0.3, len(self.all_volumes)))

    def set_volumes(self):
        self.mb.tag_set_data(self.volume_tag, self.all_volumes, np.repeat(1.0, len(self.all_volumes)))

    def set_k_and_phi_structured_spe10(self):
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        parent_parent_dir = os.path.dirname(parent_dir)
        input_dir = os.path.join(parent_parent_dir, 'input')
        os.chdir(input_dir)
        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']
        os.chdir(parent_dir)

        nx = 60
        ny = 220
        nz = 85
        perms = []
        phis = []

        v0 = self.all_volumes[0]

        points = self.mtu.get_bridge_adjacencies(v0, 3, 0)
        coords = self.mtu.get_coords(points).reshape([len(points), 3])
        maxs = coords.max(axis=0)
        mins = coords.min(axis=0)

        hs = maxs - mins

        k = 1.0
        for v in self.all_volumes:
            centroid = self.mtu.get_average_position([v])
            # ijk = np.array([centroid[0]//1.0, centroid[1]//1.0, centroid[2]//1.0])
            ijk = np.array([centroid[0]//hs[0], centroid[1]//hs[1], centroid[2]//hs[2]])
            e = int(ijk[0] + ijk[1]*nx + ijk[2]*nx*ny)
            # perm = ks[e]*k
            # fi = phi[e]
            perms.append(ks[e]*k)
            phis.append(phi[e])

        self.mb.tag_set_data(self.perm_tag, self.all_volumes, perms)
        self.mb.tag_set_data(self.phi_tag, self.all_volumes, phis)

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

    def set_vols_centroids(self):
        all_centroids = np.array([self.mtu.get_average_position([v]) for v in self.all_volumes])
        self.all_centroids = all_centroids
        for cent, v in zip(all_centroids, self.all_volumes):
            self.mb.tag_set_data(self.cent_tag, v, cent)

    def create_tags_bif(self):
        self.mi_w_tag = self.mb.tag_get_handle('MI_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.mi_o_tag = self.mb.tag_get_handle('MI_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.gama_w_tag = self.mb.tag_get_handle('GAMA_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.gama_o_tag = self.mb.tag_get_handle('GAMA_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.sor_tag = self.mb.tag_get_handle('SOR', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.swc_tag = self.mb.tag_get_handle('SWC', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.nw_tag = self.mb.tag_get_handle('NW', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.no_tag = self.mb.tag_get_handle('NO', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.sat_tag = self.mb.tag_get_handle('SAT', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.fw_tag = self.mb.tag_get_handle('FW', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.lamb_w_tag = self.mb.tag_get_handle('LAMB_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.lamb_o_tag = self.mb.tag_get_handle('LAMB_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.lbt_tag = self.mb.tag_get_handle('LBT', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.total_flux_tag = self.mb.tag_get_handle('TOTAL_FLUX', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.flux_w_tag = self.mb.tag_get_handle('FLUX_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.mobi_in_faces_tag = self.mb.tag_get_handle('MOBI_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.mobi_w_in_faces_tag = self.mb.tag_get_handle('MOBI_W_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.flux_in_faces_tag = self.mb.tag_get_handle('FLUX_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.wells_injector_tag = self.mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.wells_producer_tag = self.mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
        self.loops_tag = self.mb.tag_get_handle('LOOPS', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.total_time_tag = self.mb.tag_get_handle('TOTAL_TIME', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.fw_in_faces_tag = self.mb.tag_get_handle('FW_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.dfds_tag = self.mb.tag_get_handle('DFDS', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

    def set_keq_simple(self):
        faces_in = rng.subtract(self.all_faces, self.all_boundary_faces)
        self.mb.tag_set_data(self.keq_tag, faces_in, np.repeat(1.0, len(faces_in)))

    def set_keq_structured(self):
        v0 = self.all_volumes[0]
        nodes = self.mtu.get_bridge_adjacencies(v0, 3, 0)
        coords_nodes = self.mb.get_coords(nodes).reshape([len(nodes), 3])
        xmin = coords_nodes[:,0].min()
        xmax = coords_nodes[:,0].max()
        ymin = coords_nodes[:,1].min()
        ymax = coords_nodes[:,1].max()
        zmin = coords_nodes[:,2].min()
        zmax = coords_nodes[:,2].max()
        hs = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
        faces_in = rng.subtract(self.all_faces, self.all_boundary_faces)
        all_ks = self.mb.tag_get_data(self.perm_tag, self.all_volumes)
        all_keqs = np.zeros(len(faces_in))
        map_volumes = dict(zip(self.all_volumes, range(len(self.all_volumes))))
        areas = self.mb.tag_get_data(self.area_tag, faces_in, flat=True)
        for i, face in enumerate(faces_in):
            elems = self.mb.get_adjacencies(face, 3)
            if len(elems) < 2:
                continue
            k0 = all_ks[map_volumes[elems[0]]].reshape([3,3])
            k1 = all_ks[map_volumes[elems[1]]].reshape([3,3])
            c0 = self.all_centroids[map_volumes[elems[0]]]
            c1 = self.all_centroids[map_volumes[elems[1]]]
            direction = c1-c0
            h = np.linalg.norm(direction)
            uni = np.absolute(direction/h)
            k0 = np.dot(np.dot(k0, uni), uni)
            k1 = np.dot(np.dot(k1, uni), uni)
            area = areas[i]
            keq = area*(2*k1*k0)/(h*(k1+k0)*self.mi)
            all_keqs[i] = keq

        self.mb.tag_set_data(self.keq_tag, self.all_faces, np.repeat(0.0, len(self.all_faces)))
        self.mb.tag_set_data(self.keq_tag, faces_in, all_keqs)

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
