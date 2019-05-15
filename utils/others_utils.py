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
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
from scipy.sparse.linalg import gmres

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')
loader = importlib.machinery.SourceFileLoader('prol_tpfa', utils_dir + '/prolongation_ams.py')
prol_tpfa = loader.load_module('prol_tpfa')

class OtherUtils:
    name_keq_tag = 'K_EQ'
    name_s_grav = 'S_GRAV'
    name_dirichlet_tag = 'P'
    name_neumann_tag = 'Q'
    name_perm_tag = 'PERM'
    name_area_tag = 'AREA'
    list_names_tags = [name_keq_tag, name_s_grav, name_dirichlet_tag, name_neumann_tag, name_perm_tag, name_area_tag]
    mi = 1.0
    tz = 27.0
    gama = 10.0
    gravity = False

    def __init__(self, mb=None, mtu=None):
        if mb == None:
            raise RuntimeError('Defina o mb core do pymoab')
        if mtu == None:
            raise RuntimeError('Defina o mtu do pymoab')

        self.list_tags = [mb.tag_get_handle(name) for name in OtherUtils.list_names_tags]
        self.mb = mb
        self.mtu = topo_util.MeshTopoUtil(mb)

    @staticmethod
    def mount_local_problem(mb, map_local, faces_in=None):
        """
        retorna os indices: linha, coluna, valor, sz
        sz = tamanho da matriz
        input:
            map_local: dicionario onde keys = volumes e values = id local
            faces_in: faces internas do conjunto de volumes
        output:
            inds:
        """
        elements = rng.Range(list(map_local.keys()))
        n = len(elements)

        if faces_in == None:
            # faces = utpy.get_all_faces(OtherUtils.mb, rng.Range(elements))
            # bound_faces = utpy.get_boundary_of_volumes(OtherUtils.mb, elements)
            # faces = rng.subtract(faces, bound_faces)
            faces = rng.subtract(utpy.get_faces(self.mb, rng.Range(elements)), utpy.get_boundary_of_volumes(self.mb, elements))
        else:
            faces = faces_in

        keqs = mb.tag_get_data(self.list_tags[0], faces, flat=True)
        elems = [self.mb.get_adjacencies(face) for face in faces]
        s_gravs = self.mb.tag_get_data(self.list_tags[1], faces, flat=True)
        dict_keq = dict(zip(faces, keqs))
        dict_elems = dict(zip(faces, elems))
        dict_s_grav = dict(zip(faces, s_gravs))

        linesM = np.array([])
        colsM = np.array([])
        valuesM = np.array([])
        linesM2 = np.array([])
        valuesM2 = np.array([])
        szM = [n, n]

        b = np.zeros(n)
        s = np.zeros(n)

        for face in faces:
            elems = dict_elems[face]
            keq = dict_keq[face]
            s_grav = dict_s_grav[face]

            linesM = np.append(linesM, [map_local[elems[0]], map_local[elems[1]]])
            colsM = np.append(colsM, [map_local[elems[1]], map_local[elems[0]]])
            valuesM = np.append(valuesM, [-keq, -keq])

            ind0 = np.where(linesM2 == map_local[elems[0]])
            if len(ind0[0]) == 0:
                linesM2 = np.append(linesM2, map_local[elems[0]])
                valuesM2 = np.append(valuesM2, [keq])
            else:
                valuesM2[ind0[0]] += keq

            ind1 = np.where(linesM2 == map_local[elems[1]])
            if len(ind1[0]) == 0:
                linesM2 = np.append(linesM2, map_local[elems[1]])
                valuesM2 = np.append(valuesM2, [keq])
            else:
                valuesM2[ind1[0]] += keq

            s[map_local[elems[0]]] += s_grav
            s[map_local[elems[1]]] -= s_grav

        linesM = np.append(linesM, linesM2)
        colsM = np.append(colsM, linesM2)
        valuesM = np.append(valuesM, valuesM2)

        linesM = linesM.astype(np.int32)
        colsM = colsM.astype(np.int32)

        inds = np.array([linesM, colsM, valuesM, szM, False, False])

        if self.gravity == True:
            return inds, s
        else:
            return inds, b

    def get_flow_on_boundary(self, elements, p_tag, keq_tag, bound_faces=None):
        """
        input:
            elements: volumes ou meshset de volumes
            p_tag = tag da pressao
            bound_faces = faces no contorno do volume
        output:
            dict com keys = elements e values = fluxo na face do contorno
        """
        dict_flux = {}
        elements = utpy.get_elements(self.mb, elements)
        if bound_faces  == None:
            faces = utpy.get_boundary_of_volumes(self.mb, elements)
        else:
            faces = bound_faces

        keqs = self.mb.tag_get_data(keq_tag, faces, flat=True)

        s_gravs = self.mb.tag_get_data(self.list_tags[1], faces, flat=True)
        dict_keq = dict(zip(faces, keqs))
        dict_s_grav = dict(zip(faces, s_gravs))

        for face in faces:
            elems = self.mb.get_adjacencies(face, 3)
            if len(elems) < 2:
                continue
            keq = dict_keq[face]
            s_grav = dict_s_grav[face]

            p = self.mb.tag_get_data(p_tag, elems, flat=True)
            flux = (p[1] - p[0])*keq
            if OtherUtils.gravity == True:
                flux += s_grav


            if elems[0] in elements:
                dict_flux[elems[0]] = flux
            else:
                dict_flux[elems[1]] = -flux

        return dict_flux

    @staticmethod
    def set_boundary_dirichlet(map_local, map_values, b, inds):
        inds2 = inds.copy()
        for v in boundary_elems:
            gid = map_local[v]
            indices = np.where(inds2[0] == gid)[0]
            inds2[0] = np.delete(inds2[0], indices)
            inds2[1] = np.delete(inds2[1], indices)
            inds2[2] = np.delete(inds2[2], indices)

            inds2[0] = np.append(inds2[0], np.array([gid]))
            inds2[1] = np.append(inds2[1], np.array([gid]))
            inds2[2] = np.append(inds2[2], np.array([1.0]))
            b[gid] = map_values[v]

        inds2[0] = inds2[0].astype(np.int32)
        inds2[1] = inds2[1].astype(np.int32)

        return inds2, b

    @staticmethod
    def set_boundary_dirichlet_matrix(map_local, map_values, b, T):
        t = T.shape[0]
        T2 = T.copy()
        zeros = np.zeros(t)
        for v, value in map_values.items():
            gid = map_local[v]
            T2[gid] = zeros
            T2[gid, gid] = 1.0
            b[gid] = value

        return T2, b

    @staticmethod
    def set_boundary_dirichlet_matrix_v02(ids, values, b, T):
        t = T.shape[0]
        T2 = T.copy()
        n1 = len(ids)
        T2[ids] = sp.lil_matrix((n1, t))
        T2[ids, ids] = np.ones(n1)
        b[ids] = values

        return T2, b

    @staticmethod
    def set_boundary_neumann(map_local, map_values, b):
        for v, val in map_values.items():
            gid = map_local[v]
            b[gid] += val

        return b

    @staticmethod
    def set_boundary_neumann_v02(ids, values, b):
        b[ids] = values
        return b

    @staticmethod
    def get_slice_by_inds(inds, slice_row, slice_col):
        """
        retorna um slice a partir dos inds
        input:
            inds
            slice_row
            slice_col
        output:
            inds2:
        """

        lines2 = np.array([])
        cols2 = np.array([])
        values2 = np.array([], dtype=np.float64)

        map_l = dict(zip(slice_rows, range(len(slice_rows))))
        map_c = dict(zip(slice_cols, range(len(slice_cols))))
        sz = [len(slice_rows), len(slice_cols)]

        for i in slice_rows:
            assert i in inds[0]
            indices = np.where(inds[0] == i)[0]
            cols = np.array([inds[1][j] for j in indices if inds[1][j] in slice_cols])
            vals = np.array([inds[2][j] for j in indices if inds[1][j] in slice_cols])
            lines = np.repeat(i, len(cols))

            lines2 = np.append(lines2, lines)
            cols2 = np.append(cols2, cols)
            values2 = np.append(values2, vals)

        lines2 = lines2.astype(np.int32)
        cols2 = cols2.astype(np.int32)
        local_inds_l = np.array([map_l[j] for j in lines2]).astype(np.int32)
        local_inds_c = np.array([map_c[j] for j in cols2]).astype(np.int32)

        inds2 = np.array([lines2, cols2, values2, sz, local_inds_l, local_inds_c])
        return inds2

    @staticmethod
    def get_tc_tpfa(inds_tc_wirebasket, wirebasket_numbers):
        """
        obtem a 'tpfa-lização' da matriz mpfa
        input:
            wirebasket_numbers = [ni, nf, ne, nv]
            inds_tc_wirebasket: mpfa wirebasket transmissibility  of coarse level
        output:
            inds_tc_tpfa
        """
        nni = wirebasket_numbers[0]
        nnf = wirebasket_numbers[1] + nni
        nne = wirebasket_numbers[2] + nnf
        nnv = wirebasket_numbers[3] + nne

        inds_tc_tpfa = inds_tc_wirebasket.copy()

        inds_tc_tpfa = OtherUtils.indices_maiores(inds_tc_tpfa, 0, nni, nnf)
        inds_tc_tpfa = OtherUtils.indices_maiores(inds_tc_tpfa, nni, nnf, nne)
        inds_tc_tpfa = OtherUtils.indices_menores(inds_tc_tpfa, nnf, nne, nni)
        inds_tc_tpfa = OtherUtils.indices_menores(inds_tc_tpfa, nne, nnv, nnf)

        return inds_tc_tpfa

    @staticmethod
    def get_tmod_by_inds(inds_wirebasket, wirebasket_numbers):
        """
        obtem a transmissibilidade wirebasket modificada
        """
        # ordem: vertex, edge, face, intern
        inds = inds_wirebasket
        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]

        lines2 = np.array([], dtype=np.int32)
        cols2 = lines2.copy()
        values2 = np.array([], dtype='float64')

        lines = set(inds[0])
        sz = inds[3][:]

        verif1 = ni
        verif2 = ni+nf
        rg1 = np.arange(ni, ni+nf)

        for i in lines:
            indice = np.where(inds[0] == i)[0]
            if i < ni:
                lines2 = np.hstack((lines2, inds[0][indice]))
                cols2 = np.hstack((cols2, inds[1][indice]))
                values2 = np.hstack((values2, inds[2][indice]))
                continue
            elif i >= ni+nf+ne:
                continue
            elif i in rg1:
                verif = verif1
            else:
                verif = verif2

            lines_0 = inds[0][indice]
            cols_0 = inds[1][indice]
            vals_0 = inds[2][indice]

            inds_minors = np.where(cols_0 < verif)[0]
            vals_minors = vals_0[inds_minors]

            vals_0[np.where(cols_0 == i)[0]] += sum(vals_minors)
            inds_sup = np.where(cols_0 >= verif)[0]
            lines_0 = lines_0[inds_sup]
            cols_0 = cols_0[inds_sup]
            vals_0 = vals_0[inds_sup]


            lines2 = np.hstack((lines2, lines_0))
            cols2 = np.hstack((cols2, cols_0))
            values2 = np.hstack((values2, vals_0))

        lines2 = lines2.astype(np.int32)
        cols2 = cols2.astype(np.int32)

        inds2 = np.array([lines2, cols2, values2, sz])

        return inds2

    @staticmethod
    def indices_maiores(inds_tc_tpfa, ind1, ind2, verif):

        for i in range(ind1, ind2):
            indices = np.where(inds_tc_tpfa[0] == i)[0]
            line = inds_tc_tpfa[0][indices]
            cols = inds_tc_tpfa[1][indices]
            values = inds_tc_tpfa[2][indices]

            inds_tc_tpfa[0] = np.delete(inds_tc_tpfa[0], indices)
            inds_tc_tpfa[1] = np.delete(inds_tc_tpfa[1], indices)
            inds_tc_tpfa[2] = np.delete(inds_tc_tpfa[2], indices)

            indices = np.where(cols >= verif)[0]

            # soma = values[indices].sum()
            # values[cols == i] += soma
            values[cols == i] += values[indices].sum()
            line = np.delete(line, indices)
            cols = np.delete(cols, indices)
            values = np.delete(values, indices)

            inds_tc_tpfa[0] = np.append(inds_tc_tpfa[0], line)
            inds_tc_tpfa[1] = np.append(inds_tc_tpfa[1], cols)
            inds_tc_tpfa[2] = np.append(inds_tc_tpfa[2], values)

        return inds_tc_tpfa

    @staticmethod
    def indices_menores(inds_tc_tpfa, ind1, ind2, verif):
        for i in range(ind1, ind2):
            indices = np.where(inds_tc_tpfa[0] == i)[0]
            line = inds_tc_tpfa[0][indices]
            cols = inds_tc_tpfa[1][indices]
            values = inds_tc_tpfa[2][indices]

            inds_tc_tpfa[0] = np.delete(inds_tc_tpfa[0], indices)
            inds_tc_tpfa[1] = np.delete(inds_tc_tpfa[1], indices)
            inds_tc_tpfa[2] = np.delete(inds_tc_tpfa[2], indices)

            indices = np.where(cols < verif)[0]

            # soma = values[indices].sum()
            # values[cols == i] += soma
            values[cols == i] += values[indices].sum()
            line = np.delete(line, indices)
            cols = np.delete(cols, indices)
            values = np.delete(values, indices)

            inds_tc_tpfa[0] = np.append(inds_tc_tpfa[0], line)
            inds_tc_tpfa[1] = np.append(inds_tc_tpfa[1], cols)
            inds_tc_tpfa[2] = np.append(inds_tc_tpfa[2], values)

        return inds_tc_tpfa

    @staticmethod
    def read_perms_and_phi_spe10():
        # nx = 60
        # ny = 220
        # nz = 85
        # N = nx*ny*nz
        # l1 = [N, 2*N, 3*N]
        # l2 = [0, 1, 2]
        #
        # ks = np.loadtxt('spe_perm.dat')
        # t1, t2 = ks.shape
        # ks = ks.reshape((t1*t2))
        # ks2 = np.zeros((N, 9))
        #
        #
        # for i in range(0, N):
        #     # as unidades do spe_10 estao em milidarcy
        #     # unidade de darcy em metro quadrado =  (1 Darcy)*(9.869233e-13 m^2/Darcy)
        #     # fonte -- http://www.calculator.org/property.aspx?name=permeability
        #     ks2[i, 0] = ks[i]#*(10**(-3))# *9.869233e-13
        #
        # cont = 0
        # for i in range(N, 2*N):
        #     ks2[cont, 4] = ks[i]#*(10**(-3))# *9.869233e-13
        #     cont += 1
        #
        # cont = 0
        # for i in range(2*N, 3*N):
        #     ks2[cont, 8] = ks[i]#*(10**(-3))# *9.869233e-13
        #     cont += 1
        #
        #
        #
        # cont = None
        # phi = np.loadtxt('spe_phi.dat')
        # t1, t2 = phi.shape
        # phi = phi.reshape(t1*t2)
        # np.savez_compressed('spe10_perms_and_phi', perms = ks2, phi = phi)
        # ks2 = None
        #
        # # obter a permeabilidade de uma regiao
        # # digitar o inicio e o fim da regiao

        ks = np.load('spe10_perms_and_phi.npz')['perms']
        phi = np.load('spe10_perms_and_phi.npz')['phi']

        # gid1 = [0, 0, 50]
        # gid2 = [gid1[0] + self.nx-1, gid1[1] + self.ny-1, gid1[2] + self.nz-1]
        #
        # gid1 = np.array(gid1)
        # gid2 = np.array(gid2)
        #
        # dif = gid2 - gid1 + np.array([1, 1, 1])
        # permeabilidade = []
        # fi = []
        #
        # cont = 0
        # for k in range(dif[2]):
        #     for j in range(dif[1]):
        #         for i in range(dif[0]):
        #             gid = gid1 + np.array([i, j, k])
        #             gid = gid[0] + gid[1]*nx + gid[2]*nx*ny
        #             # permeabilidade[cont] = ks[gid]
        #             permeabilidade.append(ks[gid])
        #             fi.append(phi[gid])
        #             cont += 1
        # cont = 0
        #
        # for volume in self.all_fine_vols:
        #     self.mb.tag_set_data(self.perm_tag, volume, permeabilidade[cont])
        #     self.mb.tag_set_data(self.fi_tag, volume, fi[cont])
        #     cont += 1
        #
        #
        #
        # # self.mb.tag_set_data(self.perm_tag, self.all_fine_vols, permeabilidade)
        # # self.mb.tag_set_data(self.fi_tag, self.all_fine_vols, fi)
        # for volume in self.all_fine_vols:
        #     gid = self.mb.tag_get_data(self.global_id_tag, volume, flat=True)
        #     perm = self.mb.tag_get_data(self.perm_tag, volume).reshape([3,3])
        #     fi2 = self.mb.tag_get_data(self.fi_tag, volume, flat = True)[0]

        return ks, phi

    @staticmethod
    def fine_transmissibility_structured(mb, mtu, map_global, faces_in=None):
        """
        """
        # mtu = topo_util.MeshTopoUtil(mb)
        keq_tag = mb.tag_get_handle(OtherUtils.name_keq_tag)
        s_grav_tag = mb.tag_get_handle(OtherUtils.name_s_grav)
        perm_tag = mb.tag_get_handle(OtherUtils.name_perm_tag)
        area_tag = mb.tag_get_handle(OtherUtils.name_area_tag)

        elements = rng.Range(np.array(list(map_global.keys())))
        n = len(elements)
        all_keqs = []
        all_s_gravs = []

        if faces_in == None:
            all_faces = utpy.get_faces(mb, rng.Range(elements))
            # bound_faces = utpy.get_boundary_of_volumes(mb, elements)
            # faces = rng.subtract(all_faces, bound_faces)
            faces = rng.subtract(all_faces, utpy.get_boundary_of_volumes(mb, elements))
        else:
            faces = faces_in
            all_faces = utpy.get_faces(mb, rng.Range(elements))

        T = sp.lil_matrix((n, n))
        s = np.zeros(n)

        # cont = 0
        for face in faces:
            #1
            keq, s_grav, elems = OtherUtils.get_kequiv_by_face_quad(mb, mtu, face, perm_tag, area_tag)
            T[map_global[elems[0]], map_global[elems[1]]] = -keq
            T[map_global[elems[1]], map_global[elems[0]]] = -keq
            T[map_global[elems[0]], map_global[elems[0]]] += keq
            T[map_global[elems[1]], map_global[elems[1]]] += keq
            s_grav *= -1
            s[map_global[elems[0]]] += s_grav
            s[map_global[elems[1]]] -= s_grav
            all_keqs.append(keq)
            all_s_gravs.append(s_grav)

        bound_faces = rng.subtract(all_faces, faces)
        mb.tag_set_data(keq_tag, bound_faces, np.repeat(0.0, len(bound_faces)))
        mb.tag_set_data(s_grav_tag, bound_faces, np.repeat(0.0, len(bound_faces)))
        mb.tag_set_data(keq_tag, faces, all_keqs)
        mb.tag_set_data(s_grav_tag, faces, all_s_gravs)

        if OtherUtils.gravity == True:
            return T, s
        else:
            return T, np.zeros(n)

    def fine_transmissibility_structured_bif(self, map_global, mobi_tag, faces_in=None):
        """
        """
        # mtu = topo_util.MeshTopoUtil(mb)
        s_grav_tag = self.mb.tag_get_handle(OtherUtils.name_s_grav)

        elements = rng.Range(np.array(list(map_global.keys())))
        n = len(elements)
        all_s_gravs = []

        if faces_in == None:
            all_faces = utpy.get_faces(mb, rng.Range(elements))
            # bound_faces = utpy.get_boundary_of_volumes(mb, elements)
            # faces = rng.subtract(all_faces, bound_faces)
            faces = rng.subtract(all_faces, utpy.get_boundary_of_volumes(mb, elements))
        else:
            faces = faces_in
            all_faces = utpy.get_faces(mb, rng.Range(elements))

        T = sp.lil_matrix((n, n))
        s = np.zeros(n)

        # cont = 0
        for face in faces:
            #1
            keq, s_grav, elems = self.get_mobi_by_face_quad_bif(face, mobi_tag)
            T[map_global[elems[0]], map_global[elems[1]]] = -keq
            T[map_global[elems[1]], map_global[elems[0]]] = -keq
            T[map_global[elems[0]], map_global[elems[0]]] += keq
            T[map_global[elems[1]], map_global[elems[1]]] += keq
            s[map_global[elems[0]]] += s_grav
            s[map_global[elems[1]]] -= s_grav
            all_s_gravs.append(s_grav)

        self.mb.tag_set_data(s_grav_tag, faces, all_s_gravs)

        if OtherUtils.gravity == True:
            return T, s
        else:
            return T, np.zeros(n)

    @staticmethod
    def get_kequiv_by_face_quad(mb, mtu, face, perm_tag, area_tag):
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

        elems = mb.get_adjacencies(face, 3)
        k1 = mb.tag_get_data(perm_tag, elems[0]).reshape([3, 3])
        k2 = mb.tag_get_data(perm_tag, elems[1]).reshape([3, 3])
        centroid1 = mtu.get_average_position([elems[0]])
        centroid2 = mtu.get_average_position([elems[1]])
        direction = centroid2 - centroid1
        uni = OtherUtils.unitary(direction)
        k1 = np.dot(np.dot(k1,uni), uni)
        k2 = np.dot(np.dot(k2,uni), uni)
        area = mb.tag_get_data(area_tag, face, flat=True)[0]
        # keq = OtherUtils.kequiv(k1, k2)*area/(OtherUtils.mi*np.linalg.norm(direction))
        keq = 1.0
        z1 = OtherUtils.tz - centroid1[2]
        z2 = OtherUtils.tz - centroid2[2]
        s_gr = OtherUtils.gama*keq*(z1-z2)

        return keq, s_gr, elems

    @staticmethod
    def get_sgrav_adjs_by_face(mb, mtu, face, keq):
        elems = mb.get_adjacencies(face, 3)
        centroid1 = mtu.get_average_position([elems[0]])
        centroid2 = mtu.get_average_position([elems[1]])
        z1 = OtherUtils.tz - centroid1[2]
        z2 = OtherUtils.tz - centroid2[2]
        s_gr = OtherUtils.gama*keq*(z1-z2)

        return s_gr, elems


    def get_mobi_by_face_quad_bif(self, face, mobi_tag):
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

        elems = self.mb.get_adjacencies(face, 3)
        mobi = self.mb.tag_get_data(mobi_tag, face, flat=True)[0]
        centroid1 = self.mtu.get_average_position([elems[0]])
        centroid2 = self.mtu.get_average_position([elems[1]])
        direction = centroid2 - centroid1
        z1 = OtherUtils.tz - centroid1[2]
        z2 = OtherUtils.tz - centroid2[2]
        s_gr = OtherUtils.gama*mobi*(z1-z2)

        return mobi, s_gr, elems


    def calculate_pcorr(self, elements, vertex_elem, pcorr_tag, pms_tag, keq_tag, faces, boundary_faces, volumes_d, volumes_n):
        map_local = dict(zip(elems, range(len(elems))))

        n = len(elems)
        T = sp.lil_matrix((n, n))
        b = np.zeros(n)

        for i, face in enumerate(faces):
            elems = self.mb.get_adjacencies(face, 3)
            keq, s_grav, elems = self.get_mobi_by_face_quad_bif(face, keq_tag)
            if face in boundary_faces:
                p = self.mb.tag_get_data(pms_tag, elems, flat=True)
                flux = (p[1] - p[0])*keq
                if self.gravity == True:
                    flux += s_grav

                b[map_local[elems[0]]] += flux
                b[map_local[elems[1]]] -= flux
                continue

            T[map_local[elems[0]], map_local[elems[1]]] = -keq
            T[map_local[elems[1]], map_local[elems[0]]] = -keq
            T[map_local[elems[0]], map_local[elems[0]]] += keq
            T[map_local[elems[1]], map_local[elems[1]]] += keq

        T[map_local[vertex], map_local[vertex]] = 1
        p = self.mb.tag_get_data(pms_tag, vertex, flat=True)[0]
        b[map_local[vertex]] = p

        bound1 = rng.intersect(elems, volumes_d)
        bound1 = rng.subtract(bound1, vertex)
        if len(bound1) > 0:
            ids = np.array([map_local[v] for v in bound1])
            T[ids] = sp.lil_matrix((len(ids), n))
            T[ids, ids] = np.ones(len(ids))
            ps = self.mb.tag_get_data(pms_tag, bound1, flat=True)
            b[ids] = ps

        bound2 = rng.intersect(elems, volumes_n)
        bound2 = rng.subtract(bound2, vertex)
        if len(bound2) > 0:
            ids = np.array([map_local[v] for v in bound2])
            qs = self.mb.tag_get_data(self.list_tags[3], bound2, flat=True)
            b[ids] += qs

        x = linalg.spsolve(T, b)
        self.mb.tag_set_data(pcorr_tag, elements, x)

    @staticmethod
    def unitary(l):
        """
        obtem o vetor unitario positivo da direcao de l

        """
        uni = np.absolute(l/np.linalg.norm(l))
        # uni = np.abs(uni)

        return uni

    @staticmethod
    def kequiv(k1,k2):
        """
        obbtem o k equivalente entre k1 e k2

        """
        # keq = ((2*k1*k2)/(h1*h2))/((k1/h1) + (k2/h2))
        keq = (2*k1*k2)/(k1+k2)

        return keq

    @staticmethod
    def lu_inv(M):
        """
        M = matriz do scipy
        """
        M = M.tocsc()

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

    @staticmethod
    def get_op_by_wirebasket_Tf(Tf_wire, wirebasket_numbers):
        matrices_tf = find(Tf_wire)
        inds_tf = [matrices_tf[0], matrices_tf[1], matrices_tf[2], Tf_wire.shape]
        del matrices_tf
        inds_tf_mod = OtherUtils.get_tmod_by_inds(inds_tf, wirebasket_numbers)
        Tf_mod = sp.lil_matrix(tuple(inds_tf_mod[3]))
        Tf_mod[inds_tf_mod[0], inds_tf_mod[1]] = inds_tf_mod[2]
        OP1_AMS = prol_tpfa.get_op_AMS_TPFA(Tf_mod, wirebasket_numbers).tocsc()

        return OP1_AMS

    @staticmethod
    def get_op_by_wirebasket_Tf_wire_coarse(Tf_wire_coarse, wirebasket_numbers):
        matrices_tf = find(Tf_wire_coarse)
        inds_tf = [matrices_tf[0], matrices_tf[1], matrices_tf[2], Tf_wire_coarse.shape]
        del matrices_tf
        inds_tf_mod = OtherUtils.get_tc_tpfa(inds_tf, wirebasket_numbers)
        inds_tf_mod = OtherUtils.get_tmod_by_inds(inds_tf_mod, wirebasket_numbers)
        Tf_mod = sp.lil_matrix(tuple(inds_tf_mod[3]))
        Tf_mod[inds_tf_mod[0], inds_tf_mod[1]] = inds_tf_mod[2]
        OP2_AMS = prol_tpfa.get_op_AMS_TPFA_dep(Tf_mod, wirebasket_numbers).tocsc()

        return OP2_AMS

    @staticmethod
    def get_solution(T, b):
        T = T.tocsc()
        x = linalg.spsolve(T, b)
        return x

    @staticmethod
    def get_Tmod_by_sparse_wirebasket_matrix(Tf_wire, wirebasket_numbers):

        Tmod = Tf_wire.copy().tolil()
        ni = wirebasket_numbers[0]
        nf = wirebasket_numbers[1]
        ne = wirebasket_numbers[2]
        nv = wirebasket_numbers[3]

        nni = wirebasket_numbers[0]
        nnf = wirebasket_numbers[1] + nni
        nne = wirebasket_numbers[2] + nnf
        nnv = wirebasket_numbers[3] + nne

        #internos
        Aii = Tmod[0:nni, 0:nni]
        Aif = Tmod[0:nni, nni:nnf]

        #faces
        Aff = Tmod[nni:nnf, nni:nnf]
        Afe = Tmod[nni:nnf, nnf:nne]
        soma = Aif.transpose().sum(axis=1)
        d1 = np.matrix(Aff.diagonal()).reshape([nf, 1])
        d1 += soma
        Aff.setdiag(d1)

        #arestas
        Aee = Tmod[nnf:nne, nnf:nne]
        Aev = Tmod[nnf:nne, nne:nnv]
        soma = Afe.transpose().sum(axis=1)
        d1 = np.matrix(Aee.diagonal()).reshape([ne, 1])
        d1 += soma
        Aee.setdiag(d1)
        Ivv = sp.identity(nv)

        As = {}
        As['Aii'] = Aii
        As['Aif'] = Aif
        As['Aff'] = Aff
        As['Afe'] = Afe
        As['Aee'] = Aee
        As['Aev'] = Aev
        As['Ivv'] = Ivv

        return As
