import numpy as np
from pymoab import core, types, rng, topo_util, skinner
import os
import sys
import io
import yaml
import scipy.sparse as sp
import time

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils


class sol_direta_bif:
    def __init__(self, mb, mtu, all_volumes):
        self.mi_w = mb.tag_get_data(mb.tag_get_handle('MI_W'), 0, flat=True)[0]
        self.mi_o = mb.tag_get_data(mb.tag_get_handle('MI_O'), 0, flat=True)[0]
        self.gama_w = mb.tag_get_data(mb.tag_get_handle('GAMA_W'), 0, flat=True)[0]
        self.gama_o = mb.tag_get_data(mb.tag_get_handle('GAMA_O'), 0, flat=True)[0]
        self.Sor = mb.tag_get_data(mb.tag_get_handle('SOR'), 0, flat=True)[0]
        self.Swc = mb.tag_get_data(mb.tag_get_handle('SWC'), 0, flat=True)[0]
        self.nw = mb.tag_get_data(mb.tag_get_handle('NW'), 0, flat=True)[0]
        self.no = mb.tag_get_data(mb.tag_get_handle('NO'), 0, flat=True)[0]
        self.tz = mb.tag_get_data(mb.tag_get_handle('TZ'), 0, flat=True)[0]
        self.loops = mb.tag_get_data(mb.tag_get_handle('LOOPS'), 0, flat=True)[0]
        self.total_time = mb.tag_get_data(mb.tag_get_handle('TOTAL_TIME'), 0, flat=True)[0]
        self.gravity = mb.tag_get_data(mb.tag_get_handle('GRAVITY'), 0, flat=True)[0]
        self.volume_tag = mb.tag_get_handle('VOLUME')
        self.sat_tag = mb.tag_get_handle('SAT')
        self.fw_tag = mb.tag_get_handle('FW')
        self.lamb_w_tag = mb.tag_get_handle('LAMB_W')
        self.lamb_o_tag = mb.tag_get_handle('LAMB_O')
        self.lbt_tag = mb.tag_get_handle('LBT')
        self.keq_tag = mb.tag_get_handle('K_EQ')
        self.mobi_in_faces_tag = mb.tag_get_handle('MOBI_IN_FACES')
        self.fw_in_faces_tag = mb.tag_get_handle('FW_IN_FACES')
        self.phi_tag = mb.tag_get_handle('PHI')
        self.total_flux_tag = mb.tag_get_handle('TOTAL_FLUX')
        self.flux_w_tag = mb.tag_get_handle('FLUX_W')
        self.flux_in_faces_tag = mb.tag_get_handle('FLUX_IN_FACES')
        self.wells_injector = mb.tag_get_data(mb.tag_get_handle('WELLS_INJECTOR'), 0, flat=True)
        self.wells_injector = mb.get_entities_by_handle(self.wells_injector[0])
        self.wells_producer = mb.tag_get_data(mb.tag_get_handle('WELLS_PRODUCER'), 0, flat=True)
        self.wells_producer = mb.get_entities_by_handle(self.wells_producer[0])
        self.s_grav_tag = mb.tag_get_handle('S_GRAV')
        self.s_grav_volume_tag = mb.tag_get_handle('S_GRAV_VOLUME', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        self.cent_tag = mb.tag_get_handle('CENT')
        self.dfds_tag = mb.tag_get_handle('DFDS')
        self.finos_tag = mb.tag_get_handle('finos')
        self.pf_tag = mb.tag_get_handle('PF')
        self.mb = mb
        self.mtu = mtu
        self.gama = self.gama_w + self.gama_o
        self.fimin = mb.tag_get_data(self.phi_tag, all_volumes, flat=True).min()
        self.Vmin = mb.tag_get_data(self.volume_tag, all_volumes, flat=True).min()
        historico = [np.array(['vpi','tempo', 'prod_agua', 'prod_oleo', 'wor'])]
        np.save('historico', historico)
        self.delta_t = 0.0
        self.V_total = mb.tag_get_data(self.volume_tag, all_volumes, flat=True)
        self.V_total = float((self.V_total*mb.tag_get_data(self.phi_tag, all_volumes, flat=True)).sum())
        self.vpi = 0.0

    def calculate_total_flux(self, volumes, faces):
        mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        map_local = dict(zip(volumes, range(len(volumes))))

        fluxos = np.zeros(len(volumes))
        fluxos_w = fluxos.copy()
        flux_in_faces = np.zeros(len(faces))
        fluxo_grav_volumes = np.zeros(len(volumes))


        for i, face in enumerate(faces):
            elems = self.mb.get_adjacencies(face, 3)
            mobi = mobi_in_faces[i]
            s_grav = s_gravs_faces[i]
            fw = fws_faces[i]
            ps = self.mb.tag_get_data(self.pf_tag, elems, flat=True)
            flux = (ps[1] - ps[0])*mobi
            id0 = map_local[elems[0]]
            id1 = map_local[elems[1]]
            if self.gravity == True:
                flux += s_grav
                fluxo_grav_volumes[id0] += s_grav
                fluxo_grav_volumes[id1] -= s_grav
            fluxos[id0] += flux
            fluxos_w[id0] += flux*fw
            fluxos[id1] -= flux
            fluxos_w[id1] -= flux*fw
            flux_in_faces[i] = flux

        self.mb.tag_set_data(self.total_flux_tag, volumes, fluxos)
        self.mb.tag_set_data(self.flux_w_tag, volumes, fluxos_w)
        self.mb.tag_set_data(self.flux_in_faces_tag, faces, flux_in_faces)
        self.mb.tag_set_data(self.s_grav_volume_tag, volumes, fluxo_grav_volumes)

    def calculate_sat(self, volumes, loop):
        """
        calcula a saturacao do passo de tempo corrente
        """

        t1 = time.time()
        lim = 1e-4
        all_qw = self.mb.tag_get_data(self.flux_w_tag, volumes, flat=True)
        all_fis = self.mb.tag_get_data(self.phi_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        # all_Vs = self.mb.tag_get_data(self.volume_tag, volumes, flat=True)

        sats_2 = np.zeros(len(volumes))

        for i, volume in enumerate(volumes):
            # gid = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            sat1 = all_sats[i]
            if volume in self.wells_injector:
                sats_2[i] = sat1
                continue
            qw = all_qw[i]

            if abs(qw) < lim:
                sats_2[i] = sat1
                continue
            elif qw < 0.0:
                print('qw < 0')
                print(qw)
                print('i')
                print(i)
                print('loop')
                print(loop)
                print('\n')
                import pdb; pdb.set_trace()
            else:
                pass

            # if self.loop > 1:
            #     import pdb; pdb.set_trace()
            fi = all_fis[i]
            # sat = sat1 + qw*(delta_t/(fi*V))
            sat = sat1 + qw*(self.delta_t/(self.fimin*self.Vmin))
            if sat1 > sat:
                print('erro na saturacao')
                print('sat1 > sat')
                import pdb; pdb.set_trace()
            elif sat > 0.8:
                #sat = 1 - self.Sor
                print("Sat > 1")
                print(sat)
                print('i')
                print(i)
                print('loop')
                print(loop)
                print('\n')
                import pdb; pdb.set_trace()
                sat = 0.8

            #elif sat < 0 or sat > (1 - self.Sor):
            elif sat < 0 or sat > 1:
                print('Erro: saturacao invalida')
                print('Saturacao: {0}'.format(sat))
                print('Saturacao anterior: {0}'.format(sat1))
                print('i: {0}'.format(i))
                print('fi: {0}'.format(fi))
                # print('V: {0}'.format(V))
                print('delta_t: {0}'.format(delta_t))
                print('loop: {0}'.format(loop))
                import pdb; pdb.set_trace()

                sys.exit(0)

            else:
                pass

            sats_2[i] = sat

        t2 = time.time()
        print('tempo calculo saturacao loop_{0}: {1}'.format(loop, t2-t1))
        self.mb.tag_set_data(self.sat_tag, volumes, sats_2)

    def cfl(self, faces_in):
        """
        cfl usando fluxo maximo
        """

        cfl = 0.5
        qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, faces_in, flat=True)).max()
        dfdsmax = self.mb.tag_get_data(self.dfds_tag, faces_in, flat=True).max()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()

        self.delta_t = cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        self.vpi += vpi

    def get_hist(self, t):
        flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True)
        fws = self.mb.tag_get_data(self.fw_tag, self.wells_producer, flat=True)

        qw = (flux_total_prod*fws).sum()*self.delta_t
        qo = (flux_total_prod.sum() - qw)*self.delta_t
        wor = qw/float(qo)


        hist = np.array([self.vpi, t, qw, qo, wor])
        historico = np.load('historico.npy')
        historico = np.append(historico, hist)
        np.save('historico', historico)

    def pol_interp(self, S):
        # S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
        # krw = (S_temp)**(self.nw)
        # kro = (1 - S_temp)**(self.no)
        if S > (1 - self.Sor):
            krw = 1.0
            kro = 0.0
        elif S < self.Swc:
            krw = 0.0
            kro = 1.0
        else:
            krw = ((S - self.Swc)/float(1 - self.Swc - self.Sor))**(self.nw)
            kro = ((1 - S - self.Swc)/float(1 - self.Swc - self.Sor))**(self.no)

        return krw, kro

    def set_lamb(self, all_volumes):
        """
        seta o lambda
        """
        all_sats = self.mb.tag_get_data(self.sat_tag, all_volumes, flat=True)
        all_lamb_w = np.zeros(len(all_volumes))
        all_lamb_o = all_lamb_w.copy()
        all_lbt = all_lamb_w.copy()
        all_fw = all_lamb_w.copy()

        for i, sat in enumerate(all_sats):
            # volume = all_volumes[i]
            krw, kro = self.pol_interp(sat)
            # lamb_w = krw/mi_w
            # lamb_o = kro/mi_o
            # lbt = lamb_w + lamb_o
            # fw = lamb_w/float(lbt)
            # all_fw[i] = fw
            # all_lamb_w[i] = lamb_w
            # all_lamb_o[i] = lamb_o
            # all_lbt[i] = lbt
            all_lamb_w[i] = krw/self.mi_w
            all_lamb_o[i] = kro/self.mi_o
            all_lbt[i] = all_lamb_o[i] + all_lamb_w[i]
            all_fw[i] = all_lamb_w[i]/float(all_lbt[i])

        self.mb.tag_set_data(self.lamb_w_tag, all_volumes, all_lamb_w)
        self.mb.tag_set_data(self.lamb_o_tag, all_volumes, all_lamb_o)
        self.mb.tag_set_data(self.lbt_tag, all_volumes, all_lbt)
        self.mb.tag_set_data(self.fw_tag, all_volumes, all_fw)

    def set_mobi_faces(self, volumes, faces):
        lim = 1e-5
        lim_sat = 0.001

        map_volumes = dict(zip(volumes, range(len(volumes))))
        all_lbt = self.mb.tag_get_data(self.lbt_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        all_fws = self.mb.tag_get_data(self.fw_tag, volumes, flat=True)
        all_centroids = self.mb.tag_get_data(self.cent_tag, volumes)

        all_flux_in_faces = self.mb.tag_get_data(self.flux_in_faces_tag, faces, flat=True)
        all_keqs = self.mb.tag_get_data(self.keq_tag, faces, flat=True)
        all_mobi_in_faces = np.zeros(len(faces))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()

        for i, face in enumerate(faces):
            elems = self.mb.get_adjacencies(face, 3)
            # lbt0 = self.mb.tag_get_data(self.lbt_tag, elems[0], flat=True)[0]
            # lbt1 = self.mb.tag_get_data(self.lbt_tag, elems[1], flat=True)[0]
            id0 = map_volumes[elems[0]]
            id1 = map_volumes[elems[1]]
            lbt0 = all_lbt[id0]
            lbt1 = all_lbt[id1]
            fw0 = all_fws[id0]
            fw1 = all_fws[id1]
            sat0 = all_sats[id0]
            sat1 = all_sats[id1]
            if abs(sat0 - sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))

            keq = all_keqs[i]
            flux_in_face = all_flux_in_faces[i]
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = keq*lbt0
                all_fw_in_face[i] = fw0
                # continue
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = keq*lbt1
                all_fw_in_face[i] = fw1
                # continue
            elif flux_in_face < 0:
                all_mobi_in_faces[i] = keq*(lbt0)
                all_fw_in_face[i] = fw0
            else:
                all_mobi_in_faces[i] = keq*(lbt1)
                all_fw_in_face[i] = fw1

            all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])

        self.mb.tag_set_data(self.mobi_in_faces_tag, faces, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, faces, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, faces, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, faces, all_dfds)

    def get_Tf_and_b(self, all_volumes, map_volumes, faces_in, dict_tags):
        lines_tf = []
        cols_tf = []
        data_tf = []
        lines_ttf = []
        cols_ttf = []
        data_ttf = []

        all_s_gravs = self.mb.tag_get_data(dict_tags['S_GRAV'], faces_in, flat=True)
        s_grav = np.zeros(len(all_volumes))
        all_mobis = self.mb.tag_get_data(self.mobi_in_faces_tag, faces_in, flat=True)

        for i, f in enumerate(faces_in):
            keq = all_mobis[i]
            adjs = self.mb.get_adjacencies(f, 3)
            id_0 = map_volumes[adjs[0]]
            id_1 = map_volumes[adjs[1]]
            # Gid_1=all_ids_reord[id_0]
            # Gid_2=all_ids_reord[id_1]
            Gid_1 = id_0
            Gid_2 = id_1
            lines_tf.append(Gid_1)
            cols_tf.append(Gid_2)
            data_tf.append(keq)

            lines_tf.append(Gid_2)
            cols_tf.append(Gid_1)
            data_tf.append(keq)

            if Gid_1 in lines_ttf:
                index = lines_ttf.index(Gid_1)
                data_ttf[index] -= keq
            else:
                lines_ttf.append(Gid_1)
                cols_ttf.append(Gid_1)
                data_ttf.append(-keq)

            if Gid_2 in lines_ttf:
                index = lines_ttf.index(Gid_2)
                data_ttf[index] -= keq
            else:
                lines_ttf.append(Gid_2)
                cols_ttf.append(Gid_2)
                data_ttf.append(-keq)

            flux_grav = -all_s_gravs[i]
            s_grav[id_0] += flux_grav
            s_grav[id_1] -= flux_grav

        lines_tf += lines_ttf
        cols_tf += cols_ttf
        data_tf += data_ttf


        n = len(map_volumes.keys())
        Tf = sp.csc_matrix((data_tf,(lines_tf,cols_tf)),shape=(n, n))
        Tf = Tf.tolil()
        if self.gravity == False:
            s_grav = np.zeros(n)

        return Tf, s_grav

    def solution_PF(self, all_volumes, map_volumes, faces_in, dict_tags):
        Tf, b = self.get_Tf_and_b(all_volumes, map_volumes, faces_in, dict_tags)
        volumes_d = self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['P']]), np.array([None]))
        volumes_n = self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['Q']]), np.array([None]))
        map_values = dict(zip(volumes_d, self.mb.tag_get_data(dict_tags['P'], volumes_d, flat=True)))
        Tf, b = oth.set_boundary_dirichlet_matrix(map_volumes, map_values, b, Tf)
        values_n = self.mb.tag_get_data(dict_tags['Q'], volumes_n, flat=True)
        map_values = dict(zip(volumes_n, values_n))
        b = oth.set_boundary_neumann(map_volumes, map_values, b)
        x = oth.get_solution(Tf, b)
        self.mb.tag_set_data(self.pf_tag, all_volumes, x)
        np.save('pf', x)

    def set_Pf(self, all_volumes):
        pf = np.load('pf.npy')
        self.mb.tag_set_data(self.pf_tag, all_volumes, pf)
