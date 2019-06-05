import numpy as np
from pymoab import core, types, rng, topo_util, skinner
import os
import sys
import io
import yaml
import scipy.sparse as sp
import time
from processor import conversao as conv
from utils.others_utils import OtherUtils as oth
import pdb
#

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
fly_bif_mult_dir = os.path.join(bifasico_dir, 'sol_multiescala')

import importlib.machinery
# loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
# oth = loader.load_module('others_utils').OtherUtils

# mi_w = 1.0
# mi_o = 1.2
# gama_w = 10.0
# gama_o = 9.0
# gama = gama_w + gama_o
# Sor = 0.2
# Swc = 0.2
# nw = 2
# no = 2
# tz = 100
# gravity = False
# V = 1.0

class bifasico:
    def __init__(self, mb, mtu, all_volumes, data_loaded):

        self.cfl_ini = 0.5
        self.delta_t_min = 100000
        self.perm_tag = mb.tag_get_handle('PERM')
        # self.mi_w = mb.tag_get_data(mb.tag_get_handle('MI_W'), 0, flat=True)[0]
        self.mi_w = float(data_loaded['dados_bifasico']['mi_w'])
        # self.mi_o = mb.tag_get_data(mb.tag_get_handle('MI_O'), 0, flat=True)[0]
        self.mi_o = float(data_loaded['dados_bifasico']['mi_o'])
        # self.gama_w = mb.tag_get_data(mb.tag_get_handle('GAMA_W'), 0, flat=True)[0]
        self.gama_w = float(data_loaded['dados_bifasico']['gama_w'])
        # self.gama_o = mb.tag_get_data(mb.tag_get_handle('GAMA_O'), 0, flat=True)[0]
        self.gama_o = float(data_loaded['dados_bifasico']['gama_o'])
        # self.Sor = mb.tag_get_data(mb.tag_get_handle('SOR'), 0, flat=True)[0]
        self.Sor = float(data_loaded['dados_bifasico']['Sor'])
        # self.Swc = mb.tag_get_data(mb.tag_get_handle('SWC'), 0, flat=True)[0]
        self.Swc = float(data_loaded['dados_bifasico']['Swc'])
        # self.nw = mb.tag_get_data(mb.tag_get_handle('NW'), 0, flat=True)[0]
        self.nw = float(data_loaded['dados_bifasico']['nwater'])
        # self.no = mb.tag_get_data(mb.tag_get_handle('NO'), 0, flat=True)[0]
        self.no = float(data_loaded['dados_bifasico']['noil'])
        # self.tz = mb.tag_get_data(mb.tag_get_handle('TZ'), 0, flat=True)[0]
        self.tz = 1.0
        # self.loops = mb.tag_get_data(mb.tag_get_handle('LOOPS'), 0, flat=True)[0]
        self.loops = int(data_loaded['dados_bifasico']['loops'])
        # self.total_time = mb.tag_get_data(mb.tag_get_handle('TOTAL_TIME'), 0, flat=True)[0]
        self.total_time = float(data_loaded['dados_bifasico']['total_time'])
        # self.gravity = mb.tag_get_data(mb.tag_get_handle('GRAVITY'), 0, flat=True)[0]
        self.gravity = data_loaded['gravity']
        self.volume_tag = mb.tag_get_handle('VOLUME')
        self.sat_tag = mb.tag_get_handle('SAT')
        self.sat_last_tag = mb.tag_get_handle('SAT_LAST')
        # self.fw_tag = mb.tag_get_handle('FW')
        self.fw_tag = mb.tag_get_handle('FW')
        # self.lamb_w_tag = mb.tag_get_handle('LAMB_W')
        self.lamb_w_tag = mb.tag_get_handle('LAMB_W')
        # self.lamb_o_tag = mb.tag_get_handle('LAMB_O')
        self.lamb_o_tag = mb.tag_get_handle('LAMB_O')
        # self.lbt_tag = mb.tag_get_handle('LBT')
        self.lbt_tag = mb.tag_get_handle('LBT')
        self.keq_tag = mb.tag_get_handle('K_EQ')
        self.kharm_tag = mb.tag_get_handle('KHARM')
        # self.mobi_in_faces_tag = mb.tag_get_handle('MOBI_IN_FACES')
        self.mobi_in_faces_tag = mb.tag_get_handle('MOBI_IN_FACES')
        # self.fw_in_faces_tag = mb.tag_get_handle('FW_IN_FACES')
        self.fw_in_faces_tag = mb.tag_get_handle('FW_IN_FACES')
        self.phi_tag = mb.tag_get_handle('PHI')
        # self.total_flux_tag = mb.tag_get_handle('TOTAL_FLUX')
        self.total_flux_tag = mb.tag_get_handle('TOTAL_FLUX')
        # self.flux_w_tag = mb.tag_get_handle('FLUX_W')
        self.flux_w_tag = mb.tag_get_handle('FLUX_W')
        # self.flux_in_faces_tag = mb.tag_get_handle('FLUX_IN_FACES')
        self.flux_in_faces_tag = mb.tag_get_handle('FLUX_IN_FACES')
        self.wells_injector = mb.tag_get_data(mb.tag_get_handle('WELLS_INJECTOR'), 0, flat=True)
        self.wells_injector = mb.get_entities_by_handle(self.wells_injector[0])
        self.wells_producer = mb.tag_get_data(mb.tag_get_handle('WELLS_PRODUCER'), 0, flat=True)
        self.wells_producer = mb.get_entities_by_handle(self.wells_producer[0])
        # self.kdif_tag = mb.tag_get_handle('KDIF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        # self.s_grav_tag = mb.tag_get_handle('S_GRAV')
        self.s_grav_tag = mb.tag_get_handle('S_GRAV')
        self.cent_tag = mb.tag_get_handle('CENT')
        # self.dfds_tag = mb.tag_get_handle('DFDS')
        self.dfds_tag = mb.tag_get_handle('DFDS')
        self.finos_tag = mb.tag_get_handle('finos')
        self.gamav_tag = mb.tag_get_handle('GAMAV')
        self.gamaf_tag = mb.tag_get_handle('GAMAF')
        self.map_all_volumes = dict(zip(all_volumes, range(len(all_volumes))))
        self.all_centroids = mb.tag_get_data(self.cent_tag, all_volumes)
        self.mb = mb
        self.mtu = mtu
        self.gama = self.gama_w + self.gama_o
        phis = mb.tag_get_data(self.phi_tag, all_volumes, flat=True)
        bb = np.nonzero(phis)[0]
        phis = phis[bb]
        self.fimin = phis.min()
        v0 = all_volumes[0]
        points = self.mtu.get_bridge_adjacencies(v0, 3, 0)
        # coords = (self.k_pe_m)*self.mb.get_coords(points).reshape(len(points), 3)
        coords = self.mb.get_coords(points).reshape(len(points), 3)
        maxs = coords.max(axis=0)
        mins = coords.min(axis=0)
        hs = maxs - mins
        # converter pe para metro
        # hs[0] = conv.pe_to_m(hs[0])
        # hs[1] = conv.pe_to_m(hs[1])
        # hs[1] = conv.pe_to_m(hs[1])

        self.hs = hs
        vol = hs[0]*hs[1]*hs[2]
        # self.Areas = (self.k_pe_m**2)*np.array([hs[1]*hs[2], hs[0]*hs[2], hs[0]*hs[1]])
        self.Areas = np.array([hs[1]*hs[2], hs[0]*hs[2], hs[0]*hs[1]])
        self.mb.tag_set_data(self.volume_tag, all_volumes, np.repeat(vol, len(all_volumes)))
        self.Vmin = vol
        historico = np.array(['vpi', 'tempo', 'prod_agua', 'prod_oleo', 'wor', 'dt'])
        np.save('historico', historico)
        self.V_total = mb.tag_get_data(self.volume_tag, all_volumes, flat=True)
        self.V_total = float((self.V_total*mb.tag_get_data(self.phi_tag, all_volumes, flat=True)).sum())
        self.vpi = 0.0
        d1_tag = self.mb.tag_get_handle('d1')
        self.vertices = set(self.mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([d1_tag]), np.array([3])))

    def calc_cfl_dep0(self, all_faces_in):
        """
        cfl usando fluxo maximo
        """
        self.cfl = self.cfl_ini
        self.all_faces_in = all_faces_in
        qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, all_faces_in, flat=True)).max()
        dfdsmax = self.mb.tag_get_data(self.dfds_tag, all_faces_in, flat=True).max()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.flux_total_producao = self.flux_total_prod
        self.delta_t = self.cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        # self.delta_t = self.cfl*(fi*V)/float(q*dfds)
        # vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        # self.vpi += vpi

    def calc_cfl(self, all_faces_in):
        """
        cfl usando fluxo maximo
        """
        lim_sup = 1e5
        self.cfl = self.cfl_ini
        self.all_faces_in = all_faces_in
        qs = self.mb.tag_get_data(self.flux_in_faces_tag, all_faces_in, flat=True)
        dfdss = self.mb.tag_get_data(self.dfds_tag, all_faces_in, flat=True)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in all_faces_in]
        all_volumes = self.mtu.get_bridge_adjacencies(all_faces_in, 2, 3)
        delta_ts = np.zeros(len(all_volumes))
        faces_volumes = [self.mtu.get_bridge_adjacencies(v, 3, 2) for v in all_volumes]
        phis = self.mb.tag_get_data(self.phi_tag, all_volumes, flat=True)
        Vs = self.mb.tag_get_data(self.volume_tag, all_volumes, flat=True)
        map_faces = dict(zip(all_faces_in, range(len(all_faces_in))))

        # self.delta_t = self.cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)

        for i, v in enumerate(all_volumes):
            V = Vs[i]
            phi = phis[i]
            if phi == 0:
                delta_ts[i] = lim_sup
                continue
            faces = faces_volumes[i]
            faces = rng.intersect(all_faces_in, faces)
            ids_faces = [map_faces[f] for f in faces]
            q_faces = qs[ids_faces]
            dfdss_faces = dfdss[ids_faces]
            qmax = q_faces.max()
            ind = np.where(q_faces == qmax)[0]
            dfds = dfdss_faces[ind][0]
            if dfds == 0.0:
                dt1 = lim_sup
            else:
                qmax = abs(qmax)
                dt1 = self.cfl*(phi*V)/float(qmax*dfds)
                if dt1 < 0:
                    print('erro')
                    import pdb; pdb.set_trace()

            dfds_max = dfdss_faces.max()
            if dfds_max == 0:
                dt2 = dt1
            else:
                ind = np.where(dfdss_faces == dfds_max)[0]
                q2 = abs(q_faces[ind][0])
                dt2 = self.cfl*(phi*V)/float(q2*dfds_max)
                if dt2 < 0:
                    print('erro')
                    import pdb; pdb.set_trace()

            delta_ts[i] = min([dt1, dt2])
            if delta_ts[i] > self.delta_t_min:
                delta_ts[i] = self.delta_t_min


        self.delta_t = delta_ts.min()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.flux_total_producao = self.flux_total_prod

        # self.delta_t = self.cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        # vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        # self.vpi += vpi

    def rec_cfl_dep0(self, cfl):

        cfl = 0.5*cfl
        print('novo cfl', cfl)
        qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, self.all_faces_in, flat=True)).max()
        dfdsmax = self.mb.tag_get_data(self.dfds_tag, self.all_faces_in, flat=True).max()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.delta_t = cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        # vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        # self.vpi += vpi
        return cfl

    def rec_cfl_dep1(self, cfl):

        cfl = 0.5*cfl
        self.cfl = cfl
        print('novo cfl', cfl)
        lim_sup = 1e5
        qs = self.mb.tag_get_data(self.flux_in_faces_tag, self.all_faces_in, flat=True)
        dfdss = self.mb.tag_get_data(self.dfds_tag, self.all_faces_in, flat=True)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in self.all_faces_in]
        all_volumes = self.mtu.get_bridge_adjacencies(self.all_faces_in, 2, 3)
        delta_ts = np.zeros(len(all_volumes))
        faces_volumes = [self.mtu.get_bridge_adjacencies(v, 3, 2) for v in all_volumes]
        phis = self.mb.tag_get_data(self.phi_tag, all_volumes, flat=True)
        Vs = self.mb.tag_get_data(self.volume_tag, all_volumes, flat=True)
        map_faces = dict(zip(self.all_faces_in, range(len(self.all_faces_in))))

        # self.delta_t = self.cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)

        for i, v in enumerate(all_volumes):

            V = Vs[i]
            phi = phis[i]
            if phi == 0:
                delta_ts[i] = lim_sup
                continue
            faces = faces_volumes[i]
            faces = rng.intersect(self.all_faces_in, faces)
            ids_faces = [map_faces[f] for f in faces]
            q_faces = qs[ids_faces]
            dfdss_faces = dfdss[ids_faces]
            qmax = q_faces.max()
            ind = np.where(q_faces == qmax)[0]
            dfds = dfdss_faces[ind][0]
            if dfds == 0.0:
                dt1 = lim_sup
            else:
                qmax = abs(qmax)
                dt1 = self.cfl*(phi*V)/float(qmax*dfds)
                if dt1 < 0:
                    print('erro')
                    import pdb; pdb.set_trace()

            dfds_max = dfdss_faces.max()
            if dfds_max == 0:
                dt2 = dt1
            else:
                ind = np.where(dfdss_faces == dfds_max)[0]
                q2 = abs(q_faces[ind][0])
                dt2 = self.cfl*(phi*V)/float(q2*dfds_max)
                if dt2 < 0:
                    print('erro')
                    import pdb; pdb.set_trace()

            delta_ts[i] = min([dt1, dt2])
            if delta_ts[i] > self.delta_t_min:
                delta_ts[i] = self.delta_t_min


        self.delta_t = delta_ts.min()
        print(f'novo delta_t: {self.delta_t}')
        print('\n')

        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.flux_total_producao = self.flux_total_prod
        return cfl

    def rec_cfl(self, cfl):

        k = 0.5
        cfl = k*cfl
        print('novo cfl', cfl)
        # qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, self.all_faces_in, flat=True)).max()
        # dfdsmax = self.mb.tag_get_data(self.dfds_tag, self.all_faces_in, flat=True).max()
        # self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        # self.delta_t = cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        self.delta_t *= k
        # vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        # self.vpi += vpi
        return cfl

    def set_sat_in(self, all_volumes):
        """
        seta a saturacao inicial
        """

        self.mb.tag_set_data(self.sat_tag, all_volumes, np.repeat(0.2, len(all_volumes)))
        self.mb.tag_set_data(self.sat_tag, self.wells_injector, np.repeat(1.0, len(self.wells_injector)))

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
        all_gamav = all_lamb_w.copy()

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
            gama = (self.gama_w*all_lamb_w[i] + self.gama_o*all_lamb_o[i])/(all_lbt[i])
            all_gamav[i] = gama

        self.mb.tag_set_data(self.lamb_w_tag, all_volumes, all_lamb_w)
        self.mb.tag_set_data(self.lamb_o_tag, all_volumes, all_lamb_o)
        self.mb.tag_set_data(self.lbt_tag, all_volumes, all_lbt)
        self.mb.tag_set_data(self.fw_tag, all_volumes, all_fw)
        self.mb.tag_set_data(self.gamav_tag, all_volumes, all_gamav)

    def set_mobi_faces_ini_dep0(self, all_volumes, all_faces_in):
        lim = 1e-5

        all_lbt = self.mb.tag_get_data(self.lbt_tag, all_volumes, flat=True)
        map_lbt = dict(zip(all_volumes, all_lbt))
        del all_lbt
        all_centroids = np.array([self.mtu.get_average_position([v]) for v in all_volumes])
        for i, v in enumerate(all_volumes):
            self.mb.tag_set_data(self.cent_tag, v, all_centroids[i])
        map_centroids = dict(zip(all_volumes, all_centroids))
        del all_centroids
        all_fw = self.mb.tag_get_data(self.fw_tag, all_volumes, flat=True)
        map_fw = dict(zip(all_volumes, all_fw))
        del all_fw
        all_sats = self.mb.tag_get_data(self.sat_tag, all_volumes, flat=True)
        map_sat = dict(zip(all_volumes, all_sats))
        del all_sats
        all_keqs = self.mb.tag_get_data(self.keq_tag, all_faces_in, flat=True)
        all_mobi_in_faces = np.zeros(len(all_faces_in))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        Adjs = [self.mb.get_adjacencies(face, 3) for face in all_faces_in]

        for i, face in enumerate(all_faces_in):
            elems = Adjs[i]
            lbt0 = map_lbt[elems[0]]
            lbt1 = map_lbt[elems[1]]
            fw0 = map_fw[elems[0]]
            fw1 = map_fw[elems[1]]
            sat0 = map_sat[elems[0]]
            sat1 = map_sat[elems[1]]
            if abs(sat0-sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            keq = all_keqs[i]
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = keq*lbt0
                all_fw_in_face[i] = fw0
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = keq*lbt1
                all_fw_in_face[i] = fw1
            else:
                all_mobi_in_faces[i] = keq*(lbt0 + lbt1)/2.0
                all_fw_in_face[i] = (fw0 + fw1)/2.0

            all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(map_centroids[elems[1]][2] - map_centroids[elems[0]][2])

        self.mb.tag_set_data(self.mobi_in_faces_tag, all_faces_in, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, all_faces_in, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, all_faces_in, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, all_faces_in, all_dfds)

    def set_mobi_faces_ini_dep1(self, all_volumes, all_faces_in):
        lim = 1e-5

        map_volumes = dict(zip(all_volumes, range(len(all_volumes))))

        all_lbt = self.mb.tag_get_data(self.lbt_tag, all_volumes, flat=True)
        # all_centroids = np.array([self.mtu.get_average_position([v]) for v in all_volumes])
        # for i, v in enumerate(all_volumes):
        #     self.mb.tag_set_data(self.cent_tag, v, all_centroids[i])
        all_centroids = self.all_centroids
        all_fw = self.mb.tag_get_data(self.fw_tag, all_volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, all_volumes, flat=True)
        all_ks = self.mb.tag_get_data(self.perm_tag, all_volumes)
        all_gamav = self.mb.tag_get_data(self.gamav_tag, all_volumes, flat=True)
        all_kdif = self.mb.tag_get_data(self.k_eq_tag, all_faces_in, flat=True)

        all_mobi_in_faces = np.zeros(len(all_faces_in))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        Adjs = [self.mb.get_adjacencies(face, 3) for face in all_faces_in]
        self.Adjs = Adjs

        for i, face in enumerate(all_faces_in):
            elems = Adjs[i]
            id0 = map_volumes[elems[0]]
            id1 = map_volumes[elems[1]]
            lbt0 = all_lbt[id0]
            lbt1 = all_lbt[id1]
            fw0 = all_fw[id0]
            fw1 = all_fw[id1]
            sat0 = all_sats[id0]
            sat1 = all_sats[id1]
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            k0 = all_ks[id0].reshape([3,3])
            k1 = all_ks[id1].reshape([3,3])
            direction = all_centroids[id1] - all_centroids[id0]
            norma = np.linalg.norm(direction)
            uni = np.absolute(direction/norma)
            k0 = np.dot(np.dot(k0, uni), uni)
            k1 = np.dot(np.dot(k1, uni), uni)
            h = np.dot(self.hs, uni)
            area = np.dot(self.Areas, uni)
            if abs(sat0-sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = k0*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = k1*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
            else:
                all_mobi_in_faces[i] = ((2*k0*k1)/(k0+k1))*(lbt0 + lbt1)/2.0
                # all_fw_in_face[i] = (fw0 + fw1)/2.0
                all_fw_in_face[i] = 0.0
                gamaf = (gama0 + gama1)/2.0
            all_mobi_in_faces[i] *= area/h
            all_kdif[i] = all_kdif[i]*((lbt0 + lbt1)/2)
            # all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf

        # self.mb.tag_set_data(self.mobi_in_faces_tag, all_faces_in, all_mobi_in_faces)
        self.mb.tag_set_data(self.mobi_in_faces_tag, all_faces_in, all_kdif)
        self.mb.tag_set_data(self.s_grav_tag, all_faces_in, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, all_faces_in, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, all_faces_in, all_dfds)
        self.mb.tag_set_data(self.gamaf_tag, all_faces_in, all_gamaf)
        self.mb.tag_set_data(self.kdif_tag, all_faces_in, all_kdif)

    def set_mobi_faces_ini(self, all_volumes, all_faces_in):
        lim = 1e-5

        map_volumes = dict(zip(all_volumes, range(len(all_volumes))))

        all_lbt = self.mb.tag_get_data(self.lbt_tag, all_volumes, flat=True)
        # all_centroids = np.array([self.mtu.get_average_position([v]) for v in all_volumes])
        # for i, v in enumerate(all_volumes):
        #     self.mb.tag_set_data(self.cent_tag, v, all_centroids[i])
        all_centroids = self.all_centroids
        all_fw = self.mb.tag_get_data(self.fw_tag, all_volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, all_volumes, flat=True)
        all_gamav = self.mb.tag_get_data(self.gamav_tag, all_volumes, flat=True)

        all_kharm = self.mb.tag_get_data(self.kharm_tag, all_faces_in, flat=True)

        all_mobi_in_faces = np.zeros(len(all_faces_in))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        # Adjs = [self.mb.get_adjacencies(face, 3) for face in all_faces_in]
        Adjs = self.Adjs

        for i, face in enumerate(all_faces_in):
            elems = Adjs[i]
            kharm = all_kharm[i]
            id0 = map_volumes[elems[0]]
            id1 = map_volumes[elems[1]]
            lbt0 = all_lbt[id0]
            lbt1 = all_lbt[id1]
            fw0 = all_fw[id0]
            fw1 = all_fw[id1]
            sat0 = all_sats[id0]
            sat1 = all_sats[id1]
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            if abs(sat0-sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = kharm*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = kharm*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
            else:
                all_mobi_in_faces[i] = kharm*(lbt0 + lbt1)/2.0
                all_fw_in_face[i] = (fw0 + fw1)/2.0
                gamaf = (gama0 + gama1)/2.0
            # all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf

        self.mb.tag_set_data(self.mobi_in_faces_tag, all_faces_in, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, all_faces_in, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, all_faces_in, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, all_faces_in, all_dfds)
        self.mb.tag_set_data(self.gamaf_tag, all_faces_in, all_gamaf)

    def set_mobi_faces_dep0(self, volumes, faces, finos0=None):

        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        # finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.15
        finos = self.mb.create_meshset()
        self.mb.tag_set_data(self.finos_tag, 0, finos)
        if finos0 == None:
            self.mb.add_entities(finos, self.wells_injector)
            self.mb.add_entities(finos, self.wells_producer)
        else:
            self.mb.add_entities(finos, finos0)

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
            if abs(sat0 - sat1) > lim_sat:
                self.mb.add_entities(finos, elems)

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
        # vols_finos = self.mb.get_entities_by_handle(finos)
        # self.mb.tag_set_data(finos_val, vols_finos, np.repeat(1.0, len(vols_finos)))

    def set_mobi_faces_dep1(self, volumes, faces, finos0=None):

        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        # finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.15
        finos = self.mb.create_meshset()
        self.mb.tag_set_data(self.finos_tag, 0, finos)
        if finos0 == None:
            self.mb.add_entities(finos, self.wells_injector)
            self.mb.add_entities(finos, self.wells_producer)
        else:
            self.mb.add_entities(finos, finos0)

        map_volumes = dict(zip(volumes, range(len(volumes))))
        all_lbt = self.mb.tag_get_data(self.lbt_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        all_fws = self.mb.tag_get_data(self.fw_tag, volumes, flat=True)
        all_gamav = self.mb.tag_get_data(self.gamav_tag, volumes, flat=True)
        # all_centroids = self.mb.tag_get_data(self.cent_tag, volumes)
        all_centroids = self.all_centroids
        all_ks = self.mb.tag_get_data(self.perm_tag, volumes)

        all_flux_in_faces = self.mb.tag_get_data(self.flux_in_faces_tag, faces, flat=True)
        # all_keqs = self.mb.tag_get_data(self.keq_tag, faces, flat=True)
        all_mobi_in_faces = np.zeros(len(faces))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]

        for i, face in enumerate(faces):
            # elems = self.mb.get_adjacencies(face, 3)
            elems = Adjs[i]
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
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            k0 = all_ks[id0].reshape([3,3])
            k1 = all_ks[id1].reshape([3,3])
            direction = all_centroids[id1] - all_centroids[id0]
            norma = np.linalg.norm(direction)
            uni = np.absolute(direction/norma)
            k0 = np.dot(np.dot(k0, uni), uni)
            k1 = np.dot(np.dot(k1, uni), uni)
            h = np.dot(self.hs, uni)
            area = np.dot(self.Areas, uni)

            if abs(sat0 - sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            if abs(sat0 - sat1) > lim_sat:
                self.mb.add_entities(finos, elems)

            flux_in_face = all_flux_in_faces[i]
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = k0*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
                # continue
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = k1*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
                # continue
            elif flux_in_face < 0:
                all_mobi_in_faces[i] = k0*(lbt0)
                all_fw_in_face[i] = fw0
                gamaf = gama0
            else:
                all_mobi_in_faces[i] = k1*(lbt1)
                all_fw_in_face[i] = fw1
                gamaf = gama1

            all_mobi_in_faces[i] *= area/h
            # all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf

        self.mb.tag_set_data(self.mobi_in_faces_tag, faces, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, faces, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, faces, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, faces, all_dfds)
        self.mb.tag_set_data(self.gamaf_tag, faces, all_gamaf)
        # vols_finos = self.mb.get_entities_by_handle(finos)
        # self.mb.tag_set_data(finos_val, vols_finos, np.repeat(1.0, len(vols_finos)))

    def set_mobi_faces_dep2(self, volumes, faces, finos0=None):

        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        # finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.15
        # finos = self.mb.create_meshset()
        # self.mb.tag_set_data(self.finos_tag, 0, finos)
        # if finos0 == None:
        #     self.mb.add_entities(finos, self.wells_injector)
        #     self.mb.add_entities(finos, self.wells_producer)
        # else:
        #     self.mb.add_entities(finos, finos0)

        map_volumes = dict(zip(volumes, range(len(volumes))))
        all_lbt = self.mb.tag_get_data(self.lbt_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        all_fws = self.mb.tag_get_data(self.fw_tag, volumes, flat=True)
        all_gamav = self.mb.tag_get_data(self.gamav_tag, volumes, flat=True)
        # all_centroids = self.mb.tag_get_data(self.cent_tag, volumes)
        all_centroids = self.all_centroids
        all_ks = self.mb.tag_get_data(self.perm_tag, volumes)

        all_flux_in_faces = self.mb.tag_get_data(self.flux_in_faces_tag, faces, flat=True)
        # all_keqs = self.mb.tag_get_data(self.keq_tag, faces, flat=True)
        all_mobi_in_faces = np.zeros(len(faces))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]

        for i, face in enumerate(faces):
            # elems = self.mb.get_adjacencies(face, 3)
            elems = Adjs[i]
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
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            k0 = all_ks[id0].reshape([3,3])
            k1 = all_ks[id1].reshape([3,3])
            direction = all_centroids[id1] - all_centroids[id0]
            norma = np.linalg.norm(direction)
            uni = np.absolute(direction/norma)
            k0 = np.dot(np.dot(k0, uni), uni)
            k1 = np.dot(np.dot(k1, uni), uni)
            h = np.dot(self.hs, uni)
            area = np.dot(self.Areas, uni)

            if abs(sat0 - sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            # if abs(sat0 - sat1) > lim_sat:
            #     self.mb.add_entities(finos, elems)

            flux_in_face = all_flux_in_faces[i]
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = k0*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
                # continue
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = k1*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
                continue
            elif flux_in_face < 0:
                all_mobi_in_faces[i] = k0*(lbt0)
                all_fw_in_face[i] = fw0
                gamaf = gama0
            else:
                all_mobi_in_faces[i] = k1*(lbt1)
                all_fw_in_face[i] = fw1
                gamaf = gama1

            all_mobi_in_faces[i] *= area/h
            # all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf

        self.mb.tag_set_data(self.mobi_in_faces_tag, faces, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, faces, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, faces, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, faces, all_dfds)
        self.mb.tag_set_data(self.gamaf_tag, faces, all_gamaf)

    def set_mobi_faces(self, volumes, faces, finos0=None):

        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        # finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.15
        # finos = self.mb.create_meshset()
        # self.mb.tag_set_data(self.finos_tag, 0, finos)
        # if finos0 == None:
        #     self.mb.add_entities(finos, self.wells_injector)
        #     self.mb.add_entities(finos, self.wells_producer)
        # else:
        #     self.mb.add_entities(finos, finos0)

        map_volumes = dict(zip(volumes, range(len(volumes))))
        all_lbt = self.mb.tag_get_data(self.lbt_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        all_fws = self.mb.tag_get_data(self.fw_tag, volumes, flat=True)
        all_gamav = self.mb.tag_get_data(self.gamav_tag, volumes, flat=True)
        # all_centroids = self.mb.tag_get_data(self.cent_tag, volumes)
        all_centroids = self.all_centroids
        # all_ks = self.mb.tag_get_data(self.perm_tag, volumes)

        all_flux_in_faces = self.mb.tag_get_data(self.flux_in_faces_tag, faces, flat=True)
        all_kharm = self.mb.tag_get_data(self.kharm_tag, faces, flat=True)
        # all_keqs = self.mb.tag_get_data(self.keq_tag, faces, flat=True)
        all_mobi_in_faces = np.zeros(len(faces))
        all_s_gravs = all_mobi_in_faces.copy()
        all_fw_in_face = all_mobi_in_faces.copy()
        all_dfds = all_mobi_in_faces.copy()
        all_gamaf = all_mobi_in_faces.copy()
        # Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]
        Adjs = self.Adjs

        for i, face in enumerate(faces):
            # elems = self.mb.get_adjacencies(face, 3)
            elems = Adjs[i]
            kharm = all_kharm[i]
            id0 = map_volumes[elems[0]]
            id1 = map_volumes[elems[1]]
            lbt0 = all_lbt[id0]
            lbt1 = all_lbt[id1]
            fw0 = all_fws[id0]
            fw1 = all_fws[id1]
            sat0 = all_sats[id0]
            sat1 = all_sats[id1]
            gama0 = all_gamav[id0]
            gama1 = all_gamav[id1]

            if abs(sat0 - sat1) < lim:
                all_dfds[i] = 0.0
            else:
                all_dfds[i] = abs((fw0 - fw1)/(sat0 - sat1))
            # if abs(sat0 - sat1) > lim_sat:
            #     self.mb.add_entities(finos, elems)

            flux_in_face = all_flux_in_faces[i]
            if elems[0] in self.wells_injector:
                all_mobi_in_faces[i] = kharm*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
                # continue
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = kharm*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1
                continue
            elif flux_in_face < 0:
                all_mobi_in_faces[i] = kharm*lbt0
                all_fw_in_face[i] = fw0
                gamaf = gama0
            else:
                all_mobi_in_faces[i] = kharm*lbt1
                all_fw_in_face[i] = fw1
                gamaf = gama1

            # all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_s_gravs[i] = gamaf*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])
            all_gamaf[i] = gamaf

        self.mb.tag_set_data(self.mobi_in_faces_tag, faces, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, faces, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, faces, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, faces, all_dfds)
        self.mb.tag_set_data(self.gamaf_tag, faces, all_gamaf)

        # all_flux_in_faces = self.mb.tag_get_data(self.flux_in_faces_tag, faces, flat=True)
        # fw_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        #

    def set_finos(self, finos0, meshsets_nv1):
        lim_sat = 0.1

        finos = self.mb.create_meshset()
        self.mb.tag_set_data(self.finos_tag, 0, finos)
        if finos0 == None:
            self.mb.add_entities(finos, self.wells_injector)
            self.mb.add_entities(finos, self.wells_producer)
        else:
            self.mb.add_entities(finos, finos0)

        for m in meshsets_nv1:
            elems = self.mb.get_entities_by_handle(m)
            sats = self.mb.tag_get_data(self.sat_tag, elems, flat=True)
            min_sat = sats.min()
            max_sat = sats.max()
            if max_sat - min_sat > lim_sat:
                self.mb.add_entities(finos, elems)

    def set_flux_pms_meshsets_dep0(self, volumes, faces, faces_boundary, pms_tag, pcorr_tag, pcorr2_tag=None):

        map_local = dict(zip(volumes, range(len(volumes))))
        mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)

        pcorrs = self.mb.tag_get_data(pcorr_tag, volumes, flat=True)
        fluxos = np.zeros(len(volumes))
        fluxos_w = fluxos.copy()
        flux_in_faces = np.zeros(len(faces))

        for i, face in enumerate(faces):
            elems = self.mb.get_adjacencies(face, 3)
            mobi = mobi_in_faces[i]
            s_grav = s_gravs_faces[i]
            fw = fws_faces[i]
            if face in faces_boundary:
                ps = self.mb.tag_get_data(pms_tag, elems, flat=True)
                flux = (ps[1] - ps[0])*mobi
                if self.gravity == True:
                    flux += s_grav
                if elems[0] in volumes:
                    local_id = map_local[elems[0]]
                    fluxos[local_id] += flux
                    fluxos_w[local_id] += flux*fw
                else:
                    local_id = map_local[elems[1]]
                    fluxos[local_id] -= flux
                    fluxos_w[local_id] -= flux*fw
                flux_in_faces[i] = flux

                continue

            local_id0 = map_local[elems[0]]
            local_id1 = map_local[elems[1]]
            p0 = pcorrs[local_id0]
            p1 = pcorrs[local_id1]
            flux = (p1 - p0)*mobi
            if self.gravity == True:
                flux += s_grav
            fluxos[local_id0] += flux
            fluxos_w[local_id0] += flux*fw
            fluxos[local_id1] -= flux
            fluxos_w[local_id1] -= flux*fw
            flux_in_faces[i] = flux

        self.mb.tag_set_data(self.total_flux_tag, volumes, fluxos)
        self.mb.tag_set_data(self.flux_w_tag, volumes, fluxos_w)
        self.mb.tag_set_data(self.flux_in_faces_tag, faces, flux_in_faces)

    def set_flux_pms_meshsets(self, volumes, faces, faces_boundary, pms_tag, pcorr_tag, pcorr2_tag=None):

          volumes_2 = self.mtu.get_bridge_adjacencies(volumes, 2, 3)
          map_local = dict(zip(volumes_2, range(len(volumes_2))))
          # vols_inter = rng.subtract(volumes_2, volumes)  # volumes na interface do primal que ficam por fora
          # vols3 = self.mtu.get_bridge_adjacencies(vols_inter, 2, 3)
          # vols3 = rng.intersect(volumes, vols3)
          # vols3 = rng.unite(vols3, vols_inter)
          vols3 = self.mtu.get_bridge_adjacencies(faces_boundary, 2, 3)
          # vols3: volumes separados pelas faces que ficam na interface do primal
          pms_vols3 = self.mb.tag_get_data(pms_tag, vols3, flat=True)
          map_pms_vols3 = dict(zip(vols3, pms_vols3))
          del pms_vols3

          mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
          fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
          if self.gravity:
              s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)
          else:
              s_gravs_faces = np.zeros(len(faces))

          pcorrs = self.mb.tag_get_data(pcorr_tag, volumes, flat=True)
          map_pcorrs = dict(zip(volumes, pcorrs))
          del pcorrs
          fluxos = np.zeros(len(volumes_2))
          fluxos_w = fluxos.copy()
          flux_in_faces = np.zeros(len(faces))
          Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]
          map_faces = dict(zip(faces, range(len(faces))))
          faces_in = rng.subtract(faces, faces_boundary)

          for face in faces_in:
              id_face = map_faces[face]
              elem0 = Adjs[id_face][0]
              elem1 = Adjs[id_face][1]
              id0 = map_local[elem0]
              id1 = map_local[elem1]
              mobi = mobi_in_faces[id_face]
              fw = fws_faces[id_face]
              s_grav = s_gravs_faces[id_face]
              p0 = map_pcorrs[elem0]
              p1 = map_pcorrs[elem1]
              flux = (p1-p0)*mobi + s_grav

              fluxos[id0] += flux
              fluxos_w[id0] += flux*fw
              fluxos[id1] -= flux
              fluxos_w[id1] -= flux*fw
              flux_in_faces[id_face] = flux

          for face in faces_boundary:
              id_face = map_faces[face]
              elem0 = Adjs[id_face][0]
              elem1 = Adjs[id_face][1]
              id0 = map_local[elem0]
              id1 = map_local[elem1]
              mobi = mobi_in_faces[id_face]
              fw = fws_faces[id_face]
              s_grav = s_gravs_faces[id_face]
              p0 = map_pms_vols3[elem0]
              p1 = map_pms_vols3[elem1]
              flux = (p1-p0)*mobi + s_grav

              fluxos[id0] += flux
              fluxos_w[id0] += flux*fw
              fluxos[id1] -= flux
              fluxos_w[id1] -= flux*fw
              flux_in_faces[id_face] = flux

          ids_volumes = [map_local[v] for v in volumes]
          fluxos = fluxos[ids_volumes]
          fluxos_w = fluxos_w[ids_volumes]

          self.mb.tag_set_data(self.total_flux_tag, volumes, fluxos)
          self.mb.tag_set_data(self.flux_w_tag, volumes, fluxos_w)
          self.mb.tag_set_data(self.flux_in_faces_tag, faces, flux_in_faces)

    def set_flux_pms_elems_nv0_dep(self, volumes, faces, pms_tag):

        mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        if self.gravity:
            s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        else:
            s_gravs_faces = np.zeros(len(faces))

        volumes_2 = self.mtu.get_bridge_adjacencies(volumes, 2, 3)
        map_local = dict(zip(volumes_2, range(len(volumes_2))))
        pmss = self.mb.tag_get_data(pms_tag, volumes_2, flat=True)
        fluxos = np.zeros(len(volumes_2))
        fluxos_w = fluxos.copy()
        flux_in_faces = np.zeros(len(faces))
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]

        for i, face in enumerate(faces):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id0 = map_local[elem0]
            id1 = map_local[elem1]
            mobi = mobi_in_faces[i]
            s_grav = s_gravs_faces[i]
            fw = fws_faces[i]
            flux = (pmss[id1] - pmss[id0])*mobi + s_grav
            fluxos[id0] += flux
            fluxos[id1] -= flux
            flux_in_faces[i] = flux

        ids_volumes = [map_local[v] for v in volumes]
        fluxos = fluxos[ids_volumes]
        fluxos_w = fluxos_w[ids_volumes]

        self.mb.tag_set_data(self.total_flux_tag, volumes, fluxos)
        self.mb.tag_set_data(self.flux_w_tag, volumes, fluxos_w)
        self.mb.tag_set_data(self.flux_in_faces_tag, faces, flux_in_faces)

    def set_flux_pms_elems_nv0(self, volumes, faces, pms_tag):

        mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        # mobi_in_faces = self.mb.tag_get_data(self.kdif_tag, faces, flat=True)
        fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        if self.gravity:
            s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        else:
            s_gravs_faces = np.zeros(len(faces))

        volumes_2 = self.mtu.get_bridge_adjacencies(volumes, 2, 3)
        map_local = dict(zip(volumes_2, range(len(volumes_2))))
        pmss = self.mb.tag_get_data(pms_tag, volumes_2, flat=True)
        fluxos = np.zeros(len(volumes_2))
        fluxos_w = fluxos.copy()
        flux_in_faces = np.zeros(len(faces))
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]

        for i, face in enumerate(faces):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id0 = map_local[elem0]
            id1 = map_local[elem1]
            mobi = mobi_in_faces[i]
            s_grav = s_gravs_faces[i]
            fw = fws_faces[i]
            flux = (pmss[id1] - pmss[id0])*mobi + s_grav
            fluxos[id0] += flux
            fluxos_w[id0] += flux*fw
            fluxos[id1] -= flux
            fluxos_w[id1] -= flux*fw
            flux_in_faces[i] = flux

        # pdb.set_trace()
        # tt = np.where(fluxos_w < 0)[0]
        # vols_tt = np.array(volumes_2)[tt]
        # vols_injector = np.array(self.wells_injector)
        # vols_producer = np.array(self.wells_producer)
        # map_vols_injector = [map_local[v] for v in vols_injector]
        # fl_injector = fluxos_w[map_vols_injector]
        # ff = np.setdiff1d(vols_tt, vols_injector)
        # pdb.set_trace()

        # ids_volumes = [map_local[v] for v in volumes]
        ids_volumes = list((map_local[v] for v in volumes))
        fluxos = fluxos[ids_volumes]
        fluxos_w = fluxos_w[ids_volumes]

        self.mb.tag_set_data(self.total_flux_tag, volumes, fluxos)
        self.mb.tag_set_data(self.flux_w_tag, volumes, fluxos_w)
        self.mb.tag_set_data(self.flux_in_faces_tag, faces, flux_in_faces)

    def calculate_sat(self, volumes, loop):
        """
        calcula a saturacao do passo de tempo corrente
        """
        delta_sat = 0.001
        lim_qw = 9e-8
        t1 = time.time()
        lim = 1e-4
        all_qw = self.mb.tag_get_data(self.flux_w_tag, volumes, flat=True)

        # inds = np.where(all_qw<0)[0]
        # volumes_inds = rng.Range(np.array(volumes)[inds])
        # volumes_inds = rng.subtract(volumes_inds, self.wells_injector)
        # tag_ident1 = self.mb.tag_get_handle('identificador1', 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)
        # self.mb.tag_set_data(tag_ident1, volumes_inds, np.repeat(1, len(volumes_inds)))

        all_fis = self.mb.tag_get_data(self.phi_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        all_volumes = self.mb.tag_get_data(self.volume_tag, volumes, flat=True)
        all_fw = self.mb.tag_get_data(self.fw_tag, volumes, flat=True)
        all_total_flux = self.mb.tag_get_data(self.total_flux_tag, volumes, flat=True)
        # all_Vs = self.mb.tag_get_data(self.volume_tag, volumes, flat=True)
        # vv = self.mb.create_meshset()
        # self.mb.add_entities(vv, volumes)
        # self.mb.write_file('testtt.vtk', [vv])

        sats_2 = np.zeros(len(volumes))

        for i, volume in enumerate(volumes):
            # gid = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            sat1 = all_sats[i]
            V = all_volumes[i]
            if volume in self.wells_injector or sat1 == 1-self.Sor:
                sats_2[i] = sat1
                continue
            qw = all_qw[i]
            if qw < 0 and abs(qw) < lim_qw:
                qw = 0.0

            # if abs(qw) < lim:
            #     sats_2[i] = sat1
            #     continue
            # if qw < 0.0:
            #     print('qw < 0')
            #     print(qw)
            #     print('i')
            #     print(i)
            #     print('loop')
            #     print(loop)
            #     print('\n')
            #     return True
            # else:
            #     pass

            # if self.loop > 1:
            #     import pdb; pdb.set_trace()
            fi = all_fis[i]
            if fi == 0.0:
                sats_2[i] = sat1
                continue
            if volume in self.wells_producer:
                fw = all_fw[i]
                flux = all_total_flux[i]
                qw_out = flux*fw
            else:
                qw_out = 0.0
                fw = None
                flux = None

            sat = sat1 + (qw - qw_out)*(self.delta_t/(fi*V))
            # sat = sat1 + qw*(self.delta_t/(self.fimin*self.Vmin))
            # if sat1 > sat:
            #     print('erro na saturacao')
            #     print('sat1 > sat')
            #     return True
            if sat > (1-self.Sor) - delta_sat and sat < ((1-self.Sor)) + delta_sat:
                sat = 1-self.Sor
            elif sat > self.Swc - delta_sat and sat < self.Swc + delta_sat:
                sat = self.Swc

            elif sat > 1-self.Sor:
                #sat = 1 - self.Sor
                print("Sat > 0.8")
                print(sat)
                print('i')
                print(i)
                print('loop')
                print(loop)
                print('\n')

                # sat = 0.8
                return True

            elif sat < self.Swc:
                # vv = self.mb.create_meshset()
                # self.mb.add_entities(vv, volumes)
                # self.mb.write_file('testtt.vtk', [vv])
                #
                # pdb.set_trace()
                print('erro2')
                return True


            # elif sat > sat1 + 0.2:
            #     print('sat > sat1')
            #     print(f'sat: {sat}')
            #     print(f'sat1: {sat1}\n')
            #     return True

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
                return True

            else:
                pass

            sats_2[i] = sat

        t2 = time.time()
        # print('tempo calculo saturacao loop_{0}: {1}'.format(loop, t2-t1))
        self.mb.tag_set_data(self.sat_tag, volumes, sats_2)
        self.mb.tag_set_data(self.sat_last_tag, volumes, all_sats)
        return False

    def calculate_sat_vpi(self, volumes):
        """
        calcula a saturacao com o novo passo de tempo
        """
        t1 = time.time()
        lim = 1e-4
        all_qw = self.mb.tag_get_data(self.flux_w_tag, volumes, flat=True)
        all_fis = self.mb.tag_get_data(self.phi_tag, volumes, flat=True)
        all_sats = self.mb.tag_get_data(self.sat_tag, volumes, flat=True)
        all_volumes = self.mb.tag_get_data(self.volume_tag, volumes, flat=True)
        # all_Vs = self.mb.tag_get_data(self.volume_tag, volumes, flat=True)

        sats_2 = np.zeros(len(volumes))

        for i, volume in enumerate(volumes):
            # gid = mb.tag_get_data(self.global_id_tag, volume, flat=True)[0]
            sat1 = all_sats[i]
            V = all_volumes[i]
            if volume in self.wells_injector:
                sats_2[i] = sat1
                continue
            qw = all_qw[i]

            # if abs(qw) < lim:
            #     sats_2[i] = sat1
            #     continue

            # if self.loop > 1:
            #     import pdb; pdb.set_trace()
            fi = all_fis[i]
            if fi == 0.0:
                sats_2[i] = sat1
                continue
            sat = sat1 + qw*(self.delta_t/(fi*V))
            # sat = sat1 + qw*(self.delta_t/(self.fimin*self.Vmin))

            sats_2[i] = sat

        t2 = time.time()
        self.mb.tag_set_data(self.sat_tag, volumes, sats_2)
        self.mb.tag_set_data(self.sat_last_tag, volumes, all_sats)

    def create_flux_vector_pms(self, mb, mtu, p_corr_tag, volumes, k_eq_tag, dfdsmax=0, qmax=0):
        soma_inj = 0
        soma_prod = 0
        lim = 1e-4
        lim2 = 1e-7
        store_flux_pms_2 = np.zeros(len(volumes))

        fine_elems_in_primal = rng.Range(volumes)
        all_faces_in_primal = mtu.get_bridge_adjacencies(fine_elems_in_primal, 2, 3)
        all_keqs = mb.tag_get_data(k_eq_tag, all_faces_in_primal, flat=True)

        for i, face in enumerate(all_faces_in_primal):
            #1

            qw = 0
            flux = {}
            map_values = dict(zip(all_elems, values))
            fw_vol = self.mb.tag_get_data(self.fw_tag, elem, flat=True)[0]
            sat_vol = self.mb.tag_get_data(self.sat_tag, elem, flat=True)[0]
            for adj in all_elems[0:-1]:
                #4
                gid2 = self.mb.tag_get_data(self.global_id_tag, adj, flat=True)[0]
                if adj not in fine_elems_in_primal:
                    #5
                    pvol = self.mb.tag_get_data(self.pms_tag, elem, flat= True)[0]
                    padj = self.mb.tag_get_data(self.pms_tag, adj, flat= True)[0]
                #4
                else:
                    #5
                    pvol = self.mb.tag_get_data(self.pcorr_tag, elem, flat= True)[0]
                    padj = self.mb.tag_get_data(self.pcorr_tag, adj, flat= True)[0]
                #4
                q = -(padj - pvol)*map_values[adj]
                flux[adj] = q
                sat_adj = self.mb.tag_get_data(self.sat_tag, adj, flat=True)[0]
                fw_adj = self.mb.tag_get_data(self.fw_tag, adj, flat=True)[0]
                if q < 0:
                    fw = fw_vol
                else:
                    fw = fw_adj
                qw += fw*q
                if abs(sat_adj - sat_vol) < lim or abs(fw_adj -fw_vol) < lim:
                    continue
                dfds = abs((fw_adj - fw_vol)/(sat_adj - sat_vol))
                if dfds > self.dfdsmax:
                    self.dfdsmax = dfds
            #3
            store_flux_pms_2[elem] = flux
            if abs(sum(flux.values())) > lim2 and elem not in self.wells:
                #4
                print('nao esta dando conservativo na malha fina o fluxo multiescala')
                print(gid_vol)
                print(sum(flux.values()))
                import pdb; pdb.set_trace()
            #3
            self.mb.tag_set_data(self.flux_fine_pf_tag, elem, sum(flux.values()))
            qmax = max(list(map(abs, flux.values())))
            if qmax > self.qmax:
                self.qmax = qmax
            #3
            if elem in self.wells_prod:
                #4
                qw_out = sum(flux.values())*fw_vol
                qo_out = sum(flux.values())*(1 - fw_vol)
                self.prod_o.append(qo_out)
                self.prod_w.append(qw_out)
                qw -= qw_out
            #3
            if abs(qw) < lim and qw < 0.0:
                qw = 0.0
            elif qw < 0 and elem not in self.wells_inj:
                print('gid')
                print(gid_vol)
                print('qw < 0')
                print(qw)
                import pdb; pdb.set_trace()
            else:
                pass
            self.mb.tag_set_data(self.flux_w_tag, elem, qw)

        soma_inj = []
        soma_prod = []
        soma2 = 0
        with open('fluxo_multiescala_bif{0}.txt'.format(self.loop), 'w') as arq:
            for volume in self.wells:
                gid = self.mb.tag_get_data(self.global_id_tag, volume, flat = True)[0]
                values = self.store_flux_pms[volume].values()
                arq.write('gid:{0} , fluxo:{1}\n'.format(gid, sum(values)))
                if volume in self.wells_inj:
                    soma_inj.append(sum(values))
                else:
                    soma_prod.append(sum(values))
                # print('\n')
                soma2 += sum(values)
            arq.write('\n')
            arq.write('soma_inj:{0}\n'.format(sum(soma_inj)))
            arq.write('soma_prod:{0}\n'.format(sum(soma_prod)))
            arq.write('tempo:{0}'.format(self.tempo))

        self.store_flux_pms = store_flux_pms_2

    def calculate_pcorr_dep0(self, mb, elems_in_meshset, vertice, faces_boundary, faces, pcorr_tag, pms_tag, volumes_d, volumes_n, dict_tags, pcorr2_tag=None):
        """
        mb = core do pymoab
        elems_in_meshset = elementos dentro de um meshset
        vertice = elemento que é vértice do meshset
        faces_boundary = faces do contorno do meshset
        faces = todas as faces do meshset
        pcorr_tag = tag da pressao corrigida
        pms_tag = tag da pressao multiescala

        """
        allmobis = mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        s_gravs = mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        n = len(elems_in_meshset)
        map_local = dict(zip(elems_in_meshset, range(n)))
        T = sp.lil_matrix((n, n))
        b = np.zeros(n)
        s_grav_elems = np.zeros(n)

        for i, face in enumerate(faces):
            mobi = -allmobis[i]
            s_g = -s_gravs[i]
            elems = mb.get_adjacencies(face, 3)
            if face in faces_boundary:
                p = mb.tag_get_data(pms_tag, elems, flat=True)
                flux = (p[1] - p[0])*mobi
                if elems[0] in elems_in_meshset:
                    local_id = map_local[elems[0]]
                    b[local_id] += flux
                    s_grav_elems[local_id] += s_g
                else:
                    local_id = map_local[elems[1]]
                    b[local_id] -= flux
                    s_grav_elems[local_id] -= s_g

                continue

            local_id0 = map_local[elems[0]]
            local_id1 = map_local[elems[1]]
            s_grav_elems[local_id0] += s_g
            s_grav_elems[local_id1] -= s_g


            T[local_id0, local_id0] += mobi
            T[local_id1, local_id1] += mobi
            T[[local_id0, local_id1], [local_id1, local_id0]] = [-mobi, -mobi]

        if self.gravity == True:
            b += s_grav_elems

        d_vols = rng.intersect(elems_in_meshset, volumes_d)
        d_vols = rng.unite(d_vols, rng.Range(vertice))
        map_values = dict(zip(d_vols, mb.tag_get_data(pms_tag, d_vols, flat=True)))
        T, b = oth.set_boundary_dirichlet_matrix(map_local, map_values, b, T)
        n_vols = rng.intersect(volumes_n, elems_in_meshset)
        if len(n_vols) > 0:
            map_values = dict(zip(n_vols, mb.tag_get_data(dict_tags['Q'], n_vols, flat=True)))
            b = oth.set_boundary_neumann(map_local, map_values, b)

        x = oth.get_solution(T, b)
        mb.tag_set_data(pcorr_tag, elems_in_meshset, x)
        if pcorr2_tag == None:
            pass
        else:
            mb.tag_set_data(pcorr2_tag, elems_in_meshset, x)

    def calculate_pcorr(self, mb, elems_in_meshset, vertice, faces_boundary, faces, pcorr_tag, pms_tag, volumes_d, volumes_n, dict_tags, pcorr2_tag=None):
        """
        mb = core do pymoab
        elems_in_meshset = elementos dentro de um meshset
        vertice = elemento que é vértice do meshset
        faces_boundary = faces do contorno do meshset
        faces = todas as faces do meshset
        pcorr_tag = tag da pressao corrigida
        pms_tag = tag da pressao multiescala

        """
        allmobis = mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        vols3 = self.mtu.get_bridge_adjacencies(faces_boundary, 2, 3)
        vols_inter = rng.subtract(vols3, elems_in_meshset)
        pms_vols3 = self.mb.tag_get_data(pms_tag, vols3, flat=True)
        map_pms_vols3 = dict(zip(vols3, pms_vols3))
        del pms_vols3

        volumes_2 = self.mtu.get_bridge_adjacencies(elems_in_meshset, 2, 3)
        if self.gravity:
            s_gravs = mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        else:
            s_gravs = np.zeros(len(faces))
        n = len(elems_in_meshset)

        map_local = dict(zip(elems_in_meshset, range(n)))
        lines = []
        cols = []
        data = []
        b = np.zeros(n)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]
        faces_in = rng.subtract(faces, faces_boundary)
        map_id_faces = dict(zip(faces, range(len(faces))))

        for face in faces_in:
            id_face = map_id_faces[face]
            mobi = allmobis[id_face]
            s_g = -s_gravs[id_face]
            id_face = map_id_faces[face]
            elem0 = Adjs[id_face][0]
            elem1 = Adjs[id_face][1]
            id0 = map_local[elem0]
            id1 = map_local[elem1]
            b[id0] += s_g
            b[id1] -= s_g
            lines += [id0, id1]
            cols += [id1, id0]
            data += [mobi, mobi]

        for face in faces_boundary:
            id_face = map_id_faces[face]
            mobi = allmobis[id_face]
            s_g = -s_gravs[id_face]
            elem0 = Adjs[id_face][0]
            elem1 = Adjs[id_face][1]
            vvv = True
            try:
                id = map_local[elem0]
            except KeyError:
                id = map_local[elem1]
                vvv = False
            flux = -(map_pms_vols3[elem1] - map_pms_vols3[elem0])*mobi
            if vvv:
                b[id] += s_g + flux
            else:
                b[id] -= s_g + flux

        T = sp.csc_matrix((data,(lines,cols)),shape=(n, n))
        T = T.tolil()
        d1 = np.array(T.sum(axis=1)).reshape(1, n)[0]*(-1)
        T.setdiag(d1)

        level = np.unique(mb.tag_get_data(dict_tags['l3_ID'], elems_in_meshset, flat=True))

        d_vols = rng.Range(vertice)
        map_values = dict(zip(d_vols, mb.tag_get_data(pms_tag, d_vols, flat=True)))
        T, b = oth.set_boundary_dirichlet_matrix(map_local, map_values, b, T)

        x = oth.get_solution(T, b)
        mb.tag_set_data(pcorr_tag, elems_in_meshset, x)
        if pcorr2_tag == None:
            pass
        else:
            mb.tag_set_data(pcorr2_tag, elems_in_meshset, x)

    def calculate_pcorr_v5(self, mb, elems_in_meshset, vertice, faces_boundary, faces, pcorr_tag, pms_tag, volumes_d, volumes_n, dict_tags, pcorr2_tag=None):
        """
        mb = core do pymoab
        elems_in_meshset = elementos dentro de um meshset
        vertice = elemento que é vértice do meshset
        faces_boundary = faces do contorno do meshset
        faces = todas as faces do meshset
        pcorr_tag = tag da pressao corrigida
        pms_tag = tag da pressao multiescala

        """
        if vertice not in self.vertices:
            import pdb; pdb.set_trace()

        allmobis = mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        allkdif = mb.tag_get_data(self.kdif_tag, faces, flat=True)
        vols3 = self.mtu.get_bridge_adjacencies(faces_boundary, 2, 3)
        vols_inter = rng.subtract(vols3, elems_in_meshset)
        pms_vols3 = self.mb.tag_get_data(pms_tag, vols3, flat=True)
        map_pms_vols3 = dict(zip(vols3, pms_vols3))
        del pms_vols3

        volumes_2 = self.mtu.get_bridge_adjacencies(elems_in_meshset, 2, 3)
        if self.gravity:
            s_gravs = mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        else:
            s_gravs = np.zeros(len(faces))
        n = len(elems_in_meshset)

        map_local = dict(zip(elems_in_meshset, range(n)))
        lines = []
        cols = []
        data = []
        b = np.zeros(n)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces]
        faces_in = rng.subtract(faces, faces_boundary)
        map_id_faces = dict(zip(faces, range(len(faces))))

        for face in faces_in:
            id_face = map_id_faces[face]
            mobi = allkdif[id_face]
            s_g = -s_gravs[id_face]
            elem0 = Adjs[id_face][0]
            elem1 = Adjs[id_face][1]
            id0 = map_local[elem0]
            id1 = map_local[elem1]
            b[id0] += s_g
            b[id1] -= s_g
            lines += [id0, id1]
            cols += [id1, id0]
            data += [mobi, mobi]

        for face in faces_boundary:
            id_face = map_id_faces[face]
            mobi = allkdif[id_face]
            s_g = -s_gravs[id_face]
            elem0 = Adjs[id_face][0]
            elem1 = Adjs[id_face][1]
            vvv = True
            try:
                id = map_local[elem0]
            except KeyError:
                id = map_local[elem1]
                vvv = False
            flux = -(map_pms_vols3[elem1] - map_pms_vols3[elem0])*mobi + s_g
            if vvv:
                b[id] += flux
            else:
                b[id] -= flux

        T = sp.csc_matrix((data,(lines,cols)),shape=(n, n))
        T = T.tolil()
        d1 = np.array(T.sum(axis=1)).reshape(1, n)[0]*(-1)
        T.setdiag(d1)

        level = np.unique(mb.tag_get_data(dict_tags['l3_ID'], elems_in_meshset, flat=True))

        # d_vols = rng.Range(vertice)
        d_vols = np.array([vertice])
        # map_values = dict(zip(d_vols, mb.tag_get_data(pms_tag, d_vols, flat=True)))
        # T, b = oth.set_boundary_dirichlet_matrix(map_local, map_values, b, T)
        T[map_local[vertice]] = 0.0
        T[map_local[vertice], map_local[vertice]] = 1.0
        b[map_local[vertice]] = mb.tag_get_data(pms_tag, vertice, flat=True)[0]

        x = oth.get_solution(T, b)
        mb.tag_set_data(pcorr_tag, elems_in_meshset, x)
        if pcorr2_tag == None:
            pass
        else:
            mb.tag_set_data(pcorr2_tag, elems_in_meshset, x)

    def calculate_pcorr_v3(self, mb, faces_boundary_nv1, pms_tag, p_corr_tag, vertices, tags, all_volumes):
        # TODO: fazer desacoplamento para pressão corrigida
        n = len(all_volumes)
        meshsets_nv1 = self.mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags['PRIMAL_ID_1']]), np.array([None]))
        ids_globais = self.mb.tag_get_data(tags['ID_reord_tag'], all_volumes, flat=True)
        map_volumes = dict(zip(all_volumes, range(len(all_volumes))))
        pmss = self.mb.tag_get_data(pms_tag, all_volumes, flat=True)

        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces_boundary_nv1]
        if self.gravity:
            s_grav_faces = self.mb.tag_get_data(tags['S_GRAV'], faces_boundary_nv1, flat=True)
        else:
            s_grav_faces = np.zeros(len(faces_boundary_nv1))
        ids_vertices = self.mb.tag_get_data(tags['ID_reord_tag'], vertices, flat=True)
        presc = self.mb.tag_get_data(pms_tag, vertices, flat=True)
        map_presc = dict(zip(vertices, presc))
        # pms0 = self.mb.tag_get_data(pms_tag, elems0, flat=True)
        # pms1 = self.mb.tag_get_data(pms_tag, elems1, flat=True)
        # ids_elems0 = self.mb.tag_get_data(tags['ID_reord_tag'], elems0, flat=True)
        # ids_elems1 = self.mb.tag_get_data(tags['ID_reord_tag'], elems1, flat=True)

        # mobis_in_faces_boundary = self.Tf[ids_elems0, ids_elems1].todense()
        # mobis_in_faces_boundary = self.mb.tag_get_data(self.mobi_in_faces_tag, faces_boundary_nv1, flat=True)
        # flux = (pms1 - pms0)*np.absolute(mobis_in_faces_boundary) + s_grav_faces

        Tf = self.Tf.copy()
        # Tf = Tf.tolil()
        b = np.zeros(len(all_volumes))

        ids_nos_primais = self.mb.tag_get_data(tags['IDS_NA_PRIMAL_2'], all_volumes, flat=True)
        ids_elems0 = []
        ids_elems1 = []
        mobis2 = []
        fluxs = []
        map_ids_globais_in_primais = dict(zip(ids_globais, ids_nos_primais))
        map_ids_primais_in_globais = dict(zip(ids_nos_primais, ids_globais))

        for i, face in enumerate(faces_boundary_nv1):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id0 = ids_globais[map_volumes[elem0]]
            id1 = ids_globais[map_volumes[elem1]]
            # id0 = ids_globais[map_volumes[elem0]]
            # id1 = ids_globais[map_volumes[elem1]]
            mobi = Tf[id0, id1]
            Tf[id0, id1] = 0
            Tf[id1, id0] = 0
            Tf[id0, id0] += mobi
            Tf[id1, id1] += mobi
            # mobis2.append(mobi)
            # flux = (pmss[id1] - pmss[id0])*abs(mobi) + s_grav_faces[i]
            flux = (pmss[map_volumes[elem1]] - pmss[map_volumes[elem0]])*mobi + s_grav_faces[i]
            b[id0] += flux
            b[id1] -= flux

        Tf[ids_vertices] = sp.lil_matrix((len(ids_vertices), n))
        Tf[ids_vertices, ids_vertices] = np.ones(len(vertices))
        b[ids_vertices] = presc

        G = sp.csc_matrix((np.ones(len(all_volumes)), (ids_nos_primais, ids_globais)), shape=(n, n))
        Gt = G.transpose()
        b0 = b.copy()
        b = G.dot(b)
        Tf = G.dot(Tf)
        Tf = Tf.dot(Gt)
        resp = np.zeros(len(all_volumes))

        cont = 0

        for m in meshsets_nv1:
            cont += 1
            elems = self.mb.get_entities_by_handle(m)
            level = np.unique(self.mb.tag_get_data(tags['l3_ID'], elems, flat=True))[0]
            if level == 1:
                continue
            vert = rng.intersect(elems, vertices)
            if level == 3:
                import pdb; pdb.set_trace()
            try:
                val_vert = map_presc[vert[0]]
            except:
                import pdb; pdb.set_trace()

            ids_glob = self.mb.tag_get_data(tags['ID_reord_tag'], elems, flat=True)
            ids_primal = self.mb.tag_get_data(tags['IDS_NA_PRIMAL_2'], elems, flat=True)
            idsl = ids_primal
            tt = Tf[idsl.min():idsl.max()+1, idsl.min():idsl.max()+1]
            b2 = b[idsl]
            resp[ids_glob] = oth.get_solution(tt, b2)

        v1 = np.arange(len(all_volumes))

        G2 = sp.csc_matrix((np.ones(len(all_volumes)), (v1, ids_globais)), shape=(n, n))
        resp = G2.dot(resp)
        self.mb.tag_set_data(p_corr_tag, all_volumes, resp)

    def calculate_pcorr_v4(self, elems_in_meshset, pcorr_tag, dict_tags):
        """
        mb = core do pymoab
        elems_in_meshset = elementos dentro de um meshset
        vertice = elemento que é vértice do meshset
        faces_boundary = faces do contorno do meshset
        faces = todas as faces do meshset
        pcorr_tag = tag da pressao corrigida
        pms_tag = tag da pressao multiescala

        """
        n = len(elems_in_meshset)
        level = np.unique(self.mb.tag_get_data(dict_tags['l3_ID'], elems_in_meshset, flat=True))[0]
        ids_nos_primais = self.mb.tag_get_data(dict_tags['IDS_NA_PRIMAL_' + str(level)], elems_in_meshset, flat=True)
        map_volumes = dict(zip(elems_in_meshset, range(n)))
        idsl = ids_nos_primais

        if level == 2:
            Tf = self.Tf2[idsl.min():idsl.max()+1, idsl.min():idsl.max()+1]
            b = self.b2[idsl]
        elif level == 3:
            Tf = self.Tf3[idsl.min():idsl.max()+1, idsl.min():idsl.max()+1]
            b = self.b3[idsl]

        # if level == 3:
        #     import pdb; pdb.set_trace()

        resp = oth.get_solution(Tf, b)
        # if level < 3:
        #     self.mb.tag_set_data(pcorr_tag, elems_in_meshset, resp)
        # else:
        #     self.mb.tag_set_data(pcorr_tag, elems_in_meshset, np.zeros(len(elems_in_meshset)))

        self.mb.tag_set_data(pcorr_tag, elems_in_meshset, resp)

    def get_flux_coarse_volumes_dep0(self, tags, all_volumes, boundary_faces_levels, pms_tag, meshsets_vetices_levels):

        vertices_nv2 = self.mb.get_entities_by_handle(meshsets_vetices_levels[0])
        vertices_nv3 = self.mb.get_entities_by_handle(meshsets_vetices_levels[1])

        n = len(all_volumes)
        map_volumes = dict(zip(all_volumes, range(n)))
        ids_nos_primais = self.mb.tag_get_data(tags['IDS_NA_PRIMAL_2'], all_volumes, flat=True)
        ids_globais = self.mb.tag_get_data(tags['ID_reord_tag'], all_volumes, flat=True)
        G = sp.csc_matrix((np.ones(len(all_volumes)), (ids_nos_primais, ids_globais)), shape=(n, n))
        Gt = G.transpose()

        b = np.zeros(n)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in boundary_faces_levels[0]]
        pmss = [self.mb.tag_get_data(pms_tag, elems, flat=True) for elems in Adjs]

        if self.gravity:
            s_grav_faces = self.mb.tag_get_data(tags['S_GRAV'], boundary_faces_levels[0], flat=True)
        else:
            s_grav_faces = np.zeros(len(boundary_faces_levels[0]))

        Tf = self.Tf.copy()

        for i, face in enumerate(boundary_faces_levels[0]):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id_loc_0 = map_volumes[elem0]
            id_loc_1 = map_volumes[elem1]
            id0 = ids_globais[id_loc_0]
            id1 = ids_globais[id_loc_1]
            s_g = s_grav_faces[i]
            mobi = Tf[id0, id1]
            Tf[id0, id1] = 0
            Tf[id1, id0] = 0
            Tf[id1, id1] += mobi
            Tf[id0, id0] += mobi
            flux = (pmss[i][1] - pmss[i][0])*mobi + s_g
            b[id0] -= flux
            b[id1] += flux

        pms_vertices = self.mb.tag_get_data(pms_tag, vertices_nv2, flat=True)
        ids_vertices = self.mb.tag_get_data(tags['ID_reord_tag'], vertices_nv2, flat=True)
        b[ids_vertices] = pms_vertices
        Tf[ids_vertices] = sp.lil_matrix((len(ids_vertices), n))
        Tf[ids_vertices, ids_vertices] = np.ones(len(ids_vertices))

        Tf = G.dot(Tf)
        Tf = Tf.dot(Gt)
        b = G.dot(b)

        self.Tf2 = Tf
        self.b2 = b

        ids_nos_primais = self.mb.tag_get_data(tags['IDS_NA_PRIMAL_3'], all_volumes, flat=True)
        G = sp.csc_matrix((np.ones(len(all_volumes)), (ids_nos_primais, ids_globais)), shape=(n, n))
        Gt = G.transpose()

        b = np.zeros(n)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in boundary_faces_levels[1]]
        pmss = [self.mb.tag_get_data(pms_tag, elems, flat=True) for elems in Adjs]

        Tf = self.Tf.copy()

        for i, face in enumerate(boundary_faces_levels[1]):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id_loc_0 = map_volumes[elem0]
            id_loc_1 = map_volumes[elem1]
            id0 = ids_globais[id_loc_0]
            id1 = ids_globais[id_loc_1]
            s_g = s_grav_faces[i]
            mobi = Tf[id0, id1]
            Tf[id0, id1] = 0
            Tf[id1, id0] = 0
            Tf[id1, id1] += mobi
            Tf[id0, id0] += mobi
            flux = (pmss[i][1] - pmss[i][0])*mobi + s_g
            b[id0] -= flux
            b[id1] += flux

        pms_vertices = self.mb.tag_get_data(pms_tag, vertices_nv3, flat=True)
        ids_vertices = self.mb.tag_get_data(tags['ID_reord_tag'], vertices_nv3, flat=True)
        b[ids_vertices] = pms_vertices
        Tf[ids_vertices] = sp.lil_matrix((len(ids_vertices), n))
        Tf[ids_vertices, ids_vertices] = np.ones(len(ids_vertices))

        Tf = G.dot(Tf)
        Tf = Tf.dot(Gt)
        b = G.dot(b)

        self.Tf3 = Tf
        self.b3 = b

    def get_flux_coarse_volumes(self, tags, all_volumes, boundary_faces_levels, pms_tag, meshsets_vetices_levels):

        vertices_nv2 = self.mb.get_entities_by_handle(meshsets_vetices_levels[0])
        vertices_nv3 = self.mb.get_entities_by_handle(meshsets_vetices_levels[1])

        n = len(all_volumes)
        map_volumes = dict(zip(all_volumes, range(n)))
        ids_nos_primais = self.mb.tag_get_data(tags['IDS_NA_PRIMAL_3'], all_volumes, flat=True)
        ids_globais = self.mb.tag_get_data(tags['ID_reord_tag'], all_volumes, flat=True)
        G = sp.csc_matrix((np.ones(len(all_volumes)), (ids_nos_primais, ids_globais)), shape=(n, n))
        Gt = G.transpose()

        b = np.zeros(n)
        Adjs = [self.mb.get_adjacencies(face, 3) for face in boundary_faces_levels[1]]
        pmss = [self.mb.tag_get_data(pms_tag, elems, flat=True) for elems in Adjs]

        if self.gravity:
            s_grav_faces = self.mb.tag_get_data(tags['S_GRAV'], boundary_faces_levels[0], flat=True)
        else:
            s_grav_faces = np.zeros(len(boundary_faces_levels[0]))

        Tf = self.Tf.copy()

        for i, face in enumerate(boundary_faces_levels[1]):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id_loc_0 = map_volumes[elem0]
            id_loc_1 = map_volumes[elem1]
            id0 = ids_globais[id_loc_0]
            id1 = ids_globais[id_loc_1]
            s_g = s_grav_faces[i]
            mobi = Tf[id0, id1]
            Tf[id0, id1] = 0
            Tf[id1, id0] = 0
            Tf[id1, id1] += mobi
            Tf[id0, id0] += mobi
            flux = -(pmss[i][1] - pmss[i][0])*mobi + s_g
            b[id0] += flux
            b[id1] -= flux

        pms_vertices = self.mb.tag_get_data(pms_tag, vertices_nv3, flat=True)
        ids_vertices = self.mb.tag_get_data(tags['ID_reord_tag'], vertices_nv3, flat=True)
        b[ids_vertices] = pms_vertices
        Tf[ids_vertices] = sp.lil_matrix((len(ids_vertices), n))
        Tf[ids_vertices, ids_vertices] = np.ones(len(ids_vertices))

        Tf = G.dot(Tf)
        Tf = Tf.dot(Gt)
        b = G.dot(b)

        self.Tf3 = Tf.copy()
        self.b3 = b.copy()

        Tf = Gt.dot(Tf)
        Tf = Tf.dot(G)
        b = Gt.dot(b)

        ids_nos_primais = self.mb.tag_get_data(tags['IDS_NA_PRIMAL_2'], all_volumes, flat=True)
        G = sp.csc_matrix((np.ones(len(all_volumes)), (ids_nos_primais, ids_globais)), shape=(n, n))
        Gt = G.transpose()

        faces2 = rng.subtract(boundary_faces_levels[0], boundary_faces_levels[1])
        Adjs = [self.mb.get_adjacencies(face, 3) for face in faces2]
        pmss = [self.mb.tag_get_data(pms_tag, elems, flat=True) for elems in Adjs]

        if self.gravity:
            s_grav_faces = self.mb.tag_get_data(tags['S_GRAV'], faces2, flat=True)
        else:
            s_grav_faces = np.zeros(len(faces2))

        for i, face in enumerate(faces2):
            elem0 = Adjs[i][0]
            elem1 = Adjs[i][1]
            id_loc_0 = map_volumes[elem0]
            id_loc_1 = map_volumes[elem1]
            id0 = ids_globais[id_loc_0]
            id1 = ids_globais[id_loc_1]
            s_g = s_grav_faces[i]
            mobi = Tf[id0, id1]
            Tf[id0, id1] = 0
            Tf[id1, id0] = 0
            Tf[id1, id1] += mobi
            Tf[id0, id0] += mobi
            flux = -(pmss[i][1] - pmss[i][0])*mobi + s_g
            b[id0] += flux
            b[id1] -= flux

        pms_vertices = self.mb.tag_get_data(pms_tag, vertices_nv2, flat=True)
        ids_vertices = self.mb.tag_get_data(tags['ID_reord_tag'], vertices_nv2, flat=True)
        b[ids_vertices] = pms_vertices
        Tf[ids_vertices] = sp.lil_matrix((len(ids_vertices), n))
        Tf[ids_vertices, ids_vertices] = np.ones(len(ids_vertices))

        Tf = G.dot(Tf)
        Tf = Tf.dot(Gt)
        b = G.dot(b)

        self.Tf2 = Tf.copy()
        self.b2 = b.copy()

    def get_hist_ms_dep0(self, t, dt, loop):

        flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True)
        fws = self.mb.tag_get_data(self.fw_tag, self.wells_producer, flat=True)

        qw = (flux_total_prod*fws).sum()*self.delta_t
        qo = (flux_total_prod.sum()- qw)*self.delta_t
        wor = qw/float(qo)
        vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        self.vpi += vpi

        hist = np.array([self.vpi, t, qw, qo, wor, dt])
        name = 'historico_' + str(loop)
        np.save(name, hist)

    def get_hist_ms(self, t, dt, loop):

        flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True)
        fws = self.mb.tag_get_data(self.fw_tag, self.wells_producer, flat=True)

        qw = (flux_total_prod*fws).sum()*self.delta_t
        qo = (flux_total_prod.sum()- qw)*self.delta_t
        wor = qw/float(qo)
        vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        self.vpi += vpi
        self.hist = np.array([self.vpi, t, qw, qo, wor, dt])

    def print_hist(self, loop):
        name = 'historico_' + str(loop)
        np.save(name, self.hist)

    def verificar_cfl(self, volumes, loop):
        t0 = time.time()
        print('entrou verificar cfl')
        # cfl_min = 0.1
        erro_cfl = True
        cfl = self.cfl
        contagem = 0

        while erro_cfl:
            erro_cfl = self.calculate_sat(volumes, loop)
            if erro_cfl:
                cfl = self.rec_cfl(cfl)
                self.cfl = cfl
                contagem += 1
                # if cfl < cfl_min:
                #     cfl = cfl*0.1
                #     self.cfl = cfl
                #     erro_cfl = self.calculate_sat(volumes, loop)
                #     erro_cfl = False
                # else:
                #     self.cfl = cfl
            if contagem > 1000:
                print('cfl nao convergiu')
                print(f'cfl: {cfl}')
        t1 = time.time()
        dt = t1 - t0
        print('saiu de verificar cfl')
        print(f'tempo: {dt}')
