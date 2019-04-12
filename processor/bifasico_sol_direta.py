import numpy as np
from pymoab import core, types, rng, topo_util, skinner
import os
import sys
import io
import yaml
import scipy.sparse as sp
import time
import conversao as conv

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
        self.k2 = 0.9
        self.cfl_ini = self.k2
        self.cfl = self.k2
        self.perm_tag = mb.tag_get_handle('PERM')
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
        self.sat_last_tag = mb.tag_get_handle('SAT_LAST', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
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
        phis =  mb.tag_get_data(self.phi_tag, all_volumes, flat=True)
        bb = np.nonzero(phis)[0]
        phis = phis[bb]
        v0 = all_volumes[0]
        points = self.mtu.get_bridge_adjacencies(v0, 3, 0)
        coords = self.mb.get_coords(points).reshape(len(points), 3)
        maxs = coords.max(axis=0)
        mins = coords.min(axis=0)
        hs = maxs - mins
        self.hs = hs

        # hs[0] = conv.pe_to_m(hs[0])
        # hs[1] = conv.pe_to_m(hs[1])
        # hs[1] = conv.pe_to_m(hs[1])
        #
        # self.hs = hs
        # vol = hs[0]*hs[1]*hs[2]
        vol = 1.0
        self.mb.tag_set_data(self.volume_tag, all_volumes, np.repeat(vol, len(all_volumes)))
        self.Vmin = vol
        self.fimin = phis.min()
        historico = [np.array(['vpi','tempo', 'prod_agua', 'prod_oleo', 'wor', 'dt'])]
        np.save('historico', historico)
        self.delta_t = 0.0
        self.V_total = mb.tag_get_data(self.volume_tag, all_volumes, flat=True)
        self.V_total = float((self.V_total*mb.tag_get_data(self.phi_tag, all_volumes, flat=True)).sum())
        self.vpi = 0.0

    def calculate_total_flux(self, volumes, faces):

        mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        if self.gravity:
            s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        else:
            s_gravs_faces = np.zeros(len(faces))

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

    def calculate_sat_dep0(self, volumes, loop):
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

    def calculate_sat(self, volumes, loop):
        """
        calcula a saturacao do passo de tempo corrente
        """
        #self.loop = loop
        t1 = time.time()
        lim = 1e-10
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
            if sat1 == 0.8:
                sats_2[i] = sat1
                continue
            qw = all_qw[i]

            # if abs(qw) < lim:
            #     sats_2[i] = sat1
            #     continue
            if qw < -lim:
                print('abs(qw) > lim')
                print(qw)
                print('i')
                print(i)
                print('loop')
                print(loop)
                print('\n')
                import pdb; pdb.set_trace()
                return 1

            else:
                pass

            # if self.loop > 1:
            #     import pdb; pdb.set_trace()
            fi = all_fis[i]
            # if fi == 0.0:
            #     sats_2[i] = sat1
            #     continue
            sat = sat1 + qw*(self.delta_t/(fi*V))
            # sat = sat1 + qw*(self.delta_t/(self.fimin*self.Vmin))
            # if sat1 > sat + lim:
            #     print('erro na saturacao')
            #     print('sat1 > sat')
            #     return True
            if sat > 0.8:
                #sat = 1 - self.Sor
                print("Sat > 0.8")
                print(sat)
                print('i')
                print(i)
                print('loop')
                print(loop)
                print('\n')
                # self.delta_t = fi*V*(0.8 - sat1)/qw
                # self.mb.tag_set_data(self.sat_tag, volume, 0.8)
                # return 2
                return 1

            elif sat > sat1 + 0.2:
                print('sat > sat1 + 0.2')
                print(f'sat: {sat}')
                print(f'sat1: {sat1}\n')
                return 1


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
                return 1

            else:
                pass

            sats_2[i] = sat

        t2 = time.time()
        # print('tempo calculo saturacao loop_{0}: {1}'.format(loop, t2-t1))
        self.mb.tag_set_data(self.sat_tag, volumes, sats_2)
        vols2 = rng.subtract(volumes, self.wells_injector)
        satss = self.mb.tag_get_data(self.sat_tag, vols2, flat=True)
        rr = np.where(satss > 0.8)[0]
        print(rr)
        if len(rr) > 0:
            import pdb; pdb.set_trace()
        # self.mb.tag_set_data(self.sat_last_tag, volumes, all_sats)
        return 0

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

    def calc_cfl(self, faces_in):
        """
        cfl usando fluxo maximo
        """
        self.all_faces_in = faces_in
        self.cfl = self.k2

        qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, faces_in, flat=True)).max()
        dfdsmax = self.mb.tag_get_data(self.dfds_tag, faces_in, flat=True).max()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.flux_total_producao = self.flux_total_prod

        self.delta_t = self.cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        self.vpi += vpi

    def rec_cfl(self, cfl):
        cfl = 0.5*cfl
        print('novo cfl', cfl)
        qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, self.all_faces_in, flat=True)).max()
        dfdsmax = self.mb.tag_get_data(self.dfds_tag, self.all_faces_in, flat=True).max()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.delta_t = cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        self.vpi += vpi
        return cfl

    def get_hist(self, t, dt):
        flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True)
        fws = self.mb.tag_get_data(self.fw_tag, self.wells_producer, flat=True)

        qw = (flux_total_prod*fws).sum()*self.delta_t
        qo = (flux_total_prod.sum() - qw)*self.delta_t
        wor = qw/float(qo)


        hist = np.array([self.vpi, t, qw, qo, wor, dt])
        historico = np.load('historico.npy')
        historico = np.append(historico, hist)
        np.save('historico', historico)

    def pol_interp(self, S):
        # S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
        # krw = (S_temp)**(self.nw)
        # kro = (1 - S_temp)**(self.no)
        if S >= (1 - self.Sor):
            krw = 1.0
            kro = 0.0
        elif S <= self.Swc:
            krw = 0.0
            kro = 1.0
        else:

            # krw = ((S - self.Swc)/float(1 - self.Swc - self.Sor))**(self.nw)
            # kro = ((1 - S - self.Swc)/float(1 - self.Swc - self.Sor))**(self.no)
            S_temp = (S - self.Swc)/(1 - self.Swc - self.Sor)
            krw = (S_temp)**(self.nw)
            kro = (1 - S_temp)**(self.no)

        if krw < 0.0 or kro < 0.0:
            print('erro no kr')
            import pdb; pdb.set_trace()

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
            krw, kro = self.pol_interp(sat)
            # volume = all_volumes[i]
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

    def set_mobi_faces_dep0(self, volumes, faces):
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
            elif flux_in_face <= 0:
                all_mobi_in_faces[i] = keq*(lbt0)
                all_fw_in_face[i] = fw0
            else:
                all_mobi_in_faces[i] = keq*(lbt1)
                all_fw_in_face[i] = fw1

            if all_mobi_in_faces[i] < 0.0:
                print('erro mobi in faces')
                import pdb; pdb.set_trace()

            all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])

        self.mb.tag_set_data(self.mobi_in_faces_tag, faces, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, faces, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, faces, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, faces, all_dfds)

    def set_mobi_faces(self, volumes, faces, finos0=None):

        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        # finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.01
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
        all_ks =self.mb.tag_get_data(self.perm_tag, volumes)

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
            k0 = all_ks[id0].reshape([3,3])
            k1 = all_ks[id1].reshape([3,3])
            direction = all_centroids[id1] - all_centroids[id0]
            norma = np.linalg.norm(direction)
            uni = np.absolute(direction/norma)
            k0 = np.dot(np.dot(k0, uni), uni)
            k1 = np.dot(np.dot(k1, uni), uni)
            h = np.dot(self.hs, uni)

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
                # continue
            elif elems[1] in self.wells_injector:
                all_mobi_in_faces[i] = k1*lbt1
                all_fw_in_face[i] = fw1
                # continue
            elif flux_in_face < 0:
                all_mobi_in_faces[i] = k0*(lbt0)
                all_fw_in_face[i] = fw0
            else:
                all_mobi_in_faces[i] = k1*(lbt1)
                all_fw_in_face[i] = fw1
            all_mobi_in_faces[i] /= h
            all_s_gravs[i] = self.gama*all_mobi_in_faces[i]*(all_centroids[id1][2] - all_centroids[id0][2])

        self.mb.tag_set_data(self.mobi_in_faces_tag, faces, all_mobi_in_faces)
        self.mb.tag_set_data(self.s_grav_tag, faces, all_s_gravs)
        self.mb.tag_set_data(self.fw_in_faces_tag, faces, all_fw_in_face)
        self.mb.tag_set_data(self.dfds_tag, faces, all_dfds)
        # vols_finos = self.mb.get_entities_by_handle(finos)
        # self.mb.tag_set_data(finos_val, vols_finos, np.repeat(1.0, len(vols_finos)))

    def get_Tf_and_b(self, all_volumes, map_volumes, faces_in, dict_tags):
        lines_tf = []
        cols_tf = []
        data_tf = []
        lines_ttf = []
        cols_ttf = []
        data_ttf = []

        if self.gravity:
            all_s_gravs = self.mb.tag_get_data(dict_tags['S_GRAV'], faces_in, flat=True)
        else:
            all_s_gravs = np.zeros(len(faces_in))

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
            lines_tf += [Gid_1, Gid_2]
            cols_tf += [Gid_2, Gid_1]
            data_tf += [keq, keq]

            flux_grav = -all_s_gravs[i]
            s_grav[id_0] += flux_grav
            s_grav[id_1] -= flux_grav

        n = len(all_volumes)
        Tf = sp.csc_matrix((data_tf,(lines_tf,cols_tf)),shape=(n, n))
        Tf = Tf.tolil()
        d1 = np.array(Tf.sum(axis=1)).reshape(1, n)[0]*(-1)
        Tf.setdiag(d1)

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
        # np.save('pf', x)

    def set_Pf(self, all_volumes):
        pf = np.load('pf.npy')
        self.mb.tag_set_data(self.pf_tag, all_volumes, pf)

    def verificar_cfl(self, volumes, loop):
        t0 = time.time()
        print('entrou verificar cfl')
        erro_cfl = 1
        cfl = self.cfl
        contagem = 0

        while erro_cfl != 0:
            erro_cfl = self.calculate_sat(volumes, loop)
            if erro_cfl != 0:
                if erro_cfl == 1:
                    cfl = self.rec_cfl(cfl)
                    self.cfl = cfl
                elif erro_cfl == 2:
                    erro_cfl = self.calculate_sat(volumes, loop)

                contagem += 1
                # if cfl < cfl_min:
                #     cfl = cfl*0.1
                #     self.cfl = cfl
                #     erro_cfl = self.calculate_sat(volumes, loop)
                    # erro_cfl = False
                # else:
                #     self.cfl = cfl
            if contagem > 100:
                print('cfl nao converge ')
                print(self.delta_t)
                print(cfl)
                import pdb; pdb.set_trace()

        t1 = time.time()
        dt = t1 - t0
        print('saiu de verificar cfl')
        print(f'tempo: {dt}')
        print(f'loop: {loop}\n')
