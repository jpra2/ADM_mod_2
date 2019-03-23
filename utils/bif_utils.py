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
bifasico_dir = os.path.join(flying_dir, 'bifasico')
fly_bif_mult_dir = os.path.join(bifasico_dir, 'sol_multiescala')


import importlib.machinery
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils

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
        self.cent_tag = mb.tag_get_handle('CENT')
        self.dfds_tag = mb.tag_get_handle('DFDS')
        self.finos_tag = mb.tag_get_handle('finos')
        self.mb = mb
        self.mtu = mtu
        self.gama = self.gama_w + self.gama_o
        self.fimin = mb.tag_get_data(self.phi_tag, all_volumes, flat=True).min()
        self.Vmin = mb.tag_get_data(self.volume_tag, all_volumes, flat=True).min()
        historico = np.array(['vpi', 'tempo', 'prod_agua', 'prod_oleo', 'wor', 'dt'])
        np.save('historico', historico)
        self.V_total = mb.tag_get_data(self.volume_tag, all_volumes, flat=True)
        self.V_total = float((self.V_total*mb.tag_get_data(self.phi_tag, all_volumes, flat=True)).sum())
        self.vpi = 0.0

    def cfl(self, all_faces_in):
        """
        cfl usando fluxo maximo
        """
        cfl = 0.6
        qmax = np.absolute(self.mb.tag_get_data(self.flux_in_faces_tag, all_faces_in, flat=True)).max()
        dfdsmax = self.mb.tag_get_data(self.dfds_tag, all_faces_in, flat=True).max()
        self.flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True).sum()
        self.delta_t = cfl*(self.fimin*self.Vmin)/float(qmax*dfdsmax)
        vpi = (self.flux_total_prod*self.delta_t)/self.V_total
        self.vpi += vpi

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

    def set_mobi_faces_ini(self, all_volumes, all_faces_in):
        lim = 1e-5

        all_lbt = self.mb.tag_get_data(self.lbt_tag, all_volumes, flat=True)
        map_lbt = dict(zip(all_volumes, all_lbt))
        del all_lbt
        all_centroids = self.mb.tag_get_data(self.cent_tag, all_volumes)
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

        for i, face in enumerate(all_faces_in):
            elems = self.mb.get_adjacencies(face, 3)
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

    def set_mobi_faces(self, volumes, faces):
        lim = 1e-5

        """
        seta a mobilidade nas faces uma vez calculada a pressao corrigida
        """
        finos_val = self.mb.tag_get_handle('FINOS_VAL', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
        lim_sat = 0.001
        finos = self.mb.create_meshset()
        self.mb.tag_set_data(self.finos_tag, 0, finos)
        self.mb.add_entities(finos, self.wells_injector)
        self.mb.add_entities(finos, self.wells_producer)
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
        vols_finos = self.mb.get_entities_by_handle(finos)
        self.mb.tag_set_data(finos_val, vols_finos, np.repeat(1.0, len(vols_finos)))

    def set_flux_pms_meshsets(self, volumes, faces, faces_boundary, pms_tag, pcorr_tag, pcorr2_tag=None):

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

    def set_flux_pms_elems_nv0(self, volumes, faces, pms_tag):

        mobi_in_faces = self.mb.tag_get_data(self.mobi_in_faces_tag, faces, flat=True)
        fws_faces = self.mb.tag_get_data(self.fw_in_faces_tag, faces, flat=True)
        s_gravs_faces = self.mb.tag_get_data(self.s_grav_tag, faces, flat=True)
        map_local = dict(zip(volumes, range(len(volumes))))
        pmss = self.mb.tag_get_data(pms_tag, volumes, flat=True)

        fluxos = np.zeros(len(volumes))
        fluxos_w = fluxos.copy()
        flux_in_faces = np.zeros(len(faces))

        map_local = dict(zip(volumes, range(len(volumes))))

        flux_in_faces = np.zeros(len(faces))
        map_id_faces = dict(zip(faces, range(len(faces))))

        for i, face in enumerate(faces):
            elems = self.mb.get_adjacencies(face, 3)
            mobi = mobi_in_faces[i]
            s_grav = s_gravs_faces[i]
            fw = fws_faces[i]
            ps = self.mb.tag_get_data(pms_tag, elems, flat=True)
            flux = (ps[1] - ps[0])*mobi
            if self.gravity == True:
                flux += s_grav
            if elems[0] in volumes:
                local_id = map_local[elems[0]]
                fluxos[local_id] += flux
                fluxos_w[local_id] += flux*fw
            if elems[1] in volumes:
                local_id = map_local[elems[1]]
                fluxos[local_id] -= flux
                fluxos_w[local_id] -= flux*fw
            flux_in_faces[i] = flux

        self.mb.tag_set_data(self.total_flux_tag, volumes, fluxos)
        self.mb.tag_set_data(self.flux_w_tag, volumes, fluxos_w)
        self.mb.tag_set_data(self.flux_in_faces_tag, faces, flux_in_faces)

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

    def get_hist_ms(self, t, dt):
        flux_total_prod = self.mb.tag_get_data(self.total_flux_tag, self.wells_producer, flat=True)
        fws = self.mb.tag_get_data(self.fw_tag, self.wells_producer, flat=True)

        qw = (flux_total_prod*fws).sum()*self.delta_t
        qo = (flux_total_prod.sum()- qw)*self.delta_t
        wor = qw/float(qo)

        hist = np.array([self.vpi, t, qw, qo, wor, dt])
        historico = np.load('historico.npy')
        historico = np.append(historico, hist)
        np.save('historico', historico)
