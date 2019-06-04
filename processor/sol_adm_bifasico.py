import numpy as np
from math import pi, sqrt
from pymoab import core, types, rng, topo_util
import time
import os
import scipy
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find, lil_matrix
import yaml
import scipy.sparse as sp


parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils

class sol_adm_bifasico:
    def __init__(self, mb, dict_tags, gravity, all_volumes, data_loaded):
        self.internos=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['d1']]), np.array([0]))
        self.faces=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['d1']]), np.array([1]))
        self.arestas=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['d1']]), np.array([2]))
        self.vertices=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['d1']]), np.array([3]))
        self.wirebasket_elems = list(self.internos) + list(self.faces) + list(self.arestas) + list(self.vertices)
        

        self.ni=len(self.internos)
        self.nf=len(self.faces)
        self.na=len(self.arestas)
        self.nv=len(self.vertices)
        self.wirebasket_numbers = [self.ni, self.nf, self.na, self.nv]

        self.nni=self.ni
        self.nnf=self.nni+self.nf
        self.nne=self.nnf+self.na
        self.nnv=self.nne+self.nv

        self.volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['P']]), np.array([None]))
        self.volumes_n = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['Q']]), np.array([None]))

        self.gravity = gravity
        self.AMS_TO_ADM, self.COL_TO_ADM_2, self.G, self.OR1_AMS, self.OR2_AMS = self.get_AMS_TO_ADM_dict(mb, dict_tags, all_volumes)
        # self.MPFA = mb.tag_get_data(dict_tags['MPFA'], 0, flat=True)[0]
        self.MPFA = data_loaded['MPFA']

    def get_b_ini(self, mb, dict_tags, all_volumes):
        lines = []
        cols = []
        data = []
        values_d = mb.tag_get_data(dict_tags['P'], self.volumes_d, flat=True)
        values_n = mb.tag_get_data(dict_tags['Q'], self.volumes_n, flat=True)
        ids_volumes_d = mb.tag_get_data(dict_tags['ID_reord_tag'], self.volumes_d, flat=True)
        ids_volumes_n = mb.tag_get_data(dict_tags['ID_reord_tag'], self.volumes_n, flat=True)

        for i, d in enumerate(self.volumes_d):
            ID_global=ids_volumes_d[i]
            press = values_d[i]
            lines.append(ID_global)
            cols.append(0)
            data.append(press)

            #b[ID_global]=press
        for i, n in enumerate(self.volumes_n):
            vazao2 = values_n[i]
            ID_global=ids_volumes_n[i]
            lines.append(ID_global)
            cols.append(0)
            data.append(vazao2)

        b_ini = csc_matrix((data,(lines,cols)),shape=(len(all_volumes),1))
        return b_ini

    def get_AS_structured(self, mb, dict_tags, faces_in, all_volumes, mobi_tag, map_volumes):
        ni = self.ni
        nf = self.nf
        na = self.na
        nv = self.nv


        lii=[]
        lif=[]
        lff=[]
        lfe=[]
        lee=[]
        lev=[]
        lvv=[]
        l_ii = []
        l_if = []
        l_ff = []
        l_fe = []
        l_ee = []
        l_ev = []
        l_vv = []

        cii=[]
        cif=[]
        cff=[]
        cfe=[]
        cee=[]
        cev=[]
        cvv=[]
        c_ii = []
        c_if = []
        c_ff = []
        c_fe = []
        c_ee = []
        c_ev = []
        c_vv = []

        dii=[]
        dif=[]
        dff=[]
        dfe=[]
        dee=[]
        dev=[]
        dvv=[]
        d_ii = []
        d_if = []
        d_ff = []
        d_fe = []
        d_ee = []
        d_ev = []
        d_vv = []

        all_s_gravs = mb.tag_get_data(dict_tags['S_GRAV'], faces_in, flat=True)
        s_grav = np.zeros(len(all_volumes))
        all_mobis = mb.tag_get_data(mobi_tag, faces_in, flat=True)
        # import pdb; pdb.set_trace()

        # ID_reordenado_tag = dict_tags['ID_reord_tag']
        # all_ids_reord = mb.tag_get_data(ID_reordenado_tag, all_volumes, flat=True)
        # # map_volumes = dict(zip(all_volumes, range(len(all_volumes))))
        lines_tf = []
        cols_tf = []
        data_tf = []
        lines_ttf = []
        cols_ttf = []
        data_ttf = []
        print("def As")
        ty=time.time()
        for i, f in enumerate(faces_in):
            keq = all_mobis[i]
            adjs = mb.get_adjacencies(f, 3)
            # Gid_1=int(mb.tag_get_data(ID_reordenado_tag,adjs[0]))
            # Gid_2=int(mb.tag_get_data(ID_reordenado_tag,adjs[1]))
            id_0 = map_volumes[adjs[0]]
            id_1 = map_volumes[adjs[1]]
            # Gid_1=all_ids_reord[id_0]
            # Gid_2=all_ids_reord[id_1]
            Gid_1 = id_0
            Gid_2 = id_1

            # Tf[[Gid_1, Gid_2], [Gid_2, Gid_1]] = [keq, keq]
            # Tf[Gid_1, Gid_1] -= keq
            # Tf[Gid_2, Gid_2] -= keq

            lines_tf.append(Gid_1)
            cols_tf.append(Gid_2)
            data_tf.append(keq)

            lines_tf.append(Gid_2)
            cols_tf.append(Gid_1)
            data_tf.append(keq)

            # lines_tf.append(Gid_1)
            # cols_tf.append(Gid_1)
            # data_tf.append(-keq)
            #
            # lines_tf.append(Gid_2)
            # cols_tf.append(Gid_1)
            # data_tf.append(-keq)

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

                # if Gid_1 in l_ii:
                #     index = l_ii.index(Gid_1)
                #     d_ii[index] -= keq
                # else:
                #     l_ii.append(Gid_1)
                #     c_ii.append(Gid_1)
                #     d_ii.append(-keq)
                #
                # if Gid_2 in l_ii:
                #     index = l_ii.index(Gid_2)
                #     d_ii[index] -= keq
                # else:
                #     l_ii.append(Gid_2)
                #     c_ii.append(Gid_2)
                #     d_ii.append(-keq)


            elif Gid_1<ni and Gid_2>=ni and Gid_2<ni+nf:
                lif.append(Gid_1)
                cif.append(Gid_2-ni)
                dif.append(keq)

                lii.append(Gid_1)
                cii.append(Gid_1)
                dii.append(-keq)

                # if Gid_1 in l_ii:
                #     index = l_ii.index(Gid_1)
                #     d_ii[index] -= keq
                # else:
                #     l_ii.append(Gid_1)
                #     c_ii.append(Gid_1)
                #     d_ii.append(-keq)

            elif Gid_2<ni and Gid_1>=ni and Gid_1<ni+nf:
                lif.append(Gid_2)
                cif.append(Gid_1-ni)
                dif.append(keq)

                lii.append(Gid_2)
                cii.append(Gid_2)
                dii.append(-keq)

                # if Gid_2 in l_ii:
                #     index = l_ii.index(Gid_2)
                #     d_ii[index] -= keq
                # else:
                #     l_ii.append(Gid_2)
                #     c_ii.append(Gid_2)
                #     d_ii.append(-keq)

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

                # if Gid_1-ni in l_ff:
                #     index = l_ff.index(Gid_1-ni)
                #     d_ff[index] -= keq
                # else:
                #     l_ff.append(Gid_1-ni)
                #     c_ff.append(Gid_1-ni)
                #     d_ff.append(-keq)
                #
                # if Gid_2-ni in l_ff:
                #     index = l_ff.index(Gid_2-ni)
                #     d_ff[index] -= keq
                # else:
                #     l_ff.append(Gid_2-ni)
                #     c_ff.append(Gid_2-ni)
                #     d_ff.append(-keq)

            elif Gid_1>=ni and Gid_1<ni+nf and Gid_2>=ni+nf and Gid_2<ni+nf+na:
                lfe.append(Gid_1-ni)
                cfe.append(Gid_2-ni-nf)
                dfe.append(keq)

                lff.append(Gid_1-ni)
                cff.append(Gid_1-ni)
                dff.append(-keq)

                # if Gid_1-ni in l_ff:
                #     index = l_ff.index(Gid_1-ni)
                #     d_ff[index] -= keq
                # else:
                #     l_ff.append(Gid_1-ni)
                #     c_ff.append(Gid_1-ni)
                #     d_ff.append(-keq)

            elif Gid_2>=ni and Gid_2<ni+nf and Gid_1>=ni+nf and Gid_1<ni+nf+na:
                lfe.append(Gid_2-ni)
                cfe.append(Gid_1-ni-nf)
                dfe.append(keq)

                lff.append(Gid_2-ni)
                cff.append(Gid_2-ni)
                dff.append(-keq)

                # if Gid_2-ni in l_ff:
                #     index = l_ff.index(Gid_2-ni)
                #     d_ff[index] -= keq
                # else:
                #     l_ff.append(Gid_2-ni)
                #     c_ff.append(Gid_2-ni)
                #     d_ff.append(-keq)

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

                # if Gid_1-ni-nf in l_ee:
                #     index = l_ee.index(Gid_1-ni-nf)
                #     d_ee[index] -= keq
                # else:
                #     l_ee.append(Gid_1-ni-nf)
                #     c_ee.append(Gid_1-ni-nf)
                #     d_ee.append(-keq)
                #
                # if Gid_2-ni-nf in l_ee:
                #     index = l_ee.index(Gid_2-ni-nf)
                #     d_ee[index] -= keq
                # else:
                #     l_ee.append(Gid_2-ni-nf)
                #     c_ee.append(Gid_2-ni-nf)
                #     d_ee.append(-keq)

            elif Gid_1>=ni+nf and Gid_1<ni+nf+na and Gid_2>=ni+nf+na:
                lev.append(Gid_1-ni-nf)
                cev.append(Gid_2-ni-nf-na)
                dev.append(keq)

                lee.append(Gid_1-ni-nf)
                cee.append(Gid_1-ni-nf)
                dee.append(-keq)

                # if Gid_1-ni-nf in l_ee:
                #     index = l_ee.index(Gid_1-ni-nf)
                #     d_ee[index] -= keq
                # else:
                #     l_ee.append(Gid_1-ni-nf)
                #     c_ee.append(Gid_1-ni-nf)
                #     d_ee.append(-keq)

            elif Gid_2>=ni+nf and Gid_2<ni+nf+na and Gid_1>=ni+nf+na:
                lev.append(Gid_2-ni-nf)
                cev.append(Gid_1-ni-nf-na)
                dev.append(keq)

                lee.append(Gid_2-ni-nf)
                cee.append(Gid_2-ni-nf)
                dee.append(-keq)

                # if Gid_2-ni-nf in l_ee:
                #     index = l_ee.index(Gid_2-ni-nf)
                #     d_ee[index] -= keq
                # else:
                #     l_ee.append(Gid_2-ni-nf)
                #     c_ee.append(Gid_2-ni-nf)
                #     d_ee.append(-keq)

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

            flux_grav = -all_s_gravs[i]
            s_grav[id_0] += flux_grav
            s_grav[id_1] -= flux_grav

        if self.gravity == False:
            s_grav = np.zeros(len(all_volumes))

        lines_tf += lines_ttf
        cols_tf += cols_ttf
        data_tf += data_ttf
        # lii += l_ii
        # cii += c_ii
        # dii += d_ii
        # lff += l_ff
        # cff += c_ff
        # dff += d_ff
        # lee += l_ee
        # cee += c_ee
        # dee += d_ee
        # lvv += l_vv
        # cvv += c_vv
        # dvv += d_vv

        print("took: ",time.time()-ty)
        print("get As")
        n = len(all_volumes)
        Tf = csc_matrix((data_tf,(lines_tf,cols_tf)),shape=(n, n))

        Aii=csc_matrix((dii,(lii,cii)),shape=(ni,ni))
        Aif=csc_matrix((dif,(lif,cif)),shape=(ni,nf))
        Aff=csc_matrix((dff,(lff,cff)),shape=(nf,nf))
        Afe=csc_matrix((dfe,(lfe,cfe)),shape=(nf,na))
        Aee=csc_matrix((dee,(lee,cee)),shape=(na,na))
        Aev=csc_matrix((dev,(lev,cev)),shape=(na,nv))
        # Avv=csc_matrix((dvv,(lvv,cvv)),shape=(nv,nv))
        Ivv=scipy.sparse.identity(nv)
        As = {}
        As['Aii'] = Aii
        As['Aif'] = Aif
        As['Aff'] = Aff
        As['Afe'] = Afe
        As['Aee'] = Aee
        As['Aev'] = Aev
        # As['Avv'] = Avv
        As['Ivv'] = Ivv
        As['Tf'] = Tf

        return As, s_grav

    def get_AS_structured_v2(self, mb, dict_tags, faces_in, all_volumes, mobi_tag, map_volumes):
        print('get As')

        all_s_gravs = mb.tag_get_data(dict_tags['S_GRAV'], faces_in, flat=True)
        s_grav = np.zeros(len(all_volumes))
        all_mobis = mb.tag_get_data(mobi_tag, faces_in, flat=True)

        lines_tf = []
        cols_tf = []
        data_tf = []

        print("def As")
        ty=time.time()
        for i, f in enumerate(faces_in):
            keq = all_mobis[i]
            adjs = mb.get_adjacencies(f, 3)
            Gid_1 = map_volumes[adjs[0]]
            Gid_2 = map_volumes[adjs[1]]

            lines_tf += [Gid_1, Gid_2]
            cols_tf += [Gid_2, Gid_1]
            data_tf += [keq, keq]

            flux_grav = -all_s_gravs[i]
            s_grav[Gid_1] += flux_grav
            s_grav[Gid_2] -= flux_grav

        if self.gravity == False:
            s_grav = np.zeros(len(all_volumes))

        print("took: ",time.time()-ty)
        n = len(all_volumes)
        Tf = csc_matrix((data_tf,(lines_tf,cols_tf)),shape=(n, n))
        Tf = Tf.tolil()
        d1 = np.array(Tf.sum(axis=1)).reshape(1, n)[0]*(-1)
        Tf.setdiag(d1)

        As = oth.get_Tmod_by_sparse_wirebasket_matrix(Tf, self.wirebasket_numbers)
        As['Tf'] = Tf
        return As, s_grav

    def get_AS_structured_v2_v2(self, mb, dict_tags, faces_in, all_volumes, mobi_tag, map_volumes):
        print('get As')

        all_s_gravs = mb.tag_get_data(dict_tags['S_GRAV'], faces_in, flat=True)
        s_grav = np.zeros(len(all_volumes))
        all_mobis = mb.tag_get_data(self.kdif_tag, faces_in, flat=True)

        lines_tf = []
        cols_tf = []
        data_tf = []

        print("def As")
        ty=time.time()
        for i, f in enumerate(faces_in):
            keq = all_mobis[i]
            adjs = mb.get_adjacencies(f, 3)
            Gid_1 = map_volumes[adjs[0]]
            Gid_2 = map_volumes[adjs[1]]

            lines_tf += [Gid_1, Gid_2]
            cols_tf += [Gid_2, Gid_1]
            data_tf += [keq, keq]

            flux_grav = -all_s_gravs[i]
            s_grav[Gid_1] += flux_grav
            s_grav[Gid_2] -= flux_grav

        if self.gravity == False:
            s_grav = np.zeros(len(all_volumes))

        print("took: ",time.time()-ty)
        n = len(all_volumes)
        Tf = csc_matrix((data_tf,(lines_tf,cols_tf)),shape=(n, n))
        Tf = Tf.tolil()
        d1 = np.array(Tf.sum(axis=1)).reshape(1, n)[0]*(-1)
        Tf.setdiag(d1)

        As = oth.get_Tmod_by_sparse_wirebasket_matrix(Tf, self.wirebasket_numbers)
        As['Tf'] = Tf
        return As, s_grav

    def get_OP1_AMS_structured(self, As):

        invAee=oth.lu_inv(As['Aee'])
        M2=-invAee*As['Aev']
        P=vstack([M2,As['Ivv']]) #P=np.concatenate((-np.dot(np.linalg.inv(Aee),Aev),Ivv), axis=0)

        invAff=oth.lu_inv(As['Aff'])
        M3=-invAff*As['Afe']*M2
        P=vstack([M3,P])   #P=np.concatenate((-np.dot(np.linalg.inv(Aff),np.dot(Afe,P[0:na,0:nv])),P), axis=0)

        invAii=oth.lu_inv(As['Aii'])
        P=vstack([-invAii*As['Aif']*M3,P]) ##P=np.concatenate((np.dot(-np.linalg.inv(Aii),np.dot(Aif,P[0:nf,0:nv])),P),axis=0)

        return P

    def get_AMS_TO_ADM_dict(self, mb, dict_tags, all_volumes):
        AMS_TO_ADM={}
        for v in self.vertices:
            ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
            ID_AMS=int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL1_CLASSIC'],v))
            AMS_TO_ADM[str(ID_AMS)] = ID_ADM

        D2_tag = dict_tags['d2']
        v=mb.create_meshset()
        mb.add_entities(v,self.vertices)
        inte=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([0]))
        fac=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([1]))
        are=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([2]))
        ver=mb.get_entities_by_type_and_tag(v, types.MBHEX, np.array([D2_tag]), np.array([3]))

        self.inte = inte
        self.fac = fac
        self.are = are
        self.ver = ver

        lines=[]
        cols=[]
        data=[]

        fine_to_primal1_classic_tag = dict_tags['FINE_TO_PRIMAL1_CLASSIC']
        nint=len(inte)
        nfac=len(fac)
        nare=len(are)
        nver=len(ver)

        self.nint = nint
        self.nfac = nfac
        self.nare = nare
        self.nver = nver

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
        G_nv1=csc_matrix((data,(lines,cols)),shape=(self.nv,self.nv))

        lines = []
        cols = []
        data = []

        for v in self.vertices:
            ID_AMS_1=int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL1_CLASSIC'],v))
            #ID_AMS_1=int(M1.mb.tag_get_data(fine_to_primal1_classic_tag,v))
            ID_AMS_2=int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL2_CLASSIC'],v))
            lines.append(ID_AMS_2)
            cols.append(ID_AMS_1)
            data.append(1)
            i+=1
        OR2_AMS=csc_matrix((data,(lines,cols)),shape=(nver,self.nv))

        lines=[]
        cols=[]
        data=[]
        for v in all_volumes:
             elem_Global_ID = int(mb.tag_get_data(dict_tags['ID_reord_tag'], v))
             AMS_ID = int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL1_CLASSIC'], v))
             lines.append(AMS_ID)
             cols.append(elem_Global_ID)
             data.append(1)
             #OR_AMS[AMS_ID][elem_Global_ID]=1
        OR1_AMS=csc_matrix((data,(lines,cols)),shape=(self.nv,len(all_volumes)))

        COL_TO_ADM_2={}
        # ver é o meshset dos vértices da malha dual grossa
        for i in range(nver):
            v=ver[i]
            ID_AMS=int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL2_CLASSIC'],v))
            ID_ADM=int(mb.tag_get_data(dict_tags['l2_ID'],v))
            COL_TO_ADM_2[str(i)] = ID_ADM



        return AMS_TO_ADM, COL_TO_ADM_2, G_nv1, OR1_AMS, OR2_AMS

    def get_AMS_TO_ADM_dict2(self, mb, dict_tags):
        AMS_TO_ADM={}
        for v in self.vertices:
            ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
            ID_AMS=int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL1_CLASSIC'],v))
            AMS_TO_ADM[str(ID_AMS)] = ID_ADM

        COL_TO_ADM_2={}
        # ver é o meshset dos vértices da malha dual grossa
        for i in range(self.nver):
            v = self.ver[i]
            ID_AMS = int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL2_CLASSIC'],v))
            ID_ADM = int(mb.tag_get_data(dict_tags['l2_ID'],v))
            COL_TO_ADM_2[str(i)] = ID_ADM

        self.AMS_TO_ADM = AMS_TO_ADM
        self.COL_TO_ADM_2 = COL_TO_ADM_2

    def organize_OP1_ADM(self, mb, OP1_AMS, all_volumes, dict_tags):

        AMS_TO_ADM = self.AMS_TO_ADM
        OP1_AMS = OP1_AMS.tolil()
        # lines=[]
        # cols=[]
        # data=[]
        nivel_0=mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['l3_ID']]), np.array([1]))

        print("get nivel 1___")
        # print("get nivel 1___")
        # matriz=scipy.sparse.find(OP1_AMS)
        # LIN=matriz[0]
        # COL=matriz[1]
        # DAT=matriz[2]
        # del matriz
        #
        # cont=0
        # for v in nivel_0:
        #     ID_ADM=int(mb.tag_get_data(dict_tags['l1_ID'],v))
        #     ID_global=int(mb.tag_get_data(dict_tags['ID_reord_tag'],v))
        #     lines.append(ID_global)
        #     cols.append(ID_ADM)
        #     data.append(1)
        #
        #     dd=np.where(LIN==ID_global)
        #     LIN=np.delete(LIN,dd,axis=0)
        #     COL=np.delete(COL,dd,axis=0)
        #     DAT=np.delete(DAT,dd,axis=0)

        print("set_nivel 0")

        # ID_ADM=[AMS_TO_ADM[str(k)] for k in COL]
        # lines=np.concatenate([lines,LIN])
        # cols=np.concatenate([cols,ID_ADM])
        # data=np.concatenate([data,DAT])
        #
        # import pdb; pdb.set_trace()
        #
        #
        # gids_nv1_adm = np.unique(mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True))
        # n1_adm = len(gids_nv1_adm)
        # OP_ADM=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),n1_adm))
        ta1 = time.time()
        OP1 = OP1_AMS.copy()
        OP1 = OP1.tolil()
        ID_global1 = mb.tag_get_data(dict_tags['ID_reord_tag'],nivel_0, flat=True)
        OP1[ID_global1]=csc_matrix((1,OP1.shape[1]))
        gids_nv1_adm = mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True)
        n1_adm = len(np.unique(gids_nv1_adm))

        IDs_ADM1=mb.tag_get_data(dict_tags['l1_ID'],nivel_0, flat=True)

        m=find(OP1)
        l1=m[0]
        c1=m[1]
        d1=m[2]
        lines=ID_global1
        cols=IDs_ADM1
        data=np.repeat(1,len(lines))

        ID_ADM1=[AMS_TO_ADM[str(k)] for k in c1]
        lines=np.concatenate([lines,l1])
        cols=np.concatenate([cols,ID_ADM1])
        data=np.concatenate([data,d1])

        opad1=csc_matrix((data,(lines,cols)),shape=(len(all_volumes),n1_adm))
        print("opad1",time.time()-ta1)
        OP_ADM=opad1

        print("set_nivel 0")

        # lines = []
        # cols = []
        # data = []

        # for v in all_volumes:
        #     elem_Global_ID = int(mb.tag_get_data(dict_tags['ID_reord_tag'], v, flat=True))
        #     elem_ID1 = int(mb.tag_get_data(dict_tags['l1_ID'], v, flat=True))
        #     lines.append(elem_ID1)
        #     cols.append(elem_Global_ID)
        #     data.append(1)
        #     #OR_ADM[elem_ID1][elem_Global_ID]=1
        cols = mb.tag_get_data(dict_tags['ID_reord_tag'], all_volumes, flat=True)
        lines = mb.tag_get_data(dict_tags['l1_ID'], all_volumes, flat=True)
        data = np.ones(len(lines))
        OR_ADM=csc_matrix((data,(lines,cols)),shape=(n1_adm,len(all_volumes)))

        return OP_ADM, OR_ADM

    def organize_OP2_ADM(self, mb, OP2_AMS, all_volumes, dict_tags, n1_adm, n2_adm):
        t0 = time.time()

        lines=[]
        cols=[]
        data=[]

        lines_or=[]
        cols_or=[]
        data_or=[]

        My_IDs_2=[]
        for v in all_volumes:
            ID_global=int(mb.tag_get_data(dict_tags['l1_ID'],v))
            if ID_global not in My_IDs_2:
                My_IDs_2.append(ID_global)
                ID_ADM=int(mb.tag_get_data(dict_tags['l2_ID'],v))
                nivel=mb.tag_get_data(dict_tags['l3_ID'],v)
                d1=mb.tag_get_data(dict_tags['d2'],v)
                ID_AMS = int(mb.tag_get_data(dict_tags['FINE_TO_PRIMAL1_CLASSIC'], v))
                # nivel<3 refere-se aos volumes na malha fina (nivel=1) e intermédiária (nivel=2)
                # d1=3 refere-se aos volumes que são vértice na malha dual de grossa
                if nivel<3:
                    lines.append(ID_global)
                    cols.append(ID_ADM)
                    data.append(1)
                    lines_or.append(ID_ADM)
                    cols_or.append(ID_global)
                    data_or.append(1)
                    #OP_ADM_2[ID_global][ID_ADM]=1
                else:
                    lines_or.append(ID_ADM)
                    cols_or.append(ID_global)
                    data_or.append(1)
                    for i in range(OP2_AMS[ID_AMS].shape[1]):
                        p=OP2_AMS[ID_AMS, i]
                        if p>0:
                            id_ADM=self.COL_TO_ADM_2[str(i)]
                            lines.append(ID_global)
                            cols.append(id_ADM)
                            data.append(float(p))
                            #OP_ADM_2[ID_global][id_ADM]=p

        print(time.time()-t0,"organize OP_ADM_2_______________________________::::::::::::")
        OP_ADM_2=csc_matrix((data,(lines,cols)),shape=(n1_adm,n2_adm))
        OR_ADM_2=csc_matrix((data_or,(lines_or,cols_or)),shape=(n2_adm,n1_adm))

        return OP_ADM_2, OR_ADM_2

    def set_boundary_dirichlet(self, T, b, map_values, map_local):
        t = T.shape[0]
        T2 = T.copy()
        T2 = T2.tolil()
        zeros = np.zeros(t)
        for v, value in map_values.items():
            gid = map_local[v]
            T2[gid] = zeros
            T2[gid, gid] = 1.0
            b[gid] = value

        return T2, b

    def set_boundary_neumann(self, b, map_values, map_local):

        for v, value in map_values.items():
            gid = map_local[v]
            b[gid] += value

        return b

    def get_T_ADM(self, T, OR_ADM, OP_ADM):
        T_ADM = OR_ADM.dot(T)
        T_ADM = T_ADM.dot(OP_ADM)

        return T_ADM

    def get_b_ADM(self, b, OR_ADM):
        b_ADM = OR_ADM.dot(b)

        return b_ADM

    def get_OP2_AMS(self, W_AMS):

        MPFA_NO_NIVEL_2 = self.MPFA

        nv1 = self.nv

        ni = self.nint
        nf = self.nfac
        na = self.nare
        nv = self.nver

        Aii=W_AMS[0:ni,0:ni]
        Aif=W_AMS[0:ni,ni:ni+nf]
        Aie=W_AMS[0:ni,ni+nf:ni+nf+na]
        Aiv=W_AMS[0:ni,ni+nf+na:ni+nf+na+nv]

        lines=[]
        cols=[]
        data=[]
        if MPFA_NO_NIVEL_2 == False:
            # for i in range(ni):
            #     lines.append(i)
            #     cols.append(i)
            #     data.append(float(Aie.sum(axis=1)[i])+float(Aiv.sum(axis=1)[i]))
            lines = np.arange(ni)
            cols = lines.copy()
            data = Aie.sum(axis=1) + Aiv.sum(axis=1)
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

            # for i in range(nf):
            #     lines.append(i)
            #     cols.append(i)
            #     data_fi.append(float(Afi.sum(axis=1)[i]))
            #     data_fv.append(float(Afv.sum(axis=1)[i]))

            lines = np.arange(nf)
            cols = lines.copy()
            data_fi = Afi.sum(axis=1)
            data_fv = Afv.sum(axis=1)

            Sfi=csc_matrix((data_fi,(lines,cols)),shape=(nf,nf))
            Aff += Sfi
            if MPFA_NO_NIVEL_2==False:
                Sfv=csc_matrix((data_fv,(lines,cols)),shape=(nf,nf))
                Aff +=Sfv
