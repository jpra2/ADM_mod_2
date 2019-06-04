import numpy as np
from pymoab import core, types, rng, topo_util
import time
import os
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
import yaml
import io
import pdb
import conversao as conv
from utils import pymoab_utils as utpy
from utils.others_utils import OtherUtils as oth
from utils import prolongation_ams as prol_tpfa
from processor import malha_adm as adm_mesh
from processor import sol_adm_bifasico as sol_adm_bif
from utils import bif_utils
from processor import def_intermediarios as definter

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')
bifasico_dir = os.path.join(flying_dir, 'bifasico')
bifasico_sol_direta_dir = os.path.join(bifasico_dir, 'sol_direta')
bifasico_sol_multiescala_dir = os.path.join(bifasico_dir, 'sol_multiescala')
utils_dir = os.path.join(parent_parent_dir, 'utils')
output_dir = os.path.join(parent_parent_dir, 'output')
out_bif_dir = os.path.join(output_dir, 'bifasico')
out_bif_soldir_dir =  os.path.join(out_bif_dir, 'sol_direta')
out_bif_solmult_dir =  os.path.join(out_bif_dir, 'sol_multiescala')

# k_pe_m = conv.pe_to_m(1.0)
# k_md_to_m2 = conv.milidarcy_to_m2(1.0)
k_pe_m = 1.0
k_md_to_m2 = 1.0

# import importlib.machinery

# loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
# utpy = loader.load_module('pymoab_utils')
# loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
# oth = loader.load_module('others_utils').OtherUtils
# loader = importlib.machinery.SourceFileLoader('prol_tpfa', utils_dir + '/prolongation_ams.py')
# prol_tpfa = loader.load_module('prol_tpfa')
# loader = importlib.machinery.SourceFileLoader('malha_adm', parent_dir + '/malha_adm.py')
# adm_mesh = loader.load_module('malha_adm')
definter.create_names_tags()
mb, mtu, tags_1, input_file, ADM, tempos_impr, contar_loop, contar_tempo, imprimir_sempre, data_loaded = utpy.load_adm_mesh()

# tags_1['l3_ID'] = mb.tag_get_handle('NIVEL_ID')
tags_1['l3_ID'] = mb.tag_get_handle('l3_ID')

definter.def_inter(mb, tags_1)

os.chdir(flying_dir)
faces_adjs_by_dual = np.load('faces_adjs_by_dual.npy')
intern_adjs_by_dual = np.load('intern_adjs_by_dual.npy')

adm_mesh = adm_mesh.malha_adm(mb, tags_1, input_file, mtu)
all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)

definter.injector_producer_press(mb, mtu, float(data_loaded['dados_bifasico']['gama_w']), float(data_loaded['dados_bifasico']['gama_o']), data_loaded['gravity'], all_nodes)
# definter.injector_producer(mb)
# tags = [tags_1['l1_ID'], tags_1['l2_ID']]
# for tag in tags:
#     all_gids = mb.tag_get_data(tag, all_volumes, flat=True)
#     minim = min(all_gids)
#     all_gids -= minim
#     mb.tag_set_data(tag, all_volumes, all_gids)

vv = mb.create_meshset()
mb.add_entities(vv, all_volumes)
os.chdir(bifasico_sol_multiescala_dir)
# loader = importlib.machinery.SourceFileLoader('bif_utils', utils_dir + '/bif_utils.py')
# bif_utils = loader.load_module('bif_utils').bifasico(mb, mtu, all_volumes)
bif_utils = bif_utils.bifasico(mb, mtu, all_volumes, data_loaded)
bif_utils.k_pe_m = k_pe_m

bif_utils.gravity = data_loaded['gravity']
# from processor import posteriori
# posteriori.redefinir_permeabilidades(mb, all_volumes, bif_utils.all_centroids, bif_utils.perm_tag)
# loader = importlib.machinery.SourceFileLoader('sol_adm_bifasico', parent_dir + '/sol_adm_bifasico.py')
# sol_adm_bif = loader.load_module('sol_adm_bifasico')

oth.gravity = bif_utils.gravity
oth1 = oth(mb, mtu)
sol_adm = sol_adm_bif.sol_adm_bifasico(mb, tags_1, oth.gravity, all_volumes, data_loaded)

all_ids_reord = mb.tag_get_data(tags_1['ID_reord_tag'], all_volumes, flat=True)
map_global = dict(zip(all_volumes, all_ids_reord))
boundary_faces = mb.tag_get_data(tags_1['FACES_BOUNDARY'], 0, flat=True)[0]
boundary_faces = mb.get_entities_by_handle(boundary_faces)
faces_in = rng.subtract(all_faces, boundary_faces)
bif_utils.all_faces_in = faces_in
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
boundary_faces_nv2 = mb.get_entities_by_handle(mb.tag_get_data(mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(2)), 0, flat=True)[0])
boundary_faces_nv3 = mb.get_entities_by_handle(mb.tag_get_data(mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(3)), 0, flat=True)[0])
bound_faces_nv = [boundary_faces_nv2, boundary_faces_nv3]
wirebasket_numbers = [sol_adm.ni, sol_adm.nf, sol_adm.na, sol_adm.nv]
wirebasket_numbers_nv1 = [sol_adm.nint, sol_adm.nfac, sol_adm.nare, sol_adm.nver]
meshset_vertices = mb.create_meshset()
mb.add_entities(meshset_vertices, sol_adm.vertices)
meshset_vertices_nv2 = mb.create_meshset()
mb.add_entities(meshset_vertices_nv2, sol_adm.ver)
n_levels = 3
tags_1['PMS1'] = mb.tag_get_handle('PMS1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
tags_1['PMS2'] = mb.tag_get_handle('PMS2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
tags_1['PF'] = mb.tag_get_handle('PF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
tags_1['ERRO1'] = mb.tag_get_handle('ERRO1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
tags_1['ERRO2'] = mb.tag_get_handle('ERRO2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
tags_1['PCORR1'] = mb.tag_get_handle('PCORR1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
tags_1['PCORR2'] = mb.tag_get_handle('PCORR2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags_1['PRIMAL_ID_1']]), np.array([None]))
meshsets_nv2 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags_1['PRIMAL_ID_2']]), np.array([None]))
meshsets_levels = [meshsets_nv1, meshsets_nv2]

# for i in range(len(meshsets_levels)):
#     name, tag = utpy.enumerar_volumes_nivel(mb, meshsets_nv1, i+2)
#     tags_1[name] = tag


####apagar
# mi = 0.0003
# mi = 1.0
# bif_utils.mi_w = mi #Paxs
# bif_utils.mi_o = mi
bif_utils.set_sat_in(all_volumes)
bif_utils.set_lamb(all_volumes)
tags_1['S_GRAV'] = bif_utils.s_grav_tag
# kk = 1e-30
# k = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
# for v in all_volumes:
#
#     # k = mb.tag_get_data(tags_1['PERM'], v, flat=True)
#     # k = conv.milidarcy_to_m2(k)
#     mb.tag_set_data(tags_1['PERM'], v, k)

def get_ls(all_volumes):
    v0 = all_volumes[0]
    points = mtu.get_bridge_adjacencies(v0, 3, 0)
    coords = (k_pe_m)*mb.get_coords(points).reshape([len(points), 3])
    maxs = coords.max(axis=0)
    mins = coords.min(axis=0)
    h0 = maxs - mins
    points = mtu.get_bridge_adjacencies(all_volumes, 3, 0)
    coords = (k_pe_m)*mb.get_coords(points).reshape([len(points), 3])
    maxs = coords.max(axis=0)
    mins = coords.min(axis=0)
    h = maxs - mins
    return h0, h

def set_keq(all_volumes, faces_in, tags):
    unis = np.array([np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])])
    ls, Ls = get_ls(all_volumes)
    A = np.array([ls[1]*ls[2], ls[0]*ls[2], ls[0]*ls[1]])
    Adjs = [mb.get_adjacencies(face, 3) for face in faces_in]
    all_ks = mb.tag_get_data(tags['PERM'], all_volumes)
    all_centroids = mb.tag_get_data(tags['CENT'], all_volumes)
    map_volumes = dict(zip(all_volumes, range(len(all_volumes))))
    keqs = np.zeros(len(faces_in))
    for i, f in enumerate(faces_in):
        # elem0 = Adjs[i][0]
        # elem1 = Adjs[i][1]
        # id0 = map_volumes[elem0]
        # id1 = map_volumes[elem1]
        # direction = all_centroids[id1] - all_centroids[id0]
        # norma = np.linalg.norm(direction)
        # uni = np.absolute(direction/norma)
        # a = np.dot(uni, A)
        # k0 = all_ks[id0].reshape([3, 3])
        # k1 = all_ks[id1].reshape([3, 3])
        # k0 = np.dot(np.dot(k0, uni), uni)
        # k1 = np.dot(np.dot(k1, uni), uni)
        # kharm = 2*(k0*k1)/(k0 + k1)
        # keq = a*kharm/norma
        # keqs[i] = keq
        keqs[i] = 1.0

    mb.tag_set_data(tags['K_EQ'], faces_in, keqs)


# set_keq(all_volumes, faces_in, tags_1)
# info = dict()
# info['mb'] = mb
# info['all_faces'] = all_faces
# info['all_volumes'] = all_volumes
# info['volumes_d'] = sol_adm.volumes_d
# info['cent_tag'] = tags_1['CENT']
# info['press_tag'] = tags_1['P']
# info['area_tag'] = tags_1['AREA2']
# info['perm_tag'] = tags_1['PERM']
# info['k_eq_tag'] = tags_1['K_EQ']

# def1.convert_to_SI(info)
# del info

os.chdir(flying_dir)
bif_utils.all_centroids = mb.tag_get_data(tags_1['CENT'], all_volumes)
# def1.set_k1_test(mb, tags_1['PERM'], all_volumes, bif_utils.all_centroids)
# mb.write_file('testt.vtk', [vv])

bif_utils.set_mobi_faces_ini(all_volumes, faces_in)
k00 = 2.0
k01 = 1e-3
# vazao_inj = 5000*k01 #bbl/dia
# vazao_inj = conv.bbldia_to_m3seg(vazao_inj) #m3/s
# vazao_inj = 100.0
# vazao_inj = conv.bbldia_to_m3seg(vazao_inj) #m3/s
# mb.tag_set_data(tags_1['Q'], sol_adm.volumes_n, np.repeat(-1.0*vazao_inj, len(sol_adm.volumes_n)))
# press_prod = 4000 #psi
# press_prod = conv.psi_to_Pa(press_prod) #Pascal
# Lz = 27.0
# press_prod = 1.0 #psi
# z_elems_d = -1*np.array([mtu.get_average_position([v])[2] for v in sol_adm.volumes_d])
# delta_z = z_elems_d + Lz
# press_prod = bif_utils.gama*(delta_z) + press_prod
# press_prod = conv.psi_to_Pa(press_prod) #Pascal
# mb.tag_set_data(tags_1['P'], sol_adm.volumes_d, np.repeat(press_prod, len(sol_adm.volumes_d)))
# mb.tag_set_data(tags_1['P'], sol_adm.volumes_d, press_prod)
map_values_d = dict(zip(sol_adm.volumes_d, mb.tag_get_data(tags_1['P'], sol_adm.volumes_d, flat=True)))
map_values_n = dict(zip(sol_adm.volumes_n, mb.tag_get_data(tags_1['Q'], sol_adm.volumes_n, flat=True)))

finos0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([1]))
# meshsets_nv1 = mb.get_entities_by_type_and_tag(0, types.MBENTITYSET, np.array([tags_1['PRIMAL_ID_1']]), np.array([None]))

def run_PMS(n1_adm, n2_adm, loop):

    print('entrou run_PMS')
    t0 = time.time()

    As, s_grav = sol_adm.get_AS_structured_v2(mb, tags_1, faces_in, all_volumes, bif_utils.mobi_in_faces_tag, map_global)
    # with io.open('As.yaml', 'w', encoding='utf8') as outfile:
    #         yaml.dump(As, outfile, default_flow_style=False, allow_unicode=True)
    # np.save('s_grav', s_grav)

    # with open("As.yaml", 'r') as stream:
    #     As = yaml.load(stream)
    # s_grav = np.load('s_grav.npy')
    # tmod = oth.get_Tmod_by_sparse_wirebasket_matrix(As['Tf'], wirebasket_numbers)

    # OP1_AMS = oth.get_op_by_wirebasket_Tf(As['Tf'], wirebasket_numbers)

    # OP1_AMS = oth.get_op_by_wirebasket_Tf(As['Tf'], wirebasket_numbers)
    # sp.save_npz('OP1_AMS', OP1_AMS)
    # OP1_AMS = sp.load_npz('OP1_AMS.npz')

    # OP1_AMS = sol_adm.get_OP1_AMS_structured(As)
    #
    # OP1_AMS = prol_tpfa.get_op_AMS_TPFA(As)
    OP1_AMS = prol_tpfa.get_op_AMS_TPFA_top(mb, faces_adjs_by_dual, intern_adjs_by_dual, sol_adm.ni, sol_adm.nf, bif_utils.mobi_in_faces_tag, As)
    OP1_ADM, OR1_ADM = sol_adm.organize_OP1_ADM(mb, OP1_AMS, all_volumes, tags_1)

    # sp.save_npz('OP1_AMS', OP1_AMS)
    # sp.save_npz('OP1_ADM', OP1_ADM)
    # sp.save_npz('OR1_ADM', OR1_ADM)

    # OP1_AMS = sp.load_npz('OP1_AMS.npz')
    # OP1_ADM = sp.load_npz('OP1_ADM.npz')
    # OR1_ADM = sp.load_npz('OR1_ADM.npz')

    if n1_adm == n2_adm:
        OP2_ADM = sp.identity(n1_adm)
        OR2_ADM = sp.identity(n1_adm)

    else:
        T1_AMS = sol_adm.OR1_AMS.dot(As['Tf'])
        T1_AMS = T1_AMS.dot(OP1_AMS)
        W_AMS=sol_adm.G.dot(T1_AMS)
        W_AMS=W_AMS.dot(sol_adm.G.transpose())
        OP2_AMS = oth.get_op_by_wirebasket_Tf_wire_coarse(W_AMS, wirebasket_numbers_nv1)
        OP2_AMS = sol_adm.G.transpose().dot(OP2_AMS)
        OP2_ADM, OR2_ADM = sol_adm.organize_OP2_ADM(mb, OP2_AMS, all_volumes, tags_1, n1_adm, n2_adm)

    # sp.save_npz('OP2_AMS', OP2_AMS)
    # sp.save_npz('OP2_ADM', OP2_ADM)
    # sp.save_npz('OR2_ADM', OR2_ADM)

    # OP2_AMS = sp.load_npz('OP2_AMS.npz')
    # OP2_ADM = sp.load_npz('OP2_ADM.npz')
    # OR2_ADM = sp.load_npz('OR2_ADM.npz')


    Tf2 = As['Tf'].copy()
    Tf2 = Tf2.tolil()

    bif_utils.Tf = Tf2
    Tf2, b = oth.set_boundary_dirichlet_matrix(map_global, map_values_d, s_grav, Tf2)
    # Tf2, b = oth.set_boundary_dirichlet_matrix_v02(ids_volumes_d, vals_d, s_grav, Tf2)
    b = oth.set_boundary_neumann(map_global, map_values_n, b)
    # b = oth.set_boundary_neumann_v02(ids_volumes_n, vals_n, b)


    T1_ADM = OR1_ADM.dot(Tf2)
    T1_ADM = T1_ADM.dot(OP1_ADM)
    b1_ADM = OR1_ADM.dot(b)
    T1_ADM = T1_ADM.tocsc()

    # PC1_ADM = oth.get_solution(T1_ADM, b1_ADM)
    # # PC1_ADM = oth.get_solution_gmres_scipy(T1_ADM, b1_ADM)
    # Pms1 = OP1_ADM.dot(PC1_ADM)
    # mb.tag_set_data(tags_1['PMS1'], sol_adm.wirebasket_elems, Pms1)

    # Tf2 = Tf2.tocsc()
    # Pf = oth.get_solution(Tf2, b)
    # mb.tag_set_data(tags_1['PF'], sol_adm.wirebasket_elems, Pf)
    # erro1 = 100*np.absolute((Pf - Pms1)/Pf)
    # mb.tag_set_data(tags_1['ERRO1'], sol_adm.wirebasket_elems, erro1)


    T2_ADM = OR2_ADM.dot(T1_ADM)
    T2_ADM = T2_ADM.dot(OP2_ADM)
    b2_ADM = OR2_ADM.dot(b1_ADM)
    T2_ADM = T2_ADM.tocsc()

    PC2_ADM = oth.get_solution(T2_ADM, b2_ADM)
    # PC2_ADM = oth.get_solution_gmres_scipy(T2_ADM, b2_ADM)
    Pms2 = OP2_ADM.dot(PC2_ADM)
    Pms2 = OP1_ADM.dot(Pms2)
    mb.tag_set_data(tags_1['PMS2'], sol_adm.wirebasket_elems, Pms2)
    # erro2 = 100*np.absolute((Pf - Pms2)/Pf)
    # mb.tag_set_data(tags_1['ERRO2'], sol_adm.wirebasket_elems, erro2)

    # T1_ADM = OR1_ADM.dot(As['Tf'])
    # T1_ADM = T1_ADM.dot(OP1_ADM)

    t1 = time.time()
    dt = t1-t0
    # print(f'run pms {dt} \n')

def run_2(t):
    print('entrou run2')
    tini = time.time()

    elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([1]))
    # vertices_nv1 = rng.subtract(sol_adm.vertices, elems_nv0)
    vertices_nv1 = mb.get_entities_by_type_and_tag(meshset_vertices, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([2]))

    k = 0
    cont = 0

    for vert in vertices_nv1:
        t00 = time.time()
        primal_id = mb.tag_get_data(tags_1['FINE_TO_PRIMAL1_CLASSIC'], vert, flat=True)[0]
        elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['FINE_TO_PRIMAL1_CLASSIC']]), np.array([primal_id]))
        faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
        faces = rng.subtract(faces, boundary_faces)
        faces_boundary = rng.intersect(faces, bound_faces_nv[k])
        t01 = time.time()
        t02 = time.time()
        bif_utils.calculate_pcorr(mb, elems_in_meshset, vert, faces_boundary, faces, tags_1['PCORR2'], tags_1['PMS2'], sol_adm.volumes_d, sol_adm.volumes_n, tags_1)
        # bif_utils.calculate_pcorr_v4(elems_in_meshset, tags_1['PCORR2'], tags_1)
        t03 = time.time()
        bif_utils.set_flux_pms_meshsets(elems_in_meshset, faces, faces_boundary, tags_1['PMS2'], tags_1['PCORR2'])
        t04 = time.time()
        dt0 = t01 - t00
        dt1= t03 - t02
        dt2= t04 - t03
        dtt = t04 - t01
        # print(f'tempo total {dtt}')
        # print(f'tempo hd {dt0}')
        # print(f'tempo pcorr {dt1}')
        # print(f'tempo fluxo_ms {dt2} \n')

    vertices_nv2 = mb.get_entities_by_type_and_tag(meshset_vertices_nv2, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([3]))

    t0 = time.time()

    k=1
    for vert in vertices_nv2:
        primal_id = mb.tag_get_data(tags_1['FINE_TO_PRIMAL2_CLASSIC'], vert, flat=True)[0]
        elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['FINE_TO_PRIMAL2_CLASSIC']]), np.array([primal_id]))
        faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
        faces = rng.subtract(faces, boundary_faces)
        faces_boundary = rng.intersect(faces, bound_faces_nv[k])
        bif_utils.calculate_pcorr(mb, elems_in_meshset, vert, faces_boundary, faces, tags_1['PCORR2'], tags_1['PMS2'], sol_adm.volumes_d, sol_adm.volumes_n, tags_1)
        # bif_utils.calculate_pcorr_v4(elems_in_meshset, tags_1['PCORR2'], tags_1)
        bif_utils.set_flux_pms_meshsets(elems_in_meshset, faces, faces_boundary, tags_1['PMS2'], tags_1['PCORR2'])

    t1 = time.time()
    dt = t1 - t0
    # print(f'tempo nv2 fluxo {dt}\n')

    faces = mtu.get_bridge_adjacencies(elems_nv0, 3, 2)
    faces = rng.subtract(faces, boundary_faces)
    # bif_utils.set_flux_pms_elems_nv0(elems_nv0, faces, tags_1['PMS1'])
    t0 = time.time()
    bif_utils.set_flux_pms_elems_nv0(elems_nv0, faces, tags_1['PMS2'])
    t1 = time.time()
    dt = t1 - t0
    tend = time.time()
    dtt = tend - tini
    # print(f'tempo nv0 fluxo {dt}\n')
    bif_utils.calc_cfl(faces_in)
    bif_utils.verificar_cfl(all_volumes, loop)
    print(f'tempo total: {dtt}')
    print('saiu run2')

def run_2_v2(t):
    print('entrou run2')
    t0 = time.time()
    elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([1]))
    vertices_nv1 = mb.get_entities_by_type_and_tag(meshset_vertices, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([0]))
    vertices_nv1 = rng.subtract(mb.get_entities_by_handle(meshset_vertices), vertices_nv1)
    bif_utils.calculate_pcorr_v3(mb, bound_faces_nv[0], tags_1['PMS2'], tags_1['PCORR2'], vertices_nv1, tags_1, all_volumes)
    mb.write_file('exemplo.vtk', [vv])
    pdb.set_trace()

def run_3(loop):
    print('entrou run3')
    t0 = time.time()
    # bif_utils.calculate_sat(all_volumes, loop)

    t1 = time.time()
    bif_utils.set_lamb(all_volumes)
    t2 = time.time()
    bif_utils.set_mobi_faces(all_volumes, faces_in, finos0=finos0)
    bif_utils.set_finos(finos0, meshsets_nv1)
    t3 = time.time()
    adm_mesh.generate_adm_mesh(mb, all_volumes, loop=loop)
    t4 = time.time()
    sol_adm.get_AMS_TO_ADM_dict2(mb, tags_1)
    dt0 = t1-t0
    dt1 = t2-t1
    dt2 = t3-t2
    dt3 = t4 - t3

    print(f'tempo calculate_sat {dt0}')
    print(f'tempo set_lamb {dt1}')
    print(f'tempo set_mobi_faces {dt2}')
    print(f'tempo generate_adm_mesh {dt3}\n')
    print('saiu run3')

def run(t, loop):
    # n1_adm = len(np.unique(mb.tag_get_data(tags_1['l1_ID'], all_volumes, flat=True)))
    # n2_adm = len(np.unique(mb.tag_get_data(tags_1['l2_ID'], all_volumes, flat=True)))
    n1_adm = mb.tag_get_data(tags_1['l1_ID'], all_volumes, flat=True).max() + 1
    n2_adm = mb.tag_get_data(tags_1['l2_ID'], all_volumes, flat=True).max() + 1
    elems_nv0 = len(mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([1])))
    with open('volumes_finos.txt', 'a+') as fil:
        fil.write(str(n1_adm)+' '+str(n2_adm)+' '+str(elems_nv0)+' '+str(loop)+'\n')



    run_PMS(n1_adm, n2_adm, loop)
    run_2(t)
    mb.write_file('testtt.vtk',[vv])
    t2 = t+bif_utils.delta_t
    loop+=1
    return t2, loop

loops = 10
t = 0
loop = 0
t2 = 0
loop2 = 0
verif = True
verif_imp = False
cont_imp = 0
verif_vpi = False
contador = 0
ver9 = 1

vf = mb.create_meshset()
mb.add_entities(vf, all_faces)

if ADM:
    list_tempos = []
    tini = time.time()
    os.chdir(bifasico_sol_multiescala_dir)
    with open('volumes_finos.txt', 'w') as fil:
        fil.write('n1_adm n2_adm len(finos) loop\n')
        pass

    with open('tempos_simulacao_adm.txt', 'w') as fil:
        pass

    while verif:
        t0 = time.time()
        contador += 1
        # if contador > 29:
        #     contador = 0
        #     pdb.set_trace()

        t, loop = run(t, loop)



        # if cont_imp < len(tempos_impr):
        #     if bif_utils.vpi >= tempos_impr[cont_imp]:
        #         vpi = tempos_impr[cont_imp]
        #         delta_t2 = (vpi*bif_utils.V_total)/(bif_utils.flux_total_prod)
        #         bif_utils.vpi = vpi
        #         t -= bif_utils.delta_t
        #         bif_utils.delta_t = delta_t2
        #         bif_utils.calculate_sat_vpi(all_volumes)
        #         t += delta_t2
        #         verif_imp = True
        #         verif_vpi = True
        #         cont_imp += 1

        if contar_loop:
            loop2 = loop
        if contar_tempo:
            t2 = t

        t1 = time.time()
        dt = t1-t0

        print(f'loop: {loop-1}')
        print(f'delta_t: {bif_utils.delta_t}\n')

        bif_utils.get_hist_ms(t, dt, loop-1)
        ext_h5m = input_file + 'sol_multiescala_' + str(loop-1) + '.h5m'
        ext_vtk = input_file + 'sol_multiescala_' + str(loop-1) + '.vtk'

        # if loop == 1 or imprimir_sempre:
        #     mb.write_file(ext_vtk, [vv])

        if verif_vpi:
            os.chdir(out_bif_solmult_dir)
            mb.write_file(ext_vtk, [vv])
            mb.write_file(ext_h5m)
            os.chdir(bifasico_sol_multiescala_dir)
            verif_vpi = False

        # mb.write_file(ext_h5m)
        print(f'loop: {loop}')

        if t2 > bif_utils.total_time or loop2 > bif_utils.loops or bif_utils.vpi > 0.99:
            verif = False

        if verif:
            run_3(loop)

        t3 = time.time()
        dt = t3-t0

        # testando = 'teste_' + str(loop) + '.vtk'
        # mb.write_file(testando, [vv])

        with open('tempos_simulacao_adm.txt', 'a+') as fil:
            fil.write(str(dt)+'\n')

        if contador % 3 == 0:
            os.system('clear')

        if contador % 10 == 0:
            mb.write_file(ext_vtk, [vv])



    tfim = time.time()

elif ADM == False:
    os.chdir(bifasico_sol_direta_dir)
    # import importlib
    # loader = importlib.machinery.SourceFileLoader('bifasico_sol_direta', parent_dir + '/bifasico_sol_direta.py')
    # bifasico = loader.load_module('bifasico_sol_direta').sol_direta_bif(mb, mtu, all_volumes, data_loaded)
    from processor.bifasico_sol_direta import sol_direta_bif as bifasico
    bifasico = bifasico(mb, mtu, all_volumes, data_loaded)
    bifasico.gravity = bif_utils.gravity
    bifasico.mi_w = bif_utils.mi_w #Paxs
    bifasico.mi_o = bif_utils.mi_o
    bifasico.gama_w = bif_utils.gama_w
    bifasico.gama_o = bif_utils.gama_o
    bifasico.Sor = bif_utils.Sor
    bifasico.Swc = bif_utils.Swc
    bifasico.nw = bif_utils.nw
    bifasico.no = bif_utils.no
    bifasico.loops = bif_utils.loops
    bifasico.total_time = bif_utils.total_time
    bifasico.gama = bif_utils.gama
    with open('tempos_simulacao_direta.txt', 'w') as fil:
        pass

    tini = time.time()

    while verif:
        contador += 1
        list_tempos = []

        t0 = time.time()
        bifasico.solution_PF(sol_adm.wirebasket_elems, map_global, faces_in, tags_1)
        # bifasico.set_Pf(sol_adm.wirebasket_elems)
        bifasico.calculate_total_flux(all_volumes, faces_in)
        bifasico.calc_cfl(faces_in)
        bifasico.verificar_cfl(all_volumes, loop)
        print('loop: ', loop)
        print('delta_t: ', bifasico.delta_t, '\n')


        t += bifasico.delta_t
        loop += 1

        # if cont_imp < len(tempos_impr):
        #     if bifasico.vpi >= tempos_impr[cont_imp]:
        #         vpi = tempos_impr[cont_imp]
        #         delta_t2 = (vpi*bifasico.V_total)/(bifasico.flux_total_producao)
        #         bifasico.vpi = vpi
        #         t -= bifasico.delta_t
        #         bifasico.delta_t = delta_t2
        #         t += delta_t2
        #         bifasico.calculate_sat_vpi(all_volumes)
        #         verif_imp = True
        #         verif_vpi = True
        #         cont_imp += 1

        if contar_loop:
            loop2 = loop
        if contar_tempo:
            t2 = t
        t1 = time.time()
        dt = t1-t0

        bifasico.get_hist(t, dt, loop-1)

        ext_h5m = input_file + 'sol_direta_' + str(loop-1) + '.h5m'
        ext_vtk = input_file + 'sol_direta_' + str(loop-1) + '.vtk'

        if loop == 0 or imprimir_sempre:
            mb.write_file(ext_vtk, [vv])

        # if imprimir_sempre:
        #     mb.write_file(ext_vtk, [vv])

        if verif_vpi:
            os.chdir(out_bif_soldir_dir)
            mb.write_file(ext_vtk, [vv])
            mb.write_file(ext_h5m)
            os.chdir(bifasico_sol_direta_dir)
            verif_vpi = False

        mb.write_file(ext_h5m)
        print(f'loop: {loop}')

        if t2 > bifasico.total_time or loop2 > bifasico.loops or bifasico.vpi > 0.99:
            verif = False

        if verif:
            # bifasico.calculate_sat(all_volumes, loop)
            # bifasico.verificar_cfl(all_volumes, loop)
            bifasico.set_lamb(all_volumes)
            bifasico.set_mobi_faces(all_volumes, faces_in)

        t3 = time.time()
        dt = t3-t0
        with open('tempos_simulacao_direta.txt', 'a+') as fil:
            fil.write(str(dt)+'\n')

        if contador % 3 == 0:
            os.system('clear')

        # if contador % 10 == 0:
        #     mb.write_file(ext_vtk, [vv])

    tfim = time.time()


import pdb; pdb.set_trace()
import pdb; pdb.set_trace()
