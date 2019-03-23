import numpy as np
from pymoab import core, types, rng, topo_util
import time
import os
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix, vstack, hstack, linalg, identity, find
import yaml
import io


# import solucao_adm_bifasico.solucao_adm_bifasico as sol_adm_bif

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

import importlib.machinery
loader = importlib.machinery.SourceFileLoader('pymoab_utils', utils_dir + '/pymoab_utils.py')
utpy = loader.load_module('pymoab_utils')
loader = importlib.machinery.SourceFileLoader('others_utils', utils_dir + '/others_utils.py')
oth = loader.load_module('others_utils').OtherUtils
loader = importlib.machinery.SourceFileLoader('prol_tpfa', utils_dir + '/prolongation_ams.py')
prol_tpfa = loader.load_module('prol_tpfa')
loader = importlib.machinery.SourceFileLoader('malha_adm', parent_dir + '/malha_adm.py')
adm_mesh = loader.load_module('malha_adm')

mb, mtu, tags_1, input_file, ADM, tempos_impr, contar_loop, contar_tempo, imprimir_sempre = utpy.load_adm_mesh()
adm_mesh = adm_mesh.malha_adm(mb, tags_1, input_file)
all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)
vv = mb.create_meshset()
mb.add_entities(vv, all_volumes)
os.chdir(bifasico_sol_multiescala_dir)
loader = importlib.machinery.SourceFileLoader('bif_utils', utils_dir + '/bif_utils.py')
bif_utils = loader.load_module('bif_utils').bifasico(mb, mtu, all_volumes)
loader = importlib.machinery.SourceFileLoader('sol_adm_bifasico', parent_dir + '/sol_adm_bifasico.py')
sol_adm_bif = loader.load_module('sol_adm_bifasico')

oth.gravity = bif_utils.gravity
oth1 = oth(mb, mtu)
sol_adm = sol_adm_bif.sol_adm_bifasico(mb, tags_1, oth.gravity, all_volumes)
all_ids_reord = mb.tag_get_data(tags_1['ID_reord_tag'], all_volumes, flat=True)
map_global = dict(zip(all_volumes, all_ids_reord))
boundary_faces = mb.tag_get_data(tags_1['FACES_BOUNDARY'], 0, flat=True)[0]
boundary_faces = mb.get_entities_by_handle(boundary_faces)
faces_in = rng.subtract(all_faces, boundary_faces)
name_tag_faces_boundary_meshsets = 'FACES_BOUNDARY_MESHSETS_LEVEL_'
boundary_faces_nv2 = mb.get_entities_by_handle(mb.tag_get_data(mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(2)), 0, flat=True)[0])
boundary_faces_nv3 = mb.get_entities_by_handle(mb.tag_get_data(mb.tag_get_handle(name_tag_faces_boundary_meshsets+str(3)), 0, flat=True)[0])
bound_faces_nv = [boundary_faces_nv2, boundary_faces_nv3]
wirebasket_numbers = [sol_adm.ni, sol_adm.nf, sol_adm.na, sol_adm.nv]
wirebasket_numbers_nv1 = [sol_adm.nint, sol_adm.nfac, sol_adm.nare, sol_adm.nver]
map_values_d = dict(zip(sol_adm.volumes_d, mb.tag_get_data(tags_1['P'], sol_adm.volumes_d, flat=True)))
map_values_n = dict(zip(sol_adm.volumes_n, mb.tag_get_data(tags_1['Q'], sol_adm.volumes_n, flat=True)))
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

def run_PMS(n1_adm, n2_adm, loop):
    if loop > 0:
        Pms2_ant = mb.tag_get_data(tags_1['PMS2'], sol_adm.wirebasket_elems, flat=True)


    As, s_grav = sol_adm.get_AS_structured(mb, tags_1, faces_in, all_volumes, bif_utils.mobi_in_faces_tag, map_global)
    # with io.open('As.yaml', 'w', encoding='utf8') as outfile:
    #         yaml.dump(As, outfile, default_flow_style=False, allow_unicode=True)
    # np.save('s_grav', s_grav)

    # with open("As.yaml", 'r') as stream:
    #     As = yaml.load(stream)
    # s_grav = np.load('s_grav.npy')

    OP1_AMS = oth.get_op_by_wirebasket_Tf(As['Tf'], wirebasket_numbers)

    # OP1_AMS = sol_adm.get_OP1_AMS_structured(As)

    OP1_ADM, OR1_ADM = sol_adm.organize_OP1_ADM(mb, OP1_AMS, all_volumes, tags_1)
    if loop > 2:
        import pdb; pdb.set_trace()
    # sp.save_npz('OP1_AMS', OP1_AMS)
    # sp.save_npz('OP1_ADM', OP1_ADM)
    # sp.save_npz('OR1_ADM', OR1_ADM)

    # OP1_AMS = sp.load_npz('OP1_AMS.npz')
    # OP1_ADM = sp.load_npz('OP1_ADM.npz')
    # OR1_ADM = sp.load_npz('OR1_ADM.npz')


    T1_AMS = sol_adm.OR1_AMS.dot(As['Tf'])
    T1_AMS = T1_AMS.dot(OP1_AMS)
    W_AMS=sol_adm.G.dot(T1_AMS)
    W_AMS=W_AMS.dot(sol_adm.G.transpose())
    OP2_AMS = oth.get_op_by_wirebasket_Tf_wire_coarse(W_AMS, wirebasket_numbers_nv1)
    OP2_AMS = sol_adm.G.transpose().dot(OP2_AMS)

    OP2_ADM, OR2_ADM = sol_adm.organize_OP2_ADM(mb, OP2_AMS, all_volumes, tags_1, n1_adm, n2_adm)
    if loop > 1:
        import pdb; pdb.set_trace()



    Tf2 = As['Tf'].copy()
    Tf2 = Tf2.tolil()
    Tf2, b = oth.set_boundary_dirichlet_matrix(map_global, map_values_d, s_grav, Tf2)
    b = oth.set_boundary_neumann(map_global, map_values_n, b)

    T1_ADM = OR1_ADM.dot(Tf2)
    T1_ADM = T1_ADM.dot(OP1_ADM)
    b1_ADM = OR1_ADM.dot(b)
    T1_ADM = T1_ADM.tocsc()

    PC1_ADM = oth.get_solution(T1_ADM, b1_ADM)
    Pms1 = OP1_ADM.dot(PC1_ADM)
    mb.tag_set_data(tags_1['PMS1'], sol_adm.wirebasket_elems, Pms1)
    Tf2 = Tf2.tocsc()
    Pf = oth.get_solution(Tf2, b)
    mb.tag_set_data(tags_1['PF'], sol_adm.wirebasket_elems, Pf)
    erro1 = 100*np.absolute((Pf - Pms1)/Pf)
    mb.tag_set_data(tags_1['ERRO1'], sol_adm.wirebasket_elems, erro1)

    T2_ADM = OR2_ADM.dot(T1_ADM)
    T2_ADM = T2_ADM.dot(OP2_ADM)
    b2_ADM = OR2_ADM.dot(b1_ADM)
    T2_ADM = T2_ADM.tocsc()

    PC2_ADM = oth.get_solution(T2_ADM, b2_ADM)
    Pms2 = OP2_ADM.dot(PC2_ADM)
    Pms2 = OP1_ADM.dot(Pms2)
    mb.tag_set_data(tags_1['PMS2'], sol_adm.wirebasket_elems, Pms2)
    erro2 = 100*np.absolute((Pf - Pms2)/Pf)
    mb.tag_set_data(tags_1['ERRO2'], sol_adm.wirebasket_elems, erro2)

    # T1_ADM = OR1_ADM.dot(As['Tf'])
    # T1_ADM = T1_ADM.dot(OP1_ADM)

    # import pdb; pdb.set_trace()

def run_2(t):
    elems_nv0 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([1]))
    # vertices_nv1 = rng.subtract(sol_adm.vertices, elems_nv0)
    vertices_nv1 = mb.get_entities_by_type_and_tag(meshset_vertices, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([2]))

    k = 0

    for vert in vertices_nv1:
        primal_id = mb.tag_get_data(tags_1['FINE_TO_PRIMAL1_CLASSIC'], vert, flat=True)[0]
        elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['FINE_TO_PRIMAL1_CLASSIC']]), np.array([primal_id]))
        faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
        faces = rng.subtract(faces, boundary_faces)
        faces_boundary = rng.intersect(faces, bound_faces_nv[k])
        # bif_utils.calculate_pcorr(mb, elems_in_meshset, vert, faces_boundary, faces, tags_1['PCORR1'], tags_1['PMS1'], sol_adm.volumes_d, sol_adm.volumes_n, tags_1, pcorr2_tag=tags_1['PCORR2'])
        # bif_utils.set_flux_pms_meshsets(elems_in_meshset, faces, faces_boundary, tags_1['PMS1'], tags_1['PCORR1'])
        bif_utils.calculate_pcorr(mb, elems_in_meshset, vert, faces_boundary, faces, tags_1['PCORR2'], tags_1['PMS2'], sol_adm.volumes_d, sol_adm.volumes_n, tags_1)
        bif_utils.set_flux_pms_meshsets(elems_in_meshset, faces, faces_boundary, tags_1['PMS2'], tags_1['PCORR2'])


    vertices_nv2 = mb.get_entities_by_type_and_tag(meshset_vertices_nv2, types.MBHEX, np.array([tags_1['l3_ID']]), np.array([3]))

    k=1
    for vert in vertices_nv2:
        primal_id = mb.tag_get_data(tags_1['FINE_TO_PRIMAL2_CLASSIC'], vert, flat=True)[0]
        elems_in_meshset = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([tags_1['FINE_TO_PRIMAL2_CLASSIC']]), np.array([primal_id]))
        faces = mtu.get_bridge_adjacencies(elems_in_meshset, 3, 2)
        faces = rng.subtract(faces, boundary_faces)
        faces_boundary = rng.intersect(faces, bound_faces_nv[k])
        bif_utils.calculate_pcorr(mb, elems_in_meshset, vert, faces_boundary, faces, tags_1['PCORR2'], tags_1['PMS2'], sol_adm.volumes_d, sol_adm.volumes_n, tags_1)
        bif_utils.set_flux_pms_meshsets(elems_in_meshset, faces, faces_boundary, tags_1['PMS2'], tags_1['PCORR2'])

    faces = mtu.get_bridge_adjacencies(elems_nv0, 3, 2)
    faces = rng.subtract(faces, boundary_faces)
    # bif_utils.set_flux_pms_elems_nv0(elems_nv0, faces, tags_1['PMS1'])
    bif_utils.set_flux_pms_elems_nv0(elems_nv0, faces, tags_1['PMS2'])
    bif_utils.cfl(faces_in)

def run_3(loop):
    bif_utils.calculate_sat(all_volumes, loop)
    bif_utils.set_lamb(all_volumes)
    bif_utils.set_mobi_faces(all_volumes, faces_in)
    adm_mesh.generate_adm_mesh(mb, all_volumes, loop=loop)
    sol_adm.get_AMS_TO_ADM_dict2(mb, tags_1)

def run(t, loop):
    n1_adm = len(np.unique(mb.tag_get_data(tags_1['l1_ID'], all_volumes, flat=True)))
    n2_adm = len(np.unique(mb.tag_get_data(tags_1['l2_ID'], all_volumes, flat=True)))
    run_PMS(n1_adm, n2_adm, loop)
    run_2(t)
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

if ADM == True:
    os.chdir(bifasico_sol_multiescala_dir)

    while verif:
        t0 = time.time()
        t, loop = run(t, loop)

        if cont_imp < len(tempos_impr):
            if bif_utils.vpi >= tempos_impr[cont_imp]:
                vpi = tempos_impr[cont_imp]
                delta_t2 = (vpi*bif_utils.V_total)/(bif_utils.flux_total_prod)
                bif_utils.vpi = vpi
                t -= bif_utils.delta_t
                bif_utils.delta_t = delta_t2
                t += delta_t2
                verif_imp = True
                verif_vpi = True
                cont_imp += 1

        if contar_loop:
            loop2 = loop
        if contar_tempo:
            t2 = t

        t1 = time.time()
        dt = t1-t0

        bif_utils.get_hist_ms(t, dt)
        ext_h5m = input_file + 'sol_multiescala_' + str(loop) + '.h5m'
        ext_vtk = input_file + 'sol_multiescala_' + str(loop) + '.vtk'

        if loop == 0 or imprimir_sempre:
            mb.write_file(ext_vtk, [vv])

        if verif_vpi:
            os.chdir(out_bif_solmult_dir)
            mb.write_file(ext_vtk, [vv])
            mb.write_file(ext_h5m)
            os.chdir(bifasico_sol_multiescala_dir)
            verif_vpi = False

        mb.write_file(ext_h5m)
        print(f'loop: {loop}')

        if t2 > bif_utils.total_time or loop2 > bif_utils.loops or bif_utils.vpi > 0.95:
            verif = False

        if verif:
            run_3(loop)

        testando = 'teste_' + str(loop) + '.vtk'
        mb.write_file(testando, [vv])




else:
    os.chdir(bifasico_sol_direta_dir)
    loader = importlib.machinery.SourceFileLoader('bifasico_sol_direta', parent_dir + '/bifasico_sol_direta.py')
    bifasico = loader.load_module('bifasico_sol_direta').sol_direta_bif(mb, mtu, all_volumes)
    # import pdb; pdb.set_trace()

    while verif:
        t0 = time.time()
        bifasico.solution_PF(sol_adm.wirebasket_elems, map_global, faces_in, tags_1)
        # bifasico.set_Pf(sol_adm.wirebasket_elems)
        bifasico.calculate_total_flux(all_volumes, faces_in)
        bifasico.cfl(faces_in)

        t += bifasico.delta_t
        loop += 1

        if cont_imp < len(tempos_impr):
            if bifasico.vpi >= tempos_impr[cont_imp]:
                vpi = tempos_impr[cont_imp]
                delta_t2 = (vpi*bifasico.V_total)/(bifasico.flux_total_producao)
                bifasico.vpi = vpi
                t -= bifasico.delta_t
                bifasico.delta_t = delta_t2
                t += delta_t2
                verif_imp = True
                verif_vpi = True
                cont_imp += 1

        if contar_loop:
            loop2 = loop
        if contar_tempo:
            t2 = t
        t1 = time.time()
        dt = t1-t0

        bifasico.get_hist(t, dt)

        ext_h5m = input_file + 'sol_direta_' + str(loop) + '.h5m'
        ext_vtk = input_file + 'sol_direta_' + str(loop) + '.vtk'

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

        if t2 > bifasico.total_time or loop2 > bifasico.loops or bifasico.vpi > 0.95:
            verif = False

        if verif:
            bifasico.calculate_sat(all_volumes, loop)
            bifasico.set_lamb(all_volumes)
            bifasico.set_mobi_faces(all_volumes, faces_in)


import pdb; pdb.set_trace()
import pdb; pdb.set_trace()
