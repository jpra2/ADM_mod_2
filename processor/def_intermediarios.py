import numpy as np
from pymoab import core, types, rng, topo_util
import os
import conversao as conv
import pdb

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')

# k_pe_to_m = conv.pe_to_m(1.0)
# k_md_to_m2 = conv.milidarcy_to_m2(1.0)
# k_psi_to_pa = conv.psi_to_Pa(1.0)
k_pe_to_m = 1.0
k_md_to_m2 = 1.0
k_psi_to_pa = 1.0

def def_inter(mb, dict_tags):
    intermediarios_tag = mb.tag_get_handle('intermediarios', 1, types.MB_TYPE_HANDLE, types.MB_TAG_MESH, True)
    elems_nivel2 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['l3_ID']]), np.array([2]))
    intermediarios_meshset = mb.create_meshset()
    mb.add_entities(intermediarios_meshset, elems_nivel2)
    mb.tag_set_data(intermediarios_tag, 0, intermediarios_meshset)

def injector_producer(mb):
    neuman_tag = mb.tag_get_handle('Q')
    press_tag = mb.tag_get_handle('P')
    volumes_n = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([neuman_tag]), np.array([None]))
    volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([press_tag]), np.array([None]))
    wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_injector_meshset = mb.create_meshset()
    wells_producer_meshset = mb.create_meshset()
    mb.add_entities(wells_producer_meshset, volumes_d)
    mb.add_entities(wells_injector_meshset, volumes_n)
    mb.tag_set_data(wells_injector_tag, 0, wells_injector_meshset)
    mb.tag_set_data(wells_producer_tag, 0, wells_producer_meshset)

def injector_producer_press(mb, mtu, gama_w, gama_o, gravity, all_nodes):
    press_tag = mb.tag_get_handle('P')
    volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([press_tag]), np.array([None]))
    values = mb.tag_get_data(press_tag, volumes_d, flat=True)
    wells_injector_tag = mb.tag_get_handle('WELLS_INJECTOR', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_producer_tag = mb.tag_get_handle('WELLS_PRODUCER', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    wells_injector_meshset = mb.create_meshset()
    wells_producer_meshset = mb.create_meshset()
    values = mb.tag_get_data(press_tag, volumes_d, flat=True)
    m = np.mean(values)
    injectors = []
    producers = []
    for i, v in enumerate(values):
        if v > m:
            injectors.append(volumes_d[i])
        else:
            producers.append(volumes_d[i])

    producers = rng.Range(producers)
    injectors = rng.Range(injectors)
    mb.add_entities(wells_producer_meshset, producers)
    mb.add_entities(wells_injector_meshset, injectors)
    mb.tag_set_data(wells_injector_tag, 0, wells_injector_meshset)
    mb.tag_set_data(wells_producer_tag, 0, wells_producer_meshset)

    # redefinir_pressoes(mb, injectors, producers, press_tag)

    if gravity:
        set_p_with_gravity(mb, mtu, press_tag, all_nodes, injectors, producers, gama_w, gama_o)

def redefinir_pressoes(mb, wells_injector, wells_producer, press_tag):
    p1 = 10.0
    p2 = 1.0

    mb.tag_set_data(press_tag, wells_injector, np.repeat(p1, len(wells_injector)))
    mb.tag_set_data(press_tag, wells_producer, np.repeat(p2, len(wells_producer)))

def cent(mb, mtu, all_volumes):
    centroids = np.array([mtu.get_average_position([v]) for v in all_volumes])
    cent_tag = mb.tag_get_handle('CENT', 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mb.tag_set_data(cent_tag, all_volumes, centroids)

def create_names_tags():
    os.chdir(flying_dir)
    names = ['d1', 'd2', 'l1_ID', 'l2_ID', 'l3_ID', 'P', 'Q', 'FACES_BOUNDARY', 'FACES_BOUNDARY_MESHSETS_LEVEL_2', 'FACES_BOUNDARY_MESHSETS_LEVEL_3',
             'FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC', 'PRIMAL_ID_1', 'PRIMAL_ID_2', 'L2_MESHSET', 'ID_reord_tag', 'CENT', 'AREA',
             'PERM', 'K_EQ', 'KHARM', 'finos0', 'intermediarios']

    names2 = ['WELLS_PRODUCER', 'WELLS_INJECTOR']
    nn = np.array(names)
    np.save('list_names_tags', nn)

def set_p_with_gravity(mb, mtu, press_tag, all_nodes, wells_injector, wells_producer, gama_w, gama_o):

    values_inj = (k_psi_to_pa)*mb.tag_get_data(press_tag, wells_injector, flat=True)
    values_prod = (k_psi_to_pa)*mb.tag_get_data(press_tag, wells_producer, flat=True)
    coords = (k_pe_to_m)*mb.get_coords(all_nodes)
    coords = coords.reshape([len(all_nodes), 3])
    maxs = coords.max(axis=0)
    Lz = maxs[2]
    # z_elems_d = -1*np.array([mtu.get_average_position([v])[2] for v in volumes_d])
    z_elems_inj = -1*(k_pe_to_m)*np.array([mtu.get_average_position([v])[2] for v in wells_injector])
    delta_z = z_elems_inj + Lz
    pressao = gama_w*(delta_z) + values_inj
    mb.tag_set_data(press_tag, wells_injector, pressao)
    z_elems_prod = -1*(k_pe_to_m)*np.array([mtu.get_average_position([v])[2] for v in wells_producer])
    delta_z = z_elems_prod + Lz
    pressao = gama_o*(delta_z) + values_prod
    mb.tag_set_data(press_tag, wells_producer, pressao)

def set_s_grav_faces(mb, keq_tag, all_volumes, all_centroids, faces_in, gamaf_tag, s_gravf_tag, ids_0, ids_1):

    keqs = mb.tag_get_data(keq_tag, faces_in, flat=True)
    gamasf = mb.tag_get_data(gamaf_tag, faces_in, flat=True)
    cents0 = all_centroids[ids_0]
    cents1 = all_centroids[ids_1]
    delta_z = cents1[:,2] - cents0[:,2]
    s_gravsf2 = gamasf*keqs*delta_z
    mb.tag_set_data(s_gravf_tag, faces_in, s_gravsf2)
    return s_gravsf2

def converter_keq(mb, k_eq_tag, faces_in):
    kk = 1.0
    kk = conv.darcy_to_m2(kk)
    kk2 = 1.0
    kk2 = conv.pe_to_m(kk2)
    kk2 = kk2**2
    keqs = mb.tag_get_data(k_eq_tag, faces_in, flat=True)
    keqs *= kk*kk2
    mb.tag_set_data(k_eq_tag, faces_in, keqs)

def convert_to_SI(info):

    mb = info['mb']
    all_faces = info['all_faces']
    all_volumes = info['all_volumes']
    volumes_d = info['volumes_d']
    cent_tag = info['cent_tag']
    press_tag = info['press_tag']
    area_tag = info['area_tag']
    perm_tag = info['perm_tag']
    k_eq_tag = info['k_eq_tag']

    areas = (k_pe_to_m**2)*mb.tag_get_data(area_tag, all_faces, flat=True)
    mb.tag_set_data(area_tag, all_faces, areas)

    press_values = (k_psi_to_pa)*mb.tag_get_data(press_tag, volumes_d, flat=True)
    mb.tag_set_data(press_tag, volumes_d, press_values)

    centroids = (k_pe_to_m)*mb.tag_get_data(cent_tag, all_volumes)
    mb.tag_set_data(cent_tag, all_volumes, centroids)

    # keqs = (k_pe_to_m**2)*(k_md_to_m2)*mb.tag_get_data(k_eq_tag, all_faces, flat=True)
    # mb.tag_set_data(k_eq_tag, all_faces, keqs)

    perms = mb.tag_get_data(perm_tag, all_volumes)

    for i, v in enumerate(all_volumes):
        perm = perms[i]
        mb.tag_set_data(perm_tag, v, k_md_to_m2*perm)

def set_k1_test(mb, perm_tag, all_volumes, all_centroids):
    k01 = 1.0
    k02 = 100.0

    k1 = [k01, 0.0, 0.0,
          0.0, k01, 0.0,
          0.0, 0.0, k01]

    k2 = [k02, 0.0, 0.0,
          0.0, k02, 0.0,
          0.0, 0.0, k02]

    b1 = np.array([np.array([200, 0, 0]), np.array([220, 200, 90])])
    b2 = np.array([np.array([400, 100, 0]), np.array([420, 300, 90])])

    inds0 = np.where(all_centroids[:,0] > b1[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > b1[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > b1[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < b1[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < b1[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < b1[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols1 = np.array(list(c1 & c2))

    inds0 = np.where(all_centroids[:,0] > b2[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > b2[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > b2[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < b2[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < b2[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < b2[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols2 = np.array(list(c1 & c2))

    volsk1 = rng.Range(np.array(all_volumes)[inds_vols1])
    volsk2 = rng.Range(np.array(all_volumes)[inds_vols2])

    volsk1 = rng.Range(set(volsk1) | set(volsk2))
    volsk2 = rng.subtract(all_volumes, volsk1)

    for v in volsk1:
        mb.tag_set_data(perm_tag, v, k1)

    for v in volsk2:
        mb.tag_set_data(perm_tag, v , k2)

    testk1_tag = mb.tag_get_handle('testk1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    testk2_tag = mb.tag_get_handle('testk2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mb.tag_set_data(testk1_tag, volsk1, np.repeat(k01, len(volsk1)))
    mb.tag_set_data(testk2_tag, volsk2, np.repeat(k02, len(volsk2)))

def criar_tags_bifasico(mb):
    sat_last_tag = mb.tag_get_handle('SAT_LAST', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    volume_tag = mb.tag_get_handle('VOLUME', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    sat_tag = mb.tag_get_handle('SAT', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    fw_tag = mb.tag_get_handle('FW', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    lamb_w_tag = mb.tag_get_handle('LAMB_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    lamb_o_tag = mb.tag_get_handle('LAMB_O', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    lbt_tag = mb.tag_get_handle('LBT', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mobi_in_faces_tag = mb.tag_get_handle('MOBI_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    fw_in_faces_tag = mb.tag_get_handle('FW_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    total_flux_tag = mb.tag_get_handle('TOTAL_FLUX', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    flux_w_tag = mb.tag_get_handle('FLUX_W', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    flux_in_faces_tag = mb.tag_get_handle('FLUX_IN_FACES', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    s_grav_tag = mb.tag_get_handle('S_GRAV', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    s_grav_volume_tag = mb.tag_get_handle('S_GRAV_VOLUME', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    dfds_tag = mb.tag_get_handle('DFDS', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    gamav_tag = mb.tag_get_handle('GAMAV', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    gamaf_tag = mb.tag_get_handle('GAMAF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    pms1_tag = mb.tag_get_handle('PMS1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    pms2_tag = mb.tag_get_handle('PMS2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    pf_tag = mb.tag_get_handle('PF', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    erro1_tag = mb.tag_get_handle('ERRO1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    erro2_tag = mb.tag_get_handle('ERRO2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    pcorr1_tag = mb.tag_get_handle('PCORR1', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    pcorr2_tag = mb.tag_get_handle('PCORR2', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)

def carregar_dados_anterior(data_loaded, loop):

    from utils import pymoab_utils as utpy

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.dirname(parent_dir)
    input_dir = os.path.join(parent_parent_dir, 'input')
    flying_dir = os.path.join(parent_parent_dir, 'flying')
    bifasico_dir = os.path.join(flying_dir, 'bifasico')
    bifasico_sol_direta_dir = os.path.join(bifasico_dir, 'sol_direta')
    bifasico_sol_multiescala_dir = os.path.join(bifasico_dir, 'sol_multiescala')
    utils_dir = os.path.join(parent_parent_dir, 'utils')

    ADM = data_loaded['ADM']
    input_file = data_loaded['input_file']

    if ADM:
        caminho = bifasico_sol_multiescala_dir
        ext_h5m = input_file + 'sol_multiescala_' + str(loop) + '.h5m'
    else:
        caminho = bifasico_sol_direta_dir
        ext_h5m = input_file + 'sol_direta_' + str(loop) + '.h5m'

    os.chdir(caminho)
    mb = core.Core()
    mtu = topo_util.MeshTopoUtil(mb)
    mb.load_file(ext_h5m)
    name_historico = 'historico_' + str(loop) + '.npy'
    historico = np.load(name_historico)
    t_loop = historico[1]

    os.chdir(flying_dir)
    list_names_tags = np.load('list_names_tags.npy')
    tags_1 = utpy.get_all_tags_2(mb, list_names_tags)
    tempos_impr = data_loaded['tempos_vpi_impressao']
    contar_loop = data_loaded['contar_loop']
    contar_tempo = data_loaded['contar_tempo']
    imprimir_sempre = data_loaded['imprimir_sempre']
    all_nodes, all_edges, all_faces, all_volumes = utpy.get_all_entities(mb)

    return mb, mtu, tags_1, input_file, ADM, tempos_impr, contar_loop, contar_tempo, imprimir_sempre, all_nodes, all_edges, all_faces, all_volumes, t_loop
