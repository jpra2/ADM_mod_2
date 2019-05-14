import numpy as np
from pymoab import core, types, rng, topo_util
import os
import conversao as conv

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')

k_pe_to_m = conv.pe_to_m(1.0)
k_md_to_m2 = conv.milidarcy_to_m2(1.0)
k_psi_to_pa = conv.psi_to_Pa(1.0)

def def_inter(mb, dict_tags):
    intermediarios_tag = mb.tag_get_handle('intermediarios', 1, types.MB_TYPE_HANDLE, types.MB_TAG_SPARSE, True)
    elems_nivel2 = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([dict_tags['l3_ID']]), np.array(2))
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

def injector_producer_press(mb):
    press_tag = mb.tag_get_handle('P')
    volumes_d = mb.get_entities_by_type_and_tag(0, types.MBHEX, np.array([press_tag]), np.array([None]))
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

def cent(mb, mtu, all_volumes):
    centroids = np.array([mtu.get_average_position([v]) for v in all_volumes])
    cent_tag = mb.tag_get_handle('CENT', 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mb.tag_set_data(cent_tag, all_volumes, centroids)

def create_names_tags():
    os.chdir(flying_dir)
    names = ['d1', 'd2', 'l1_ID', 'l2_ID', 'l3_ID', 'P', 'Q', 'FACES_BOUNDARY', 'FACES_BOUNDARY_MESHSETS_LEVEL_2', 'FACES_BOUNDARY_MESHSETS_LEVEL_3',
             'FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC', 'PRIMAL_ID_1', 'PRIMAL_ID_2', 'L2_MESHSET', 'ID_reord_tag', 'CENT', 'AREA',
             'AREA2', 'PERM', 'K_EQ']

    names2 = ['WELLS_PRODUCER', 'WELLS_INJECTOR']
    nn = np.array(names)
    np.save('list_names_tags', nn)

def set_p_with_gravity(mb, mtu, volumes_d, press_tag, all_nodes, gama):

    values = (k_psi_to_pa)*mb.tag_get_data(press_tag, volumes_d, flat=True)
    coords = (k_pe_to_m)*mb.get_coords(all_nodes)
    coords = coords.reshape([len(all_nodes), 3])
    maxs = coords.max(axis=0)
    Lz = maxs[2]
    # z_elems_d = -1*np.array([mtu.get_average_position([v])[2] for v in volumes_d])
    z_elems_d = -1*(k_pe_to_m)*np.array([mtu.get_average_position([v])[2] for v in volumes_d])
    delta_z = z_elems_d + Lz
    pressao = gama*(delta_z) + values
    mb.tag_set_data(press_tag, volumes_d, pressao)

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
