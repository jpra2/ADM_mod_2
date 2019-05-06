import numpy as np
from pymoab import core, types, rng, topo_util
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_parent_dir = os.path.dirname(parent_dir)
input_dir = os.path.join(parent_parent_dir, 'input')
flying_dir = os.path.join(parent_parent_dir, 'flying')


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

def cent(mb, mtu, all_volumes):
    centroids = np.array([mtu.get_average_position([v]) for v in all_volumes])
    cent_tag = mb.tag_get_handle('CENT', 3, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mb.tag_set_data(cent_tag, all_volumes, centroids)

def create_names_tags():
    os.chdir(flying_dir)
    names = ['d1', 'd2', 'l1_ID', 'l2_ID', 'l3_ID', 'P', 'Q', 'FACES_BOUNDARY', 'FACES_BOUNDARY_MESHSETS_LEVEL_2', 'FACES_BOUNDARY_MESHSETS_LEVEL_3',
             'FINE_TO_PRIMAL1_CLASSIC', 'FINE_TO_PRIMAL2_CLASSIC', 'PRIMAL_ID_1', 'PRIMAL_ID_2', 'L2_MESHSET', 'ID_reord_tag']

    names2 = ['CENT', 'WELLS_PRODUCER', 'WELLS_INJECTOR']
    nn = np.array(names)
    np.save('list_names_tags', nn)
