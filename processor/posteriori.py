import numpy as np
import pdb
from pymoab import types

def redefinir_permeabilidades(mb, all_volumes, all_centroids, perm_tag):

    rg1 = np.array([np.array([6.0, 0.0, 0.0]), np.array([8.0, 19.0, 27.0])])
    rg2 = np.array([np.array([19.0, 8.0, 0.0]), np.array([21.0, 27.0, 27.0])])

    k01 = 1.0
    k02 = 0.01

    k2 = np.array([k02, 0.0, 0.0,
          0.0, k02, 0.0,
          0.0, 0.0, k02])

    inds_vols = []

    inds0 = np.where(all_centroids[:,0] > rg1[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > rg1[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > rg1[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < rg1[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < rg1[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < rg1[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols += list(c1 & c2)

    inds0 = np.where(all_centroids[:,0] > rg2[0,0])[0]
    inds1 = np.where(all_centroids[:,1] > rg2[0,1])[0]
    inds2 = np.where(all_centroids[:,2] > rg2[0,2])[0]
    c1 = set(inds0) & set(inds1) & set(inds2)
    inds0 = np.where(all_centroids[:,0] < rg2[1,0])[0]
    inds1 = np.where(all_centroids[:,1] < rg2[1,1])[0]
    inds2 = np.where(all_centroids[:,2] < rg2[1,2])[0]
    c2 = set(inds0) & set(inds1) & set(inds2)
    inds_vols += list(c1 & c2)

    vols2 = np.array(all_volumes)[inds_vols].astype(np.uint64)

    verifk_tag = mb.tag_get_handle('verifk', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_SPARSE, True)
    mb.tag_set_data(verifk_tag, all_volumes, np.repeat(k01, len(all_volumes)))
    mb.tag_set_data(verifk_tag, vols2, np.repeat(k02, len(vols2)))

    for v in vols2:
        mb.tag_set_data(perm_tag, v, k2)
