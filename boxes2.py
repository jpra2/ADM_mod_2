all_centroids = M1.all_centroids

y0 = 18.0
y1 = 1.0
y2 = 17.0

box_volumes_d = np.array([np.array([0.0, 0.0, 0.0]), np.array([y1, y0, y0])])
box_volumes_n = np.array([np.array([y2, 0.0, 0.0]), np.array([y0, y0, y0])])

# volumes com pressao prescrita
inds0 = np.where(all_centroids[:,0] > box_volumes_d[0,0])[0]
inds1 = np.where(all_centroids[:,1] > box_volumes_d[0,1])[0]
inds2 = np.where(all_centroids[:,2] > box_volumes_d[0,2])[0]
c1 = set(inds0) & set(inds1) & set(inds2)
inds0 = np.where(all_centroids[:,0] < box_volumes_d[1,0])[0]
inds1 = np.where(all_centroids[:,1] < box_volumes_d[1,1])[0]
inds2 = np.where(all_centroids[:,2] < box_volumes_d[1,2])[0]
c2 = set(inds0) & set(inds1) & set(inds2)
inds_vols_d = list(c1 & c2)
volumes_d = rng.Range(np.array(M1.all_volumes)[inds_vols_d])

# volumes com vazao prescrita
inds0 = np.where(all_centroids[:,0] > box_volumes_n[0,0])[0]
inds1 = np.where(all_centroids[:,1] > box_volumes_n[0,1])[0]
inds2 = np.where(all_centroids[:,2] > box_volumes_n[0,2])[0]
c1 = set(inds0) & set(inds1) & set(inds2)
inds0 = np.where(all_centroids[:,0] < box_volumes_n[1,0])[0]
inds1 = np.where(all_centroids[:,1] < box_volumes_n[1,1])[0]
inds2 = np.where(all_centroids[:,2] < box_volumes_n[1,2])[0]
c2 = set(inds0) & set(inds1) & set(inds2)
inds_vols_n = list(c1 & c2)
volumes_n = rng.Range(np.array(M1.all_volumes)[inds_vols_n])

inds_pocos = inds_vols_d + inds_vols_n
centroids_pocos = all_centroids[inds_pocos]

# Ci = n: Ci -> Razão de engrossamento ni nível i (em relação ao nível i-1),
# n -> número de blocos em cada uma das 3 direções (mesmo número em todas)
l1=3
l2=9
# Posição aproximada de cada completação
Cent_weels=[[0.5, 0.5, 0.5],[17.5, 17.5, 17.5]]
Cent_weels = np.array([np.array(c) for c in Cent_weels])
Cent_weels = np.append(Cent_weels, centroids_pocos, axis=0)
Cent_weels = np.unique(Cent_weels, axis=0)
