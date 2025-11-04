def gen_uv_triangle_flat(size=1.0):
    h = size * math.sqrt(3) / 2.0
    vertices = [ [ -size/2.0, 0.0, -h/3.0], [ size/2.0, 0.0, -h/3.0], [0.0, 0.0, 2.0*h/3.0] ]
    faces = [ [0,1,2] ]
    normals = [ [0.0, 1.0, 0.0] ]
    tri_pos = []
    tri_nrm = []
    for f in range(1):
        a, b, c = faces[f]
        n = normals[f]
        tri_pos.extend([vertices[a], vertices[b], vertices[c]])
        tri_nrm.extend([n, n, n])
    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1,3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1,3)
    inter = np.empty((tri_pos.shape[0], 6), dtype=np.float32)
    inter[:,0:3] = tri_pos; inter[:,3:6] = tri_nrm
    inter = inter.reshape(-1)   
    indices = np.arange(tri_pos.shape[0], dtype=np.uint32)
    return inter, indices
def gen_uv_cube_flat(size=1.0):
    n = size / 2.0
    vertices = [ [ n, n, n], [ n, n,-n], [ n,-n,-n], [ n,-n, n]
               ,[-n, n, n], [-n, n,-n], [-n,-n,-n], [-n,-n, n] ]
    faces = [ [0,1,2,3], [4,7,6,5], [0,3,7,4]
            ,[1,5,6,2], [0,4,5,1], [3,2,6,7] ]
    normals = [ [ 1, 0, 0], [-1, 0, 0], [ 0, 0, 1]
              , [ 0, 0,-1], [ 0, 1, 0], [ 0,-1, 0] ]
    tri_pos = []
    tri_nrm = []
    for f in range(6):
        a, b, c, d = faces[f]
        n = normals[f]
        tri_pos.extend([vertices[a], vertices[b], vertices[c]])
        tri_nrm.extend([n, n, n])
        tri_pos.extend([vertices[a], vertices[c], vertices[d]])
        tri_nrm.extend([n, n, n])
    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1,3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1,3)
    inter = np.empty((tri_pos.shape[0], 6), dtype=np.float32)
    inter[:,0:3] = tri_pos; inter[:,3:6] = tri_nrm
    inter = inter.reshape(-1)   
    indices = np.arange(tri_pos.shape[0], dtype=np.uint32)
    return inter, indices