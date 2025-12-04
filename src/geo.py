import numpy as np
import math

#Construção da geometria
def gen_uv_sphere_flat(radius=1.0, stacks=24, slices=48):
    #construção da Esfera com normais
    # tem que ser construída explicitamente para cada corte e fatia 
    # grelha de posições <- Nao podemos usar o glutSolidSphere
    P = np.zeros(((stacks+1)*(slices+1), 3), dtype=np.float32)
    for i in range(stacks+1):
        v = i / stacks
        theta = v * math.pi
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(slices+1):
            u = j / slices
            phi = u * 2.0 * math.pi
            sp, cp = math.sin(phi), math.cos(phi)
            x = cp * st; y = ct; z = sp * st
            P[i*(slices+1)+j] = [radius*x, radius*y, radius*z]

    #
    tri_pos = []; tri_nrm = []
    for i in range(stacks):
        for j in range(slices):
            #Cada elemento dá um QUAD, que tem que ser dividido em triangulos
            a = i*(slices+1)+j; b = a+1; c = a+(slices+1); d = c+1
            # tri1: a,b,c
            p0, p1, p2 = P[a], P[b], P[c]
            #normal <- veja-se o produto externo
            n = np.cross(p1-p0, p2-p0); ln = np.linalg.norm(n); n = n/ln if ln>0 else np.array([0,1,0], dtype=np.float32)
            tri_pos.extend([p0, p1, p2]); tri_nrm.extend([n, n, n])
            # tri2: b,d,c
            p0, p1, p2 = P[b], P[d], P[c]
            #normal <- veja-se o produto externo
            n = np.cross(p1-p0, p2-p0); 
            ln = np.linalg.norm(n); 
            n = n/ln if ln>0 else np.array([0,1,0], dtype=np.float32)
            tri_pos.extend([p0, p1, p2]); tri_nrm.extend([n, n, n])

    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1,3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1,3)
    inter = np.empty((tri_pos.shape[0], 6), dtype=np.float32)
    inter[:,0:3] = tri_pos; inter[:,3:6] = tri_nrm
    inter = inter.reshape(-1)
    indices = np.arange(tri_pos.shape[0], dtype=np.uint32)
    return inter, indices

def gen_uv_prism_flat(size=1.0, height=1.0):
    # Prisma triangular RETÂNGULO (triângulo no plano XZ com ângulo reto em (0,0))
    # - Triângulo: A=(0,0), B=(s,0), C=(0,s) em XZ
    # - Extrusão ao longo de Y com altura 'height'
    # - Winding CCW visto do exterior; normais planas por face (flat)
    s = float(size)
    h = float(height)

    y_top = +0.5 * h
    y_bot = -0.5 * h

    # Vértices das tampas
    A_top = np.array([0.0, y_top, 0.0], dtype=np.float32)
    B_top = np.array([   s, y_top, 0.0], dtype=np.float32)
    C_top = np.array([0.0, y_top,    s], dtype=np.float32)

    A_bot = np.array([0.0, y_bot, 0.0], dtype=np.float32)
    B_bot = np.array([   s, y_bot, 0.0], dtype=np.float32)
    C_bot = np.array([0.0, y_bot,    s], dtype=np.float32)

    tri_pos = []
    tri_nrm = []

    # Tampa superior (normal +Y): A_top, B_top, C_top (CCW visto de +Y)
    n_top = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tri_pos.extend([A_top, B_top, C_top]); tri_nrm.extend([n_top, n_top, n_top])

    # Tampa inferior (normal -Y): A_bot, C_bot, B_bot (CCW visto de -Y)
    n_bot = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    tri_pos.extend([A_bot, C_bot, B_bot]); tri_nrm.extend([n_bot, n_bot, n_bot])

    # Lados (retângulos → 2 triângulos): AB, BC, CA
    sides = [
        (A_top, B_top, B_bot, A_bot),  # lado AB
        (B_top, C_top, C_bot, B_bot),  # lado BC
        (C_top, A_top, A_bot, C_bot),  # lado CA
    ]
    for t0, t1, b1, b0 in sides:
        # tri1: t0, t1, b1
        n1 = np.cross(t1 - t0, b1 - t0)
        ln = np.linalg.norm(n1); n1 = n1/ln if ln > 0 else np.array([0,1,0], dtype=np.float32)
        tri_pos.extend([t0, t1, b1]); tri_nrm.extend([n1, n1, n1])
        # tri2: t0, b1, b0
        n2 = np.cross(b1 - t0, b0 - t0)
        ln = np.linalg.norm(n2); n2 = n2/ln if ln > 0 else n1
        tri_pos.extend([t0, b1, b0]); tri_nrm.extend([n2, n2, n2])

    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1,3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1,3)
    inter = np.empty((tri_pos.shape[0], 6), dtype=np.float32)
    inter[:,0:3] = tri_pos; inter[:,3:6] = tri_nrm
    inter = inter.reshape(-1)
    indices = np.arange(tri_pos.shape[0], dtype=np.uint32)
    return inter, indices
    
def gen_uv_cylinder_flat(radius=0.5, half_width=0.25, slices=32):
    """Cylinder aligned with Y axis, flat-shaded triangles.
    - radius: cylinder radius in XZ plane
    - half_width: half of the cylinder length along Y (total width = 2*half_width)
    - slices: number of radial divisions (>=3)

    Returns interleaved array [pos(3), normal(3)] and triangle indices.
    """
    slices = max(3, int(slices))
    r = float(radius)
    hw = float(half_width)

    tri_pos = []
    tri_nrm = []

    # side quads split into two triangles each
    for j in range(slices):
        u0 = j / slices
        u1 = (j + 1) / slices
        phi0 = 2.0 * math.pi * u0
        phi1 = 2.0 * math.pi * u1
        x0, z0 = math.cos(phi0) * r, math.sin(phi0) * r
        x1, z1 = math.cos(phi1) * r, math.sin(phi1) * r

        p0t = np.array([x0, +hw, z0], dtype=np.float32)
        p0b = np.array([x0, -hw, z0], dtype=np.float32)
        p1t = np.array([x1, +hw, z1], dtype=np.float32)
        p1b = np.array([x1, -hw, z1], dtype=np.float32)

        # Tri 1 (side): p0t, p0b, p1t
        n1 = np.cross(p0b - p0t, p1t - p0t)
        ln = np.linalg.norm(n1); n1 = n1/ln if ln > 0 else np.array([0,1,0], dtype=np.float32)
        tri_pos.extend([p0t, p0b, p1t]); tri_nrm.extend([n1, n1, n1])

        # Tri 2 (side): p1t, p0b, p1b
        n2 = np.cross(p0b - p1t, p1b - p1t)
        ln = np.linalg.norm(n2); n2 = n2/ln if ln > 0 else n1
        tri_pos.extend([p1t, p0b, p1b]); tri_nrm.extend([n2, n2, n2])

    # top cap (normal +Y): fan from center_top
    c_top = np.array([0.0, +hw, 0.0], dtype=np.float32)
    n_top = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for j in range(slices):
        phi0 = 2.0 * math.pi * (j / slices)
        phi1 = 2.0 * math.pi * ((j + 1) / slices)
        p0 = np.array([math.cos(phi0) * r, +hw, math.sin(phi0) * r], dtype=np.float32)
        p1 = np.array([math.cos(phi1) * r, +hw, math.sin(phi1) * r], dtype=np.float32)
        # CCW when viewed from +Y: center -> p0 -> p1
        tri_pos.extend([c_top, p0, p1]); tri_nrm.extend([n_top, n_top, n_top])

    # bottom cap (normal -Y): fan from center_bottom
    c_bot = np.array([0.0, -hw, 0.0], dtype=np.float32)
    n_bot = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    for j in range(slices):
        phi0 = 2.0 * math.pi * (j / slices)
        phi1 = 2.0 * math.pi * ((j + 1) / slices)
        p0 = np.array([math.cos(phi0) * r, -hw, math.sin(phi0) * r], dtype=np.float32)
        p1 = np.array([math.cos(phi1) * r, -hw, math.sin(phi1) * r], dtype=np.float32)
        # CCW when viewed from -Y: center -> p1 -> p0
        tri_pos.extend([c_bot, p1, p0]); tri_nrm.extend([n_bot, n_bot, n_bot])

    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1, 3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1, 3)
    inter = np.empty((tri_pos.shape[0], 6), dtype=np.float32)
    inter[:, 0:3] = tri_pos
    inter[:, 3:6] = tri_nrm
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
        # Invertido para CCW (a,c,b) e (a,d,c)
        tri_pos.extend([vertices[a], vertices[c], vertices[b]])
        tri_nrm.extend([n, n, n])
        tri_pos.extend([vertices[a], vertices[d], vertices[c]])
        tri_nrm.extend([n, n, n])
    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1,3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1,3)
    inter = np.empty((tri_pos.shape[0], 6), dtype=np.float32)
    inter[:,0:3] = tri_pos; inter[:,3:6] = tri_nrm
    inter = inter.reshape(-1)   
    indices = np.arange(tri_pos.shape[0], dtype=np.uint32)
    return inter, indices


def gen_skybox_mesh(size: float = 1.0):
    """
    Convenience helper that returns interleaved array and indices for a unit cube
    suitable for a skybox. Uses `gen_uv_cube_flat` under the hood so the
    returned interleaved layout is compatible with the existing `Mesh` helper.
    """
    return gen_uv_cube_flat(size=size)

def gen_uv_plane_flat(size=1.0, divisions=10):
    step = size / divisions
    half = size / 2.0
    tri_pos = []; tri_nrm = []; tri_uv = []
    for i in range(divisions):
        for j in range(divisions):
            x0 = -half + j * step; x1 = x0 + step
            z0 = -half + i * step; z1 = z0 + step
            y = 0.0
            # Cantos do quad
            p0 = [x0, y, z0]  # bottom-left
            p1 = [x1, y, z0]  # bottom-right
            p2 = [x1, y, z1]  # top-right
            p3 = [x0, y, z1]  # top-left
            n = [0.0, 1.0, 0.0]
            # UV coordinates (u along X, v along Z)
            u0 = j / divisions; u1 = (j + 1) / divisions
            v0 = i / divisions; v1 = (i + 1) / divisions
            uv0 = [u0, v0]
            uv1 = [u1, v0]
            uv2 = [u1, v1]
            uv3 = [u0, v1]
            # Winding CCW visto de cima (normal +Y)
            # Triângulo 1: p0 -> p2 -> p1
            tri_pos.extend([p0, p2, p1]); tri_nrm.extend([n, n, n]); tri_uv.extend([uv0, uv2, uv1])
            # Triângulo 2: p0 -> p3 -> p2
            tri_pos.extend([p0, p3, p2]); tri_nrm.extend([n, n, n]); tri_uv.extend([uv0, uv3, uv2])
    tri_pos = np.array(tri_pos, dtype=np.float32).reshape(-1,3)
    tri_nrm = np.array(tri_nrm, dtype=np.float32).reshape(-1,3)
    tri_uv = np.array(tri_uv, dtype=np.float32).reshape(-1,2)
    # flip V coordinate to match loader convention (textures were flipped on load)
    tri_uv[:,1] = 1.0 - tri_uv[:,1]
    inter = np.empty((tri_pos.shape[0], 8), dtype=np.float32)
    inter[:,0:3] = tri_pos; inter[:,3:6] = tri_nrm; inter[:,6:8] = tri_uv
    inter = inter.reshape(-1)
    indices = np.arange(tri_pos.shape[0], dtype=np.uint32)
    return inter, indices
    
def gen_uv_car_body(size=1.0):
    n = size
    altura_topo = n * 1.5  # altura máxima do topo (triângulos)
    roof_half = n * 0.5    # meia dimensão do quadrado do teto


    base = [
        [ n,-n, n], [ n,-n,-n], [-n,-n,-n], [-n,-n, n],  # 0-3: base inferior
        [ n, n, n], [ n, n,-n], [-n, n,-n], [-n, n, n]   # 4-7: topo bloco
    ]


    topo = [
        # (Removido capô frontal) Mantemos só a traseira em x=-n
        # Triângulo traseira em x=-n (normal -X)
        [-n, n, -roof_half], [-n, n,  roof_half], [-n, altura_topo, 0.0],  # 11-13
        # Quadrado central (teto plano), quadrado de lado 2*roof_half
        [-roof_half, altura_topo, -roof_half], [ roof_half, altura_topo, -roof_half],
        [ roof_half, altura_topo,  roof_half], [-roof_half, altura_topo,  roof_half]  # 14-17
    ]


    vertices = np.array(base + topo, dtype=np.float32)


    indices = [
        # Base do bloco (y = -n) normal -Y
        0,3,2, 0,2,1,
        # Topo do bloco (y = +n) normal +Y
        4,5,6, 6,7,4,
        # Laterais do bloco
        # x = +n (direito) normal +X
        0,1,5, 0,5,4,
        # x = -n (esquerdo) normal -X
        3,7,6, 3,6,2,
        # z = -n (fundo) normal -Z
        1,2,6, 1,6,5,
        # z = +n (frente) normal +Z
        0,4,7, 7,3,0,
    # Topo: apenas triângulo traseira (ao longo de X negativo)
    # traseira (x=-n) normal -X  (novos índices 8,9,10)
    8,9,10,
    # face traseira em ambas as faces (duplicar tri com ordem inversa)
    10,9,8,
        # Teto plano (quadrado central) normal +Y (ordem CCW vista de +Y)
        # novos índices 11,12,13,14
        11,12,13, 11,13,14
    ]


    normals = []
    for i in range(0, len(indices), 3):
        a = vertices[indices[i]]
        b = vertices[indices[i+1]]
        c = vertices[indices[i+2]]
        n_vec = np.cross(b - a, c - a)
        n_len = np.linalg.norm(n_vec)
        if n_len > 0:
            n_vec /= n_len
        else:
            n_vec = np.array([0,1,0], dtype=np.float32)
        normals.extend([n_vec, n_vec, n_vec])

    normals = np.array(normals, dtype=np.float32)

    inter = np.empty((len(indices), 6), dtype=np.float32)
    for i, idx in enumerate(indices):
        inter[i,0:3] = vertices[idx]
        inter[i,3:6] = normals[i]
    inter = inter.reshape(-1)

    indices = np.array(range(len(indices)), dtype=np.uint32)

    return inter, indices