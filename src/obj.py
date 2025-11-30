import os
from typing import List, Tuple, Optional, Dict

import numpy as np

from gfx import Mesh, MeshTextured


def _fix_index(i: int, arr_len: int) -> int:
    return i - 1 if i > 0 else arr_len + i


def load_obj(path: str, scale: float = 1.0, normalize: bool = False, target_max: float = 1.0, center: bool = True):
    """Load a Wavefront OBJ and return a `Mesh` or `MeshTextured`.

    Produces interleaved arrays compatible with `gfx.Mesh` (pos3, norm3)
    or `gfx.MeshTextured` (pos3, norm3, uv2) depending on whether the
    OBJ contains texture coordinates.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    positions: List[Tuple[float, float, float]] = []
    texcoords: List[Tuple[float, float]] = []
    normals: List[Tuple[float, float, float]] = []
    faces: List[List[Tuple[Optional[int], Optional[int], Optional[int]]]] = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or line[0] == '#':
                continue
            parts = line.strip().split()
            if not parts:
                continue
            p0 = parts[0]
            if p0 == 'v' and len(parts) >= 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                positions.append((x, y, z))
            elif p0 == 'vt' and len(parts) >= 3:
                u, v = float(parts[1]), float(parts[2])
                texcoords.append((u, v))
            elif p0 == 'vn' and len(parts) >= 4:
                nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                normals.append((nx, ny, nz))
            elif p0 == 'f' and len(parts) >= 4:
                face = []
                for tok in parts[1:]:
                    comps = tok.split('/')
                    vi = int(comps[0]) if comps[0] != '' else None
                    vti = int(comps[1]) if len(comps) >= 2 and comps[1] != '' else None
                    vni = int(comps[2]) if len(comps) >= 3 and comps[2] != '' else None
                    face.append((vi, vti, vni))
                faces.append(face)

    # Optionally normalize positions (center and uniformly scale to target_max)
    if normalize and len(positions) > 0:
        pos_arr = np.asarray(positions, dtype=np.float32)
        mn = pos_arr.min(axis=0)
        mx = pos_arr.max(axis=0)
        if center:
            ctr = (mn + mx) * 0.5
            pos_arr -= ctr
        ext = mx - mn
        max_dim = float(max(ext[0], ext[1], ext[2])) if ext is not None else 0.0
        if max_dim > 0.0:
            factor = float(target_max) / max_dim
            pos_arr *= factor
        # apply optional additional uniform scale
        if scale != 1.0:
            pos_arr *= float(scale)
        # rewrite positions back to list
        positions = [tuple(p) for p in pos_arr.tolist()]

    # Build unique vertices
    unique: Dict[Tuple[Optional[int], Optional[int], Optional[int]], int] = {}
    vertex_list: List[float] = []
    index_list: List[int] = []

    # If normals are missing we will compute per-vertex normals by averaging face normals
    compute_normals = len(normals) == 0
    accum_normals: List[np.ndarray] = []

    for face in faces:
        if len(face) < 3:
            continue
        tri_indices = []
        for comp in face:
            key = comp
            if key in unique:
                idx = unique[key]
            else:
                vi, vti, vni = key
                if vi is None:
                    raise RuntimeError("OBJ face with missing vertex index")
                vi_fixed = _fix_index(vi, len(positions))
                pos = positions[vi_fixed]
                if vni is not None:
                    vn_fixed = _fix_index(vni, len(normals))
                    n = normals[vn_fixed]
                else:
                    n = (0.0, 0.0, 0.0)
                if vti is not None:
                    vt_fixed = _fix_index(vti, len(texcoords))
                    uv = texcoords[vt_fixed]
                else:
                    uv = (0.0, 0.0)

                # store vertex. We'll decide format later; keep pos+norm+uv
                vertex_list.extend([float(pos[0]), float(pos[1]), float(pos[2])])
                vertex_list.extend([float(n[0]), float(n[1]), float(n[2])])
                vertex_list.extend([float(uv[0]), float(uv[1])])
                idx = len(unique)
                unique[key] = idx
            tri_indices.append(idx)

        # triangulate fan
        for i in range(1, len(tri_indices) - 1):
            index_list.append(tri_indices[0])
            index_list.append(tri_indices[i])
            index_list.append(tri_indices[i + 1])

    if len(vertex_list) == 0 or len(index_list) == 0:
        raise RuntimeError("OBJ contained no usable geometry")

    v_arr = np.asarray(vertex_list, dtype=np.float32)
    i_arr = np.asarray(index_list, dtype=np.uint32)

    # Determine whether texcoords were present by checking stride: we always stored 8 floats per vertex
    # For gfx.Mesh we need pos(3)+norm(3) stride=6, for MeshTextured pos+norm+uv stride=8
    has_uv = True

    if has_uv:
        # v_arr is Nx8 floats interleaved
        return MeshTextured(v_arr.reshape(-1, 8).astype(np.float32).flatten(), i_arr, texture=None)
    else:
        # fallback (shouldn't reach with current code path)
        inter6 = v_arr.reshape(-1, 8)[:, :6].flatten().astype(np.float32)
        return Mesh(inter6, i_arr)


__all__ = ['load_obj']
