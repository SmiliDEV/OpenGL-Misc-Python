"""Small 3D math helpers implemented with NumPy.

Provides: perspective, lookAt and utilities to convert matrices to column-major bytes.
"""
import math
import numpy as np


def perspective(fovy_rad: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy_rad * 0.5)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M


def lookAt(eye, center, up) -> np.ndarray:
    e = np.array(eye, dtype=np.float32)
    c = np.array(center, dtype=np.float32)
    u = np.array(up, dtype=np.float32)

    f = c - e
    f = f / np.linalg.norm(f)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u2 = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, 0:3] = s
    M[1, 0:3] = u2
    M[2, 0:3] = -f
    M[0, 3] = -np.dot(s, e)
    M[1, 3] = -np.dot(u2, e)
    M[2, 3] = np.dot(f, e)
    return M


def mat_to_column_major_floats(M: np.ndarray) -> np.ndarray:
    return np.array(M, dtype=np.float32).T.reshape(-1)


def mat_to_bytes_col_major(M: np.ndarray) -> bytes:
    return mat_to_column_major_floats(M).tobytes()


# Transform helpers (translate, rotate, scale) compatible with previous helpers
def translate(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = x
    M[1, 3] = y
    M[2, 3] = z
    return M


def scale(sx: float, sy: float = None, sz: float = None) -> np.ndarray:
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = sx
    M[1, 1] = sy
    M[2, 2] = sz
    return M


def rotate(angle_rad: float, axis) -> np.ndarray:
    axis = np.array(axis, dtype=np.float32)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return np.eye(4, dtype=np.float32)
    x, y, z = (axis / n).tolist()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    R3 = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float32,
    )
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R3
    return M


def normal_matrix(M: np.ndarray) -> np.ndarray:
    N = M[:3, :3]
    return np.linalg.inv(N).T.astype(np.float32)
