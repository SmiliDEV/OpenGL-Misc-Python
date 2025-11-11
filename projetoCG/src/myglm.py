import numpy as np
import math

# Transformações matemáticas! Teêm que ser todas explicitadas
def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f/aspect; M[1,1] = f
    M[2,2] = (zfar + znear) / (znear - zfar)
    M[2,3] = (2.0 * zfar * znear) / (znear - zfar)
    M[3,2] = -1.0
    return M

#Nem existe uma função de LookAt. Tem que ser construída de raíz
def lookAt(eye, center, up):
    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    f = center - eye; f = f / np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0,0:3] = s; M[1,0:3] = u; M[2,0:3] = -f
    T = np.eye(4, dtype=np.float32)
    T[0,3] = -eye[0]; T[1,3] = -eye[1]; T[2,3] = -eye[2]
    return M @ T

#funções de transformaçãp
def translate(x, y, z):
    M = np.eye(4, dtype=np.float32) 
    M[0,3]=x; M[1,3]=y; M[2,3]=z 
    return M

def scale(sx, sy=None, sz=None):
    if sy is None: sy = sx
    if sz is None: sz = sx
    M = np.eye(4, dtype=np.float32)
    M[0,0]=sx; M[1,1]=sy; M[2,2]=sz; 
    return M

#para a rotação usamos a forma de Rodrigues e temos um "rodador" compatível com
# as versões anteriores do OpenGL

def rotate(angle_rad, axis):
    axis = np.array(axis, dtype=np.float32)
    n = np.linalg.norm(axis)
    if n == 0: return np.eye(4, dtype=np.float32)
    x, y, z = axis / n
    c = math.cos(angle_rad); s = math.sin(angle_rad); C = 1.0 - c
    R3 = np.array([
        [x*x*C + c,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, z*z*C + c  ],], dtype=np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3,:3] = R3
    return M

#Cria a matriz normal como inversa da transposta da matriz do modelo
def normal_matrix(M):
    N = M[:3,:3]
    return np.linalg.inv(N).T.astype(np.float32)