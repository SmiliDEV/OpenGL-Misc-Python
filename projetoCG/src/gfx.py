import ctypes
import os
from typing import Optional, List

import numpy as np
from OpenGL.GL import *

from texture import Texture

class ShaderProgram:
    def __init__(self, vs_src: str, fs_src: str):
        self.prog = glCreateProgram()
        vs = self._compile(vs_src, GL_VERTEX_SHADER)
        fs = self._compile(fs_src, GL_FRAGMENT_SHADER)
        glAttachShader(self.prog, vs)
        glAttachShader(self.prog, fs)
        glLinkProgram(self.prog)
        glDeleteShader(vs)
        glDeleteShader(fs)
        if not glGetProgramiv(self.prog, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self.prog).decode())
        # Uniform locations aligned with shaders/basic.vert|frag
        self._u_model = glGetUniformLocation(self.prog, "uModel")
        self._u_view = glGetUniformLocation(self.prog, "uView")
        self._u_proj = glGetUniformLocation(self.prog, "uProj")
        self._u_lightdir = glGetUniformLocation(self.prog, "uLightDir")
        self._u_color = glGetUniformLocation(self.prog, "uColor")

    @classmethod
    def from_files(cls, vs_path: str, fs_path: str) -> "ShaderProgram":
        with open(vs_path, "r", encoding="utf-8") as f:
            vs_src = f.read()
        with open(fs_path, "r", encoding="utf-8") as f:
            fs_src = f.read()
        return cls(vs_src, fs_src)

    def _compile(self, src: str, kind) -> int:
        sh = glCreateShader(kind)
        glShaderSource(sh, src)
        glCompileShader(sh)
        if not glGetShaderiv(sh, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(sh).decode())
        return sh

    def use(self) -> None:
        glUseProgram(self.prog)

    def set_common(self, v_mat: np.ndarray, p_mat: np.ndarray, light_dir) -> None:
        # Shaders expect separate uView and uProj; also simple directional light
        glUniformMatrix4fv(self._u_view, 1, GL_TRUE, v_mat)
        glUniformMatrix4fv(self._u_proj, 1, GL_TRUE, p_mat)
        glUniform3fv(self._u_lightdir, 1, np.array(light_dir, dtype=np.float32))

    def set_per_object(self, model_mat: np.ndarray, albedo) -> None:
        glUniformMatrix4fv(self._u_model, 1, GL_TRUE, model_mat)
        glUniform3fv(self._u_color, 1, np.array(albedo, dtype=np.float32))

    def setBool(self, name: str, value: bool) -> None:
        glUniform1i(glGetUniformLocation(self.prog, name), int(value))

    def setInt(self, name: str, value: int) -> None:
        # ensure program is active before setting uniform to avoid GL_INVALID_OPERATION
        glUseProgram(self.prog)
        loc = glGetUniformLocation(self.prog, name)
        if loc != -1:
            glUniform1i(loc, value)

    def setFloat(self, name: str, value: float) -> None:
        glUniform1f(glGetUniformLocation(self.prog, name), value)

    def setVec2(self, name: str, value: np.ndarray) -> None:
        glUniform2fv(glGetUniformLocation(self.prog, name), 1, value)

    def setVec2(self, name: str, x: float, y: float) -> None:
        glUniform2f(glGetUniformLocation(self.prog, name), x, y)

    def setVec3(self, name: str, value: np.ndarray) -> None:
        glUniform3fv(glGetUniformLocation(self.prog, name), 1, value)

    def setVec3(self, name: str, x: float, y: float, z: float) -> None:
        glUniform3f(glGetUniformLocation(self.prog, name), x, y, z)

    def setVec4(self, name: str, value: np.ndarray) -> None:
        glUniform4fv(glGetUniformLocation(self.prog, name), 1, value)

    def setVec4(self, name: str, x: float, y: float, z: float, w: float) -> None:
        glUniform4f(glGetUniformLocation(self.prog, name), x, y, z, w)

    def setMat2(self, name: str, mat: np.ndarray) -> None:
        glUniformMatrix2fv(glGetUniformLocation(self.prog, name), 1, GL_FALSE, mat)

    def setMat3(self, name: str, mat: np.ndarray) -> None:
        glUniformMatrix3fv(glGetUniformLocation(self.prog, name), 1, GL_FALSE, mat)

    def setMat4(self, name: str, mat: np.ndarray) -> None:
        # ensure program is active and use transpose flag consistent with other helpers
        glUseProgram(self.prog)
        loc = glGetUniformLocation(self.prog, name)
        if loc != -1:
            # the codebase produces row-major numpy matrices; request transpose so GLSL receives column-major
            glUniformMatrix4fv(loc, 1, GL_TRUE, mat)
    
    

    def destroy(self) -> None:
        glDeleteProgram(self.prog)

def wrapperCreateShader(name: str) -> ShaderProgram:
    vs_path = os.path.join(os.path.dirname(__file__), 'shaders', f'{name}.vert')
    fs_path = os.path.join(os.path.dirname(__file__), 'shaders', f'{name}.frag')
    return ShaderProgram.from_files(vs_path, fs_path)

class Mesh:
    # Interleaved [pos3, normal3] with indices.
    def __init__(self, interleaved: np.ndarray, indices: np.ndarray):
        self.count = indices.size
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        stride = 6 * 4
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glBindVertexArray(0)

    def draw(self) -> None:
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def destroy(self) -> None:
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])

class MeshTextured:
    # Interleaved [pos3, normal3] with indices.
    def __init__(self, interleaved: np.ndarray, indices: np.ndarray, texture: Optional[Texture] = None):
        self.count = indices.size
        self.texture = texture
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        stride = 8 * 4
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        
        glBindVertexArray(0)

    def draw(self) -> None:
        if self.texture is not None:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture.id)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.count, GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def destroy(self) -> None:
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])