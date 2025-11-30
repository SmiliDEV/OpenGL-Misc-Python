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
        self._u_model = glGetUniformLocation(self.prog, "uModel")
        self._u_lightdir = glGetUniformLocation(self.prog, "uLightDir")
        self._u_albedo = glGetUniformLocation(self.prog, "uAlbedo")
        self._u_ambient = glGetUniformLocation(self.prog, "uAmbient")
        self._u_lightdiffuse = glGetUniformLocation(self.prog, "uLightDiffuse")
        self._u_lightcount = glGetUniformLocation(self.prog, "uLightCount")
        self._u_lightpos = glGetUniformLocation(self.prog, "uLightPos[0]")
        self._u_lightcol = glGetUniformLocation(self.prog, "uLightCol[0]")
        self._u_lightint = glGetUniformLocation(self.prog, "uLightInt[0]")
        self._u_viewpos = glGetUniformLocation(self.prog, "uViewPos")
        self._u_specular = glGetUniformLocation(self.prog, "uSpecularColor")
        self._u_shininess = glGetUniformLocation(self.prog, "uShininess")
        self._u_diffuse_factor = glGetUniformLocation(self.prog, "uDiffuseFactor")

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

    def set_common(self, v_mat: np.ndarray, p_mat: np.ndarray, light_dir, ambient=None, light_color=None, lights=None, view_pos=None) -> None:
        # Shaders expect separate uView and uProj; also simple directional light
        if self._u_viewpos != -1 and view_pos is not None:
            glUniform3fv(self._u_viewpos, 1, np.array(view_pos, dtype=np.float32))
        if self._u_lightdir != -1 and light_dir is not None:
            glUniform3fv(self._u_lightdir, 1, np.array(light_dir, dtype=np.float32))
        if self._u_ambient != -1 and ambient is not None:
            glUniform3fv(self._u_ambient, 1, np.array(ambient, dtype=np.float32))
        if self._u_lightdiffuse != -1 and light_color is not None:
            glUniform3fv(self._u_lightdiffuse, 1, np.array(light_color, dtype=np.float32))
        # lights: list of dicts with keys 'pos','col','int'
        if lights is not None and len(lights) > 0:
            count = min(len(lights), 4)
            # build flat arrays
            pos_arr = np.zeros((count, 3), dtype=np.float32)
            col_arr = np.zeros((count, 3), dtype=np.float32)
            int_arr = np.zeros((count,), dtype=np.float32)
            for i in range(count):
                pos_arr[i, :] = np.array(lights[i].get('pos', [0.0,0.0,0.0]), dtype=np.float32)
                col_arr[i, :] = np.array(lights[i].get('col', [1.0,1.0,1.0]), dtype=np.float32) * float(lights[i].get('int', 1.0))
                int_arr[i] = float(lights[i].get('int', 1.0))
            if self._u_lightcount != -1:
                glUniform1i(self._u_lightcount, count)
            if self._u_lightpos != -1:
                glUniform3fv(self._u_lightpos, count, pos_arr.flatten())
            if self._u_lightcol != -1:
                glUniform3fv(self._u_lightcol, count, col_arr.flatten())
            if self._u_lightint != -1:
                glUniform1fv(self._u_lightint, count, int_arr)
        else:
            if self._u_lightcount != -1:
                glUniform1i(self._u_lightcount, 0)

    def set_per_object(self, model_mat: np.ndarray, albedo, specular=None, shininess=None, diffuse_factor=None) -> None:
        """Set per-object uniforms (model, albedo, optional specular, shininess, diffuse_factor).

        Backwards-compatible: callers may pass only (model, albedo).
        """
        glUniformMatrix4fv(self._u_model, 1, GL_TRUE, model_mat)
        if self._u_albedo != -1:
            glUniform3fv(self._u_albedo, 1, np.array(albedo, dtype=np.float32))
        # specular
        if specular is not None and self._u_specular != -1:
            glUniform3fv(self._u_specular, 1, np.array(specular, dtype=np.float32))
        # shininess
        if shininess is not None and self._u_shininess != -1:
            glUniform1f(self._u_shininess, float(shininess))
        # diffuse factor
        if diffuse_factor is not None and self._u_diffuse_factor != -1:
            glUniform1f(self._u_diffuse_factor, float(diffuse_factor))

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