from typing import Optional
from OpenGL.GL import *
import ctypes
import numpy as np
from PIL import Image

class Shader:
    """
    Minimal Shader helper:
    - Shader.from_files(v_path, f_path) -> retorna instancia
    - shader.use(), shader.prog (program id)
    - shader.setInt/Float/Bool/Vec3/Mat4(name, value)
    """
    def __init__(self, vertex_src: str, fragment_src: str):
        self.prog = glCreateProgram()
        vert = self._compile_shader(vertex_src, GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_src, GL_FRAGMENT_SHADER)
        glAttachShader(self.prog, vert)
        glAttachShader(self.prog, frag)
        glLinkProgram(self.prog)

        # link check
        linked = glGetProgramiv(self.prog, GL_LINK_STATUS)
        if not linked:
            info = glGetProgramInfoLog(self.prog)
            glDeleteProgram(self.prog)
            glDeleteShader(vert)
            glDeleteShader(frag)
            raise RuntimeError(f"Shader link error:\n{info.decode() if isinstance(info, (bytes, bytearray)) else info}")

        glDeleteShader(vert)
        glDeleteShader(frag)

    @classmethod
    def from_files(cls, vertex_path: str, fragment_path: str):
        with open(vertex_path, 'r', encoding='utf-8') as f:
            vsrc = f.read()
        with open(fragment_path, 'r', encoding='utf-8') as f:
            fsrc = f.read()
        return cls(vsrc, fsrc)

    def _compile_shader(self, src: str, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, src)
        glCompileShader(shader)
        ok = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not ok:
            info = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            raise RuntimeError(f"Shader compile error ({shader_type}):\n{info.decode() if isinstance(info, (bytes, bytearray)) else info}")
        return shader

    def use(self):
        glUseProgram(self.prog)

    def stop(self):
        glUseProgram(0)

    def delete(self):
        if self.prog:
            glDeleteProgram(self.prog)
            self.prog = 0

    def setInt(self, name: str, value: int):
        loc = glGetUniformLocation(self.prog, name)
        if loc != -1:
            glUniform1i(loc, int(value))

    def setFloat(self, name: str, value: float):
        loc = glGetUniformLocation(self.prog, name)
        if loc != -1:
            glUniform1f(loc, float(value))

    def setBool(self, name: str, value: bool):
        self.setInt(name, 1 if value else 0)

    def setVec3(self, name: str, vec):
        loc = glGetUniformLocation(self.prog, name)
        if loc != -1:
            v = np.asarray(vec, dtype=np.float32)
            if v.size == 3:
                glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    def setMat4(self, name: str, mat):
        loc = glGetUniformLocation(self.prog, name)
        if loc == -1:
            return
        m = np.asarray(mat, dtype=np.float32)
        glUniformMatrix4fv(loc, 1, GL_FALSE, m)





class VertexBuffer:
    """Simple VBO wrapper."""
    def __init__(self, data: np.ndarray = None, usage=GL_STATIC_DRAW):
        self.id = glGenBuffers(1)
        self.usage = usage
        self.count = 0
        if data is not None:
            self.set_data(data)

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.id)

    def unbind(self):
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def set_data(self, data: np.ndarray):
        self.bind()
        self.count = int(data.size)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, self.usage)
        self.unbind()

    def delete(self):
        if self.id:
            glDeleteBuffers(1, [self.id])
            self.id = 0





class IndexBuffer:
    """EBO / IBO wrapper (uses uint32 indices)."""
    def __init__(self, indices: np.ndarray = None, usage=GL_STATIC_DRAW):
        self.id = glGenBuffers(1)
        self.usage = usage
        self.count = 0
        if indices is not None:
            self.set_data(indices)

    def bind(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.id)

    def unbind(self):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def set_data(self, indices: np.ndarray):
        # ensure uint32
        arr = np.asarray(indices, dtype=np.uint32)
        self.bind()
        self.count = arr.size
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, arr.nbytes, arr, self.usage)
        self.unbind()

    def draw(self, mode=GL_TRIANGLES, count: Optional[int] = None):
        cnt = self.count if count is None else int(count)
        glDrawElements(mode, cnt, GL_UNSIGNED_INT, ctypes.c_void_p(0))

    def delete(self):
        if self.id:
            glDeleteBuffers(1, [self.id])
            self.id = 0




class UniformBuffer:
    """UBO wrapper."""
    def __init__(self, data: np.ndarray = None, binding_point: Optional[int] = None, usage=GL_STATIC_DRAW):
        self.id = glGenBuffers(1)
        if isinstance(self.id, (list, tuple)):
            self.id = int(self.id[0])
        self.usage = usage
        self.size = 0
        self.binding_point: Optional[int] = binding_point

        if data is not None:
            self.set_data(data)

    def bind(self, binding_point: Optional[int] = None, offset: int = 0, size: Optional[int] = None):
        """
        If binding_point is None -> bind the buffer to GL_UNIFORM_BUFFER target (for uploads).
        If binding_point is provided -> bind the buffer range to that uniform binding point.
        """
        if binding_point is None:
            glBindBuffer(GL_UNIFORM_BUFFER, self.id)
        else:
            sz = self.size if size is None else int(size)
            glBindBufferRange(GL_UNIFORM_BUFFER, int(binding_point), self.id, int(offset), int(sz))
            self.binding_point = int(binding_point)

    def unbind(self, binding_point: Optional[int] = None):
        if binding_point is None:
            glBindBuffer(GL_UNIFORM_BUFFER, 0)
        else:
            # release binding point
            glBindBufferBase(GL_UNIFORM_BUFFER, int(binding_point), 0)
            if self.binding_point == binding_point:
                self.binding_point = None

    def set_data(self, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        self.size = arr.nbytes
        # bind to target for data upload
        glBindBuffer(GL_UNIFORM_BUFFER, self.id)
        glBufferData(GL_UNIFORM_BUFFER, self.size, arr, self.usage)
        # unbind target
        glBindBuffer(GL_UNIFORM_BUFFER, 0)
        # if previously bound to a binding point, re-bind the range with full size
        if self.binding_point is not None:
            glBindBufferRange(GL_UNIFORM_BUFFER, self.binding_point, self.id, 0, self.size)

    def update_subdata(self, offset: int, data: np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        glBindBuffer(GL_UNIFORM_BUFFER, self.id)
        glBufferSubData(GL_UNIFORM_BUFFER, int(offset), arr.nbytes, arr)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    def bind_shader_block(self, program, block_name: str):
        block_index = glGetUniformBlockIndex(program, block_name)

        if block_index == GL_INVALID_INDEX:
            raise RuntimeError(f"Uniform block '{block_name}' not found in program {program}.")
        
        if self.binding_point is None:
            raise RuntimeError("UniformBuffer must have a binding_point to bind to shader block.")
        
        glUniformBlockBinding(program, block_index, self.binding_point)

    def delete(self):
        if self.id:
            glDeleteBuffers(1, [self.id])
            self.id = 0






class VAOCache:
    """Cache VAOs to avoid re-creating identical attribute setups.

    Usage:
      vao = vao_cache.get_or_create(program, vertex_format, vbo_id, ebo_id)
      glBindVertexArray(vao)
    vertex_format: iterable of tuples describing attributes. Supported forms per attribute:
      (index, size, gl_type)
      (index, size, gl_type, offset)
      (index, size, gl_type, stride, offset)
      (index, size, gl_type, normalized, stride, offset)
    Offsets/strides are in bytes. normalized is truthy/False.
    The implementation is permissive to accept VBO/EBO as integers or objects with .id.
    """
    def __init__(self):
        self._cache: dict[str, int] = {}

    def _make_key(self, program, vertex_format, vbo_id, ebo_id):
        # normalize ids
        prog_id = int(program)
        vb = int(vbo_id) if vbo_id is not None else 0
        eb = int(ebo_id) if ebo_id is not None else 0
        # stable representation of vertex format
        vf_repr = repr(tuple(tuple(x for x in entry) for entry in vertex_format))
        return (prog_id, vb, eb, vf_repr)

    def get_or_create(self, program, vertex_format, vbo, ebo):
        # accept vbo/ebo as objects with .id or plain ints
        vbo_id = int(vbo.id) if hasattr(vbo, "id") else int(vbo) if vbo is not None else 0
        ebo_id = int(ebo.id) if hasattr(ebo, "id") else int(ebo) if ebo is not None else 0

        key = self._make_key(program, vertex_format, vbo_id, ebo_id)
        if key in self._cache:
            return self._cache[key]

        # create VAO
        vao = glGenVertexArrays(1)
        # glGenVertexArrays may return a sequence in some PyOpenGL builds
        if isinstance(vao, (list, tuple)):
            vao = vao[0]
        glBindVertexArray(vao)

        # bind VBO
        if vbo_id:
            glBindBuffer(GL_ARRAY_BUFFER, vbo_id)
        # bind EBO while VAO is bound so association is stored
        if ebo_id:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_id)

        # setup attributes
        for attr in vertex_format:
            if len(attr) < 3:
                raise ValueError("vertex_format entries must have at least (index, size, gl_type)")
            idx = int(attr[0])
            size = int(attr[1])
            gl_type = int(attr[2])

            # defaults
            normalized = False
            stride = 0
            offset = 0

            rest = attr[3:]
            if len(rest) == 1:
                # assume offset
                offset = int(rest[0])
            elif len(rest) == 2:
                # assume stride, offset
                stride = int(rest[0])
                offset = int(rest[1])
            elif len(rest) >= 3:
                # assume normalized, stride, offset
                normalized = bool(rest[0])
                stride = int(rest[1])
                offset = int(rest[2])

            glEnableVertexAttribArray(idx)
            glVertexAttribPointer(idx, size, gl_type,
                                  GL_TRUE if normalized else GL_FALSE,
                                  stride, ctypes.c_void_p(offset))

        # unbind VAO (EBO binding is stored in VAO; unbind ARRAY_BUFFER)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self._cache[key] = int(vao)
        return int(vao)

    def clear(self):
        """Delete all cached VAOs from GL and clear cache."""
        for vao in list(self._cache.values()):
            try:
                glDeleteVertexArrays(1, [vao])
            except Exception:
                pass
        self._cache.clear()

# module-level cache instance
_vao_cache = VAOCache()




class Texture:
    def __init__(self, path, nearest=False):
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        if nearest:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.load_texture(path)

    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = image.convert("RGBA").tobytes()
        glBindTexture(GL_TEXTURE_2D, self.id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.id)

class TextureCube:
    def __init__(self, faces):
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)

        for i, face in enumerate(faces):
            try:
                image = Image.open(face)
                if image.mode == 'RGBA':
                    img_data = image.convert('RGBA').tobytes()
                    gl_format = GL_RGBA
                else:
                    img_data = image.convert('RGB').tobytes()
                    gl_format = GL_RGB
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl_format,
                             image.width, image.height, 0, gl_format, GL_UNSIGNED_BYTE, img_data)
            except Exception as e:
                print(f"    Failed to load face {face}: {e}")

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def bind(self):
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.id)






class Pipeline:
    def __init__(self, program, vertex_format: list[tuple[int, int, Constant, int]]):
        self.program = program
        self.vertex_format = vertex_format
    
    def bind(self, vbo, ebo):
        vbo_id = vbo.id if hasattr(vbo, "id") else int(vbo) if vbo is not None else 0
        ebo_id = ebo.id if hasattr(ebo, "id") else int(ebo) if ebo is not None else 0
        vao = _vao_cache.get_or_create(self.program, self.vertex_format, vbo_id, ebo_id)
        glBindVertexArray(vao)
        




############
# TEMP
############

import ctypes
import os
from typing import Optional, List

import numpy as np
from OpenGL.GL import *

from glib import Texture

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
        self._u_emissive = glGetUniformLocation(self.prog, "uEmissive")

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

    def set_per_object(self, model_mat: np.ndarray, albedo, specular=None, shininess=None, diffuse_factor=None, emissive=False) -> None:
        """Set per-object uniforms (model, albedo, optional specular, shininess, diffuse_factor, emissive).

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
        # emissive
        if self._u_emissive != -1:
            glUniform1i(self._u_emissive, 1 if emissive else 0)

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