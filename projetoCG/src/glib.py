from typing import Optional
from OpenGL.GL import *
import ctypes
import numpy as np

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

        # shaders can be deleted after linking
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

    # uniform helpers (keeps API similar to existing code)
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
        # expects column-major (OpenGL). If your matrices are row-major, transpose or set GL_TRUE.
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






class Pipeline:
    def __init__(self, program, vertex_format: list[tuple[int, int, Constant, int]]):
        self.program = program
        self.vertex_format = vertex_format
    
    def bind(self, vbo, ebo):
        vbo_id = vbo.id if hasattr(vbo, "id") else int(vbo) if vbo is not None else 0
        ebo_id = ebo.id if hasattr(ebo, "id") else int(ebo) if ebo is not None else 0
        vao = _vao_cache.get_or_create(self.program, self.vertex_format, vbo_id, ebo_id)
        glBindVertexArray(vao)
        

    