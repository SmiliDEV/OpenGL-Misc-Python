import os
from OpenGL.GL import *
import glm
import numpy as np
from PIL import Image
from glib import *
from file import get_content_of_file_project

class Skybox:
    def __init__(self):
        self.cubemap_tex = self.load_cubemap('textures/skybox')
        
        vs_src = get_content_of_file_project('shaders/skybox.vert')
        fs_src = get_content_of_file_project('shaders/skybox.frag')
        shader = Shader(vs_src, fs_src)
        self.skybox_program = shader.prog

        self.skybox_vertices = np.array([
            -1.0,  1.0, -1.0,
            -1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0, -1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,
            -1.0,  1.0,  1.0,
            -1.0, -1.0,  1.0,

            1.0, -1.0, -1.0,
            1.0, -1.0,  1.0,
            1.0,  1.0,  1.0,
            1.0,  1.0,  1.0,
            1.0,  1.0, -1.0,
            1.0, -1.0, -1.0,

            -1.0, -1.0,  1.0,
            -1.0,  1.0,  1.0,
            1.0,  1.0,  1.0,
            1.0,  1.0,  1.0,
            1.0, -1.0,  1.0,
            -1.0, -1.0,  1.0,

            -1.0,  1.0, -1.0,
            1.0,  1.0, -1.0,
            1.0,  1.0,  1.0,
            1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            -1.0,  1.0, -1.0,

            -1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
            1.0, -1.0, -1.0,
            1.0, -1.0, -1.0,
            -1.0, -1.0,  1.0,
            1.0, -1.0,  1.0
        ], dtype=np.float32)

        self.skybox_vao = glGenVertexArrays(1)
        self.skybox_vbo = glGenBuffers(1)
        glBindVertexArray(self.skybox_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.skybox_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.skybox_vertices.nbytes, self.skybox_vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glBindVertexArray(0)

        glUseProgram(self.skybox_program)
        loc = glGetUniformLocation(self.skybox_program, 'skybox')
        if loc != -1:
            glUniform1i(loc, 0)
        glUseProgram(0)

    def load_cubemap(self, folder):
        bases = ["right", "left", "top", "bottom", "front", "back"]

        files = os.listdir(folder)
        if not files:
            raise FileNotFoundError(f"No files found in {folder}")

        faces = []
        for b in bases:
            found = None
            for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                candidate = f"{b}{ext}"
                if candidate in files:
                    found = os.path.join(folder, candidate)
                    break
            if found is None:
                for f in files:
                    if f.lower().startswith(b.lower()):
                        found = os.path.join(folder, f)
                        break
            if found is None:
                raise FileNotFoundError(f"Could not find texture for face '{b}' in {folder}. Found: {files}")
            faces.append(found)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)

        for i, face in enumerate(faces):
            img = Image.open(face).convert('RGB')
            img_data = img.tobytes()
            width, height = img.size
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        return tex_id

    def draw(self, view, projection):
        glDepthFunc(GL_LEQUAL)
        glUseProgram(self.skybox_program)
        glBindVertexArray(self.skybox_vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.cubemap_tex)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)