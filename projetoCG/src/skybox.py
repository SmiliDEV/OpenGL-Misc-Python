import os
from texture import TextureCube
from OpenGL.GL import *
from gfx import *
import numpy as np

class Skybox:
    def __init__(self):
        self.texture = TextureCube([
            os.path.join(os.path.dirname(__file__), 'textures', 'skybox', 'right.jpg'),
           os.path.join(os.path.dirname(__file__), 'textures', 'skybox', 'left.jpg'),
           os.path.join(os.path.dirname(__file__), 'textures', 'skybox', 'top.jpg'),
           os.path.join(os.path.dirname(__file__), 'textures', 'skybox', 'bottom.jpg'),
           os.path.join(os.path.dirname(__file__), 'textures', 'skybox', 'front.jpg'),
           os.path.join(os.path.dirname(__file__), 'textures', 'skybox', 'back.jpg'),
       ])
        self.skyboxShader = wrapperCreateShader('skybox')
        self.skyboxVertices = np.array([
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

        self.skyboxVAO = glGenVertexArrays(1)
        self.skyboxVBO = glGenBuffers(1)
        glBindVertexArray(self.skyboxVAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.skyboxVBO)
        glBufferData(GL_ARRAY_BUFFER, self.skyboxVertices.nbytes, self.skyboxVertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, 0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        self.skyboxShader.use()
        # sampler should be set to the texture unit index (0), not the GL texture id
        self.skyboxShader.setInt("skybox", self.texture.id)
        # diagnostic info
        try:
            print(f"Skybox: texture id={self.texture.id} VAO={self.skyboxVAO} VBO={self.skyboxVBO} shader_prog={self.skyboxShader.prog}")
            print(f"  vertex_count={len(self.skyboxVertices)//3}")
        except Exception:
            pass
        self._did_log_draw = False
    
    def draw(self, view, projection):
        if not self._did_log_draw:
            print("Skybox.draw called (first frame)")
            self._did_log_draw = True
        # DEBUG: force draw regardless of depth to rule out depth/state issues
        glDepthFunc(GL_ALWAYS)
        glDisable(GL_DEPTH_TEST)
        self.skyboxShader.use()
        # remove translation from the view matrix
        # remove translation from the view matrix without glm
        view_mat = np.array(view, dtype=np.float32)
        if view_mat.size == 16:
            view_mat = view_mat.reshape((4, 4))
        view_mat = view_mat.copy()
        view_mat[:3, 3] = 0.0
        view_mat[3, :3] = 0.0
        view_mat[3, 3] = 1.0
        view = view_mat
        self.skyboxShader.setMat4("view", view)
        self.skyboxShader.setMat4("projection", projection)
        glBindVertexArray(self.skyboxVAO)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture.id)
        # don't write to depth buffer when drawing the skybox
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glBindVertexArray(0)
        # restore depth state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)