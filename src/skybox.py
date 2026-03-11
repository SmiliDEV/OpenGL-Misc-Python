import glob
import os
from OpenGL.GL import *
import numpy as np
from PIL import Image
from .file import get_content_of_file_project

import ctypes
from .node import Node
from .glib import *

class Skybox:
    def load_cubemap(self, folder: str):
        """Load cubemap faces from `folder` and return GL texture id.

        Looks for files with common names: right,left,top,bottom,front,back
        and common extensions (.png, .jpg). Returns integer texture id.
        """
        # possible face basenames in the expected order for GL_TEXTURE_CUBE_MAP_POSITIVE_X .. POSITIVE_X+5
        names = ['right', 'left', 'top', 'bottom', 'front', 'back']
        exts = ['.png', '.jpg', '.jpeg']
        faces = []
        for n in names:
            found = None
            for e in exts:
                p = os.path.join(folder, n + e)
                if os.path.exists(p):
                    found = p
                    break
            if found is None:
                # try any file starting with the basename
                candidates = glob.glob(os.path.join(folder, n + '.*'))
                if candidates:
                    found = candidates[0]
            if found is None:
                raise FileNotFoundError(f"Cubemap face not found for '{n}' in '{folder}'")
            faces.append(found)

        tex = TextureCube(faces)
        return tex.id

    def __init__(self, shader_program: int, texture_folder: str = None):
        if texture_folder is None:
            texture_folder = os.path.join(os.path.dirname(__file__), 'assets/textures', 'skybox')

        self.cubemap_tex = self.load_cubemap(texture_folder)
        self.skybox_program = shader_program
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

def attach_skybox_node(node: Node, folder: str = None):
    """Attach a cubemap texture and compiled skybox shader to `node`.
    `node` is expected to have a `mesh` (the cube mesh). The Node's name
    should be 'Skybox' (convention used by `main.py`).
    Returns the node on success, or None on failure.
    """
    if node is None:
        return None
    if folder is None:
        folder = os.path.join(os.path.dirname(__file__), 'textures', 'skybox')

    # instantiate loader which prepares a VAO, VBO, shader and loads the cubemap
    try:
        loader = Skybox()
    except Exception:
        return None

    # attach loader resources to node so the specialized draw helper can use them
    node.cubemap_tex = getattr(loader, 'cubemap_tex', None)
    node.skybox_program = getattr(loader, 'skybox_program', None)
    node.skybox_vao = getattr(loader, 'skybox_vao', None)
    node.skybox_vbo = getattr(loader, 'skybox_vbo', None)
    return node


def draw_skybox_node(node: Node, view: np.ndarray, projection: np.ndarray):
    """Draw the node that was prepared with `attach_skybox_node`.
    This mirrors the old Skybox.draw behaviour but operates on a Node.
    """
    if node is None:
        return
    prog = getattr(node, 'skybox_program', None)
    tex = getattr(node, 'cubemap_tex', None)
    if prog is None or tex is None or getattr(node, 'mesh', None) is None:
        return

    # remove translation so skybox stays centered on camera
    view_rot = view.copy()
    view_rot[0:3, 3] = 0.0

    prev_depth_func = glGetInteger(GL_DEPTH_FUNC)
    glDepthFunc(GL_LEQUAL)
    glDepthMask(GL_FALSE)

    # disable face culling so the inside faces of the cube are visible
    prev_cull = glIsEnabled(GL_CULL_FACE)
    glDisable(GL_CULL_FACE)

    glUseProgram(prog)
    loc_p = glGetUniformLocation(prog, 'uProj')
    if loc_p != -1:
        glUniformMatrix4fv(loc_p, 1, GL_TRUE, projection.astype(np.float32))
    loc_v = glGetUniformLocation(prog, 'uView')
    if loc_v != -1:
        glUniformMatrix4fv(loc_v, 1, GL_TRUE, view_rot.astype(np.float32))

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)

    # Prefer the VAO prepared by the Skybox loader if available; fall back to node.mesh
    vao = getattr(node, 'skybox_vao', None)
    if vao is not None:
        glBindVertexArray(vao)
        # loader builds 36 vertices (6 faces * 2 tris * 3 verts)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)
    else:
        # fallback to existing mesh draw path
        node.mesh.draw()

    glUseProgram(0)

    glDepthMask(GL_TRUE)
    glDepthFunc(prev_depth_func)

    glEnable(GL_CULL_FACE)


def draw_skybox_loader(loader: 'Skybox', view: np.ndarray, projection: np.ndarray):
    """Draw directly from a Skybox loader instance (external skybox, not a Node).
    This keeps the skybox out of the scene graph and uses the loader's VAO/tex/prog.
    """
    if loader is None:
        return
    prog = getattr(loader, 'skybox_program', None)
    tex = getattr(loader, 'cubemap_tex', None)
    vao = getattr(loader, 'skybox_vao', None)
    if prog is None or tex is None:
        return

    # remove translation so skybox stays centered on camera
    view_rot = view.copy()
    view_rot[0:3, 3] = 0.0

    prev_depth_func = glGetInteger(GL_DEPTH_FUNC)
    glDepthFunc(GL_LEQUAL)
    glDepthMask(GL_FALSE)

    # disable face culling so the inside faces of the cube are visible
    prev_cull = glIsEnabled(GL_CULL_FACE)
    glDisable(GL_CULL_FACE)

    glUseProgram(prog)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex)

    if vao is not None:
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)
    else:
        # no VAO: nothing to draw
        pass

    glUseProgram(0)

    glDepthMask(GL_TRUE)
    glDepthFunc(prev_depth_func)

    # restore face culling state
    if prev_cull:
        glEnable(GL_CULL_FACE)