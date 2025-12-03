import numpy as np
from OpenGL.GL import *

class Renderer:
    """Lightweight scene renderer that minimizes shader switches.

    Usage:
      r = Renderer()
      r.render(root, vp, view_pos, light_dir, ambient, default_shader=..., common_setup=callable)

    `common_setup` is a callable(shader) used to upload per-shader globals
    (e.g. view/proj matrices and lighting) when a new shader is bound.
    """

    def __init__(self):
        self._current_shader = None
        # cache of lights collected each frame (list of dicts)
        self._lights = []

    def _use_shader(self, shader, common_setup=None, lights=None):
        if shader is None:
            return
        if shader is not self._current_shader:
            # activate shader
            if hasattr(shader, 'use'):
                shader.use()
            elif hasattr(shader, 'prog'):
                glUseProgram(shader.prog)
            else:
                try:
                    glUseProgram(int(shader))
                except Exception:
                    pass
            self._current_shader = shader
            if callable(common_setup):
                try:
                    common_setup(shader, lights)
                except TypeError:
                    # fallback to older call signature
                    common_setup(shader)

    def _render_node(self, node, parent_world, vp, view_pos, light_dir, ambient, default_shader, common_setup, lights):
        world = parent_world @ node.local

        # determine which shader will be used for this node (material preferred)
        mat = getattr(node, 'material', None)
        if mat is not None and getattr(mat, 'shader', None) is not None:
            shader = mat.shader
        else:
            shader = default_shader

        # ensure shader is active and common per-shader uniforms are set
        self._use_shader(shader, common_setup, lights)

        # Skip special skybox nodes: they are drawn with a separate helper
        if getattr(node, 'is_skybox', False) or getattr(node, 'skybox_program', None) is not None:
            # recurse without drawing this node; skybox will be drawn separately
            for c in node.children:
                self._render_node(c, world, vp, view_pos, light_dir, ambient, default_shader, common_setup, lights)
            return

        if node.mesh is not None:
            # If material present, let it bind per-object uniforms (assumes shader is active)
            if mat is not None:
                if hasattr(mat, 'bind'):
                    mat.bind(world)
                else:
                    # fallback to older apply() naming
                    if hasattr(mat, 'apply'):
                        mat.apply(shader, node, world, vp, view_pos, light_dir, ambient)
                    else:
                        # no way to bind per-object uniforms from material
                        raise RuntimeError(f"Material for node '{node.name}' cannot bind uniforms")
            else:
                # No material: prefer node.on_draw for custom per-object setup.
                # Fail fast and instruct the caller to attach a Material to the
                # mesh-owning node or implement `node.on_draw(...)` if custom
                # per-object uniforms are required.
                if callable(getattr(node, 'on_draw', None)):
                    node.on_draw(shader, node, world, vp, view_pos, light_dir, ambient)
                else:
                    raise RuntimeError(
                        f"Node '{node.name}' has no Material and no on_draw; attach a Material to this node "
                        "or implement node.on_draw(shader, node, world, vp, view_pos, light_dir, ambient)."
                    )

            # draw geometry
            node.mesh.draw()

        # recurse
        for c in node.children:
            self._render_node(c, world, vp, view_pos, light_dir, ambient, default_shader, common_setup, lights)

    def _collect_lights(self, node, parent_world=None, max_lights=4):
        if parent_world is None:
            parent_world = np.eye(4, dtype=np.float32)
        lights = []
        world = parent_world @ node.local
        if getattr(node, 'is_light', False):
            # world position of the light (transform origin)
            p = world @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            lights.append({'pos': p[0:3].tolist(), 'col': node.light_color.tolist(), 'int': float(node.light_intensity)})
        for c in node.children:
            if len(lights) >= max_lights:
                break
            lights.extend(self._collect_lights(c, world, max_lights))
        return lights

    def render(self, root, vp, view_pos, light_dir, ambient, default_shader=None, common_setup=None):
        self._current_shader = None
        # collect point lights from the scene graph
        self._lights = self._collect_lights(root)
        self._render_node(root, np.eye(4, dtype=np.float32), vp, view_pos, light_dir, ambient, default_shader, common_setup, self._lights)
