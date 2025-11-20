import numpy as np
from OpenGL.GL import *

class Material:
    """Simple material wrapper.

    Responsibility: own the GPU-side appearance of a mesh.

    - `shader`: a ShaderProgram-like object (must be activated by the Renderer)
    - `albedo_color`: fallback color when no texture is supplied
    - `albedo_texture`: optional Texture instance

    API:
      - `bind(world)` : sets per-object uniforms (model matrix, albedo or sampler).

    Note: `bind` assumes the material's `shader` is already active. The
    `Renderer` is responsible for calling `shader.use()` and for uploading
    global uniforms (view/proj/light).
    """

    def __init__(self, shader=None, albedo_color=(1.0, 1.0, 1.0), albedo_texture=None):
        self.shader = shader
        self.albedo_color = np.array(albedo_color, dtype=np.float32)
        self.albedo_texture = albedo_texture

    @classmethod
    def from_color(cls, shader, color):
        return cls(shader=shader, albedo_color=color, albedo_texture=None)

    @classmethod
    def from_texture(cls, shader, texture):
        return cls(shader=shader, albedo_color=(1.0, 1.0, 1.0), albedo_texture=texture)

    def bind(self, world):
        """Set per-object uniforms. Assumes the calling code already bound the shader.

        This will prefer the shader's `set_per_object(model, albedo)` helper when
        available. Otherwise it falls back to setting `uModel` and `uColor` and
        binding a texture to unit 0 if present.
        """
        shader_obj = getattr(self, 'shader', None)

        if shader_obj is not None and hasattr(shader_obj, 'set_per_object'):
            # shader helper expects (model_mat, albedo)
            shader_obj.set_per_object(world, self.albedo_color)
            return

        prog = getattr(shader_obj, 'prog', None) if shader_obj is not None else None
        if prog is not None:
            loc_m = glGetUniformLocation(prog, 'uModel')
            if loc_m != -1:
                glUniformMatrix4fv(loc_m, 1, GL_TRUE, world)
            loc_c = glGetUniformLocation(prog, 'uColor')
            if loc_c != -1:
                glUniform3fv(loc_c, 1, self.albedo_color)

        # bind albedo texture (unit 0) and set sampler uniform if present
        if self.albedo_texture is not None:
            try:
                glActiveTexture(GL_TEXTURE0)
                if hasattr(self.albedo_texture, 'bind'):
                    self.albedo_texture.bind()
                else:
                    glBindTexture(GL_TEXTURE_2D, getattr(self.albedo_texture, 'id', 0))
                if prog is not None:
                    loc = glGetUniformLocation(prog, 'texture1')
                    if loc != -1:
                        glUniform1i(loc, 0)
                    else:
                        loc2 = glGetUniformLocation(prog, 'uAlbedoSampler')
                        if loc2 != -1:
                            glUniform1i(loc2, 0)
            except Exception:
                # binding failure should not crash the whole renderer
                pass
