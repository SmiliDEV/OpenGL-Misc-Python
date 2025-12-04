import numpy as np
from OpenGL.GL import *

class Material:
  


    def __init__(self, shader=None, albedo_color=(1.0, 1.0, 1.0), albedo_texture=None,
                 specular_color=(1.0, 1.0, 1.0), shininess=32.0, diffuse_scale=1.0, emissive=False):
        self.shader = shader
        self.albedo_color = np.array(albedo_color, dtype=np.float32)
        self.albedo_texture = albedo_texture
        self.specular_color = np.array(specular_color, dtype=np.float32)
        self.shininess = float(shininess)
        self.diffuse_scale = float(diffuse_scale)
        self.emissive = emissive

    @classmethod
    def from_color(cls, shader, color):
        return cls(shader=shader, albedo_color=color, albedo_texture=None)

    @classmethod
    def from_texture(cls, shader, texture):
        return cls(shader=shader, albedo_color=(1.0, 1.0, 1.0), albedo_texture=texture)

    def bind(self, world):
        shader_obj = getattr(self, 'shader', None)

        if shader_obj is not None and hasattr(shader_obj, 'set_per_object'):
            # Try extended helper signature: (model, albedo, specular, shininess, diffuse_scale, emissive)
            try:
                shader_obj.set_per_object(world, self.albedo_color, self.specular_color, self.shininess, self.diffuse_scale, self.emissive)
                return
            except TypeError:
                # try older extended signature without emissive
                try:
                    shader_obj.set_per_object(world, self.albedo_color, self.specular_color, self.shininess, self.diffuse_scale)
                    return
                except TypeError:
                    # try older extended signature without diffuse_scale
                    try:
                        shader_obj.set_per_object(world, self.albedo_color, self.specular_color, self.shininess)
                        return
                    except TypeError:
                        # fallback to oldest signature
                        try:
                            shader_obj.set_per_object(world, self.albedo_color)
                            return
                        except TypeError:
                            pass

        prog = getattr(shader_obj, 'prog', None) if shader_obj is not None else None
        if prog is not None:
            loc_m = glGetUniformLocation(prog, 'uModel')
            if loc_m != -1:
                glUniformMatrix4fv(loc_m, 1, GL_TRUE, world)
            loc_c = glGetUniformLocation(prog, 'uAlbedo')
            if loc_c != -1:
                glUniform3fv(loc_c, 1, self.albedo_color)
            # specular + shininess (Blinn-Phong)
            loc_s = glGetUniformLocation(prog, 'uSpecularColor')
            if loc_s != -1:
                glUniform3fv(loc_s, 1, self.specular_color)
            loc_sh = glGetUniformLocation(prog, 'uShininess')
            if loc_sh != -1:
                glUniform1f(loc_sh, self.shininess)
            
            loc_em = glGetUniformLocation(prog, 'uEmissive')
            if loc_em != -1:
                glUniform1i(loc_em, 1 if self.emissive else 0)
            # diffuse scale uniform
            loc_ds = glGetUniformLocation(prog, 'uDiffuseFactor')
            if loc_ds != -1:
                glUniform1f(loc_ds, self.diffuse_scale)

        # bind albedo texture (unit 0) and set sampler uniform if present
        if self.albedo_texture is not None:
            try:
                glActiveTexture(GL_TEXTURE0)
                tex_id = getattr(self.albedo_texture, 'id', 0)
                if hasattr(self.albedo_texture, 'bind'):
                    self.albedo_texture.bind()
                else:
                    glBindTexture(GL_TEXTURE_2D, tex_id)
                
                if prog is not None:
                    # prefer canonical sampler name for textured materials
                    loc = glGetUniformLocation(prog, 'uAlbedoSampler')
                    if loc != -1:
                        glUniform1i(loc, 0)
                    else:
                        # backwards compatibility with older shaders
                        loc2 = glGetUniformLocation(prog, 'texture1')
                        if loc2 != -1:
                            glUniform1i(loc2, 0)
            except Exception as e:
                print(f"Material binding error: {e}")
                pass
