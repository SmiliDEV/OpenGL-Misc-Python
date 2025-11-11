from OpenGL.GL import *
from PIL import Image

class Texture:
    def __init__(self, path):
        self.id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
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

        print(f"TextureCube: creating cubemap id={self.id} with {len(faces)} faces")
        for i, face in enumerate(faces):
            try:
                print(f"  Loading face {i}: {face}")
                image = Image.open(face)
                print(f"    mode={image.mode} size={image.size}")
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                # convert to RGB unless image already has alpha and we want RGBA
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