from OpenGL.GL import *
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ctypes

VERT_SRC = '''#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
'''

FRAG_SRC = '''#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec4 uColor;
void main() {
    vec4 t = texture(uTex, vUV);
    FragColor = vec4(uColor.rgb, uColor.a * t.a) * t;
}
'''


class TextRenderer:
    def __init__(self, width, height, font_path=None, font_size=24):
        self.width = int(width)
        self.height = int(height)
        self.font_size = font_size
        self.font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, font_size)
        # compile simple shader
        self.program = self._create_program(VERT_SRC, FRAG_SRC)
        # create VAO/VBO for quad (we'll update data per draw)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # vertex: x,y,u,v (4 floats)
        glBufferData(GL_ARRAY_BUFFER, 4 * 4 * ctypes.sizeof(ctypes.c_float), None, GL_DYNAMIC_DRAW)
        # attributes
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(2 * ctypes.sizeof(ctypes.c_float)))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _create_program(self, vsrc, fsrc):
        vert = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vert, vsrc)
        glCompileShader(vert)
        if not glGetShaderiv(vert, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(vert))
        frag = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(frag, fsrc)
        glCompileShader(frag)
        if not glGetShaderiv(frag, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(frag))
        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(prog))
        glDeleteShader(vert)
        glDeleteShader(frag)
        return prog

    def draw_text(self, x, y, text, color=(1.0, 1.0, 1.0, 1.0)):
        # Render text into image
        img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        size = draw.textsize(text, font=self.font)
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, font=self.font, fill=(255, 255, 255, 255))
        w, h = img.size
        # upload as texture
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        img_data = img.transpose(Image.FLIP_TOP_BOTTOM).tobytes()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

        # compute quad in NDC
        x0 = (2.0 * x) / self.width - 1.0
        y0 = 1.0 - (2.0 * y) / self.height
        x1 = (2.0 * (x + w)) / self.width - 1.0
        y1 = 1.0 - (2.0 * (y + h)) / self.height

        verts = np.array([
            x0, y1, 0.0, 0.0,
            x0, y0, 0.0, 1.0,
            x1, y0, 1.0, 1.0,
            x1, y1, 1.0, 0.0,
        ], dtype=np.float32)

        # upload vertex data
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

        # draw
        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex)
        loc = glGetUniformLocation(self.program, 'uTex')
        if loc != -1:
            glUniform1i(loc, 0)
        locc = glGetUniformLocation(self.program, 'uColor')
        if locc != -1:
            glUniform4f(locc, *color)

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

        # cleanup texture
        glDeleteTextures([tex])

    def destroy(self):
        try:
            if self.vbo:
                glDeleteBuffers(1, [self.vbo])
        except Exception:
            pass
        try:
            if self.vao:
                glDeleteVertexArrays(1, [self.vao])
        except Exception:
            pass
        try:
            if self.program:
                glDeleteProgram(self.program)
        except Exception:
            pass
