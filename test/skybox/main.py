import os
import sys
import math
import glfw
from OpenGL.GL import *
import ctypes
import numpy as np
from PIL import Image
import glm


VERTEX_SHADER_SKYBOX = '''#version 330 core
layout(location = 0) in vec3 aPos;
out vec3 TexCoords;
uniform mat4 projection;
uniform mat4 view;
void main()
{
    TexCoords = aPos;
    mat4 rotView = mat4(mat3(view));
    gl_Position = projection * rotView * vec4(aPos, 1.0);
}
'''

FRAGMENT_SHADER_SKYBOX = '''#version 330 core
in vec3 TexCoords;
out vec4 FragColor;
uniform samplerCube skybox;
void main()
{
    FragColor = texture(skybox, TexCoords);
}
'''


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        err = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error: {err}")
    return shader


def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        err = glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error: {err}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return prog


def load_cubemap(folder):
    # Order expected: right, left, top, bottom, front, back
    bases = ["right", "left", "top", "bottom", "front", "back"]
    # gather files in folder
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
            # try any file that starts with base name
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


class Camera:
    def __init__(self, pos=glm.vec3(0,0,3), yaw=-90.0, pitch=0.0):
        self.pos = pos
        self.yaw = yaw
        self.pitch = pitch
        self.front = glm.vec3(0,0,-1)
        self.up = glm.vec3(0,1,0)
        self.right = glm.vec3(1,0,0)
        self.speed = 2.5
        self.sensitivity = 0.1
        self._update_vectors()

    def _update_vectors(self):
        yaw_r = glm.radians(self.yaw)
        pitch_r = glm.radians(self.pitch)
        front = glm.vec3(
            glm.cos(yaw_r) * glm.cos(pitch_r),
            glm.sin(pitch_r),
            glm.sin(yaw_r) * glm.cos(pitch_r)
        )
        self.front = glm.normalize(front)
        self.right = glm.normalize(glm.cross(self.front, glm.vec3(0,1,0)))
        self.up = glm.normalize(glm.cross(self.right, self.front))

    def get_view(self):
        return glm.lookAt(self.pos, self.pos + self.front, self.up)


def main():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    width, height = 1280, 720
    window = glfw.create_window(width, height, "Skybox - GLFW + PyOpenGL", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create window")
        return

    glfw.make_context_current(window)

    # Skybox cube (positions only)
    skybox_vertices = np.array([
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

    # Setup skybox VAO
    skybox_vao = glGenVertexArrays(1)
    skybox_vbo = glGenBuffers(1)
    glBindVertexArray(skybox_vao)
    glBindBuffer(GL_ARRAY_BUFFER, skybox_vbo)
    glBufferData(GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    glBindVertexArray(0)

    # Compile skybox shader
    skybox_program = link_program(VERTEX_SHADER_SKYBOX, FRAGMENT_SHADER_SKYBOX)
    # ensure the sampler uses texture unit 0
    glUseProgram(skybox_program)
    loc = glGetUniformLocation(skybox_program, 'skybox')
    if loc != -1:
        glUniform1i(loc, 0)
    glUseProgram(0)

    # Load cubemap
    cubemap_folder = os.path.join(os.path.dirname(__file__), 'textures', 'skybox')
    if not os.path.isdir(cubemap_folder):
        print(f"Aviso: pasta de texturas não encontrada: {cubemap_folder}")
        print("Coloque as 6 texturas na pasta: textures/skybox com nomes: right, left, top, bottom, front, back (com extensão .jpg/.png)")
        glfw.terminate()
        return

    try:
        cubemap_tex = load_cubemap(cubemap_folder)
    except Exception as e:
        print(f"Erro ao carregar cubemap: {e}")
        glfw.terminate()
        return

    glEnable(GL_DEPTH_TEST)

    cam = Camera()
    last_x, last_y = width / 2, height / 2
    first_mouse = True

    def mouse_callback(window, xpos, ypos):
        nonlocal last_x, last_y, first_mouse
        if first_mouse:
            last_x = xpos
            last_y = ypos
            first_mouse = False
        xoffset = xpos - last_x
        yoffset = last_y - ypos
        last_x = xpos
        last_y = ypos
        xoffset *= cam.sensitivity
        yoffset *= cam.sensitivity
        cam.yaw += xoffset
        cam.pitch += yoffset
        if cam.pitch > 89.0:
            cam.pitch = 89.0
        if cam.pitch < -89.0:
            cam.pitch = -89.0
        cam._update_vectors()

    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    glfw.set_cursor_pos_callback(window, mouse_callback)

    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        delta = current_time - last_time
        last_time = current_time

        # input
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        camera_speed = cam.speed * delta
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            cam.pos += cam.front * camera_speed
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            cam.pos -= cam.front * camera_speed
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            cam.pos -= cam.right * camera_speed
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            cam.pos += cam.right * camera_speed

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw skybox
        glDepthFunc(GL_LEQUAL)
        glUseProgram(skybox_program)
        proj = glm.perspective(glm.radians(45.0), width / height, 0.1, 100.0)
        view = cam.get_view()
        # upload matrices
        proj_loc = glGetUniformLocation(skybox_program, 'projection')
        view_loc = glGetUniformLocation(skybox_program, 'view')
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, glm.value_ptr(proj))
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))

        glBindVertexArray(skybox_vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_tex)
        glDrawArrays(GL_TRIANGLES, 0, 36)
        glBindVertexArray(0)
        glUseProgram(0)
        glDepthFunc(GL_LESS)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == '__main__':
    main()
