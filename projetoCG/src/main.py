# OpenGL com Shaders (>3.3)

import sys, math, ctypes, os
import numpy as np
import glfw
from OpenGL.GL import *
import anim  # módulo separado com animadores (src/anim.py)
from carro import Car  # estado físico do carro
from camera import Camera, update_free_camera, get_view_free
from node import Node
from gfx import MeshTextured, ShaderProgram, Mesh, wrapperCreateShader
from math3d import *
from geo import *
from skybox import Skybox, draw_skybox_loader
from renderer import Renderer
from texture import *
from window import Window
from glib import *
from material import Material
from file import *

width = 800
height = 600

cam = Camera()
last_x, last_y = width / 2, height / 2
first_mouse = True
follow_cam = None

def cursor_pos_callback(window, xpos, ypos):
    global last_x, last_y, first_mouse, cam
    if first_mouse:
        last_x = xpos
        last_y = ypos
        first_mouse = False
    xoffset = xpos - last_x
    yoffset = last_y - ypos
    
    last_x = xpos
    last_y = ypos

    # only apply mouse-look when the free camera is actively looking (RMB held)
    if getattr(cam, '_looking', False):
        xoffset *= cam.mouse_sens
        yoffset *= cam.mouse_sens
        cam.yaw += xoffset
        cam.pitch -= yoffset
        cam.pitch = max(math.radians(-85.0), min(math.radians(85.0), cam.pitch))

def key_callback(win, key, sc, action, mods):
    global follow_cam, cam
    if action in (glfw.PRESS, glfw.REPEAT):
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_P and action == glfw.PRESS:
            # Toggle camera mode. If switching to free, re-seed free cam from current follow view
            new_mode = 'free' if cam.mode != 'free' else 'follow'
            if new_mode == 'free' and follow_cam is not None:
                # Sample the current follow camera position and orient the free cam to match
                eye, ctr = follow_cam(0.0)
                dirv = ctr - eye
                n = np.linalg.norm(dirv)
                if n > 1e-6:
                    dirv = dirv / n
                # yaw = atan2(z, x), pitch = asin(y)
                yaw = math.atan2(dirv[2], dirv[0])
                yclamped = max(-1.0, min(1.0, float(dirv[1])))
                pitch = math.asin(yclamped)
                cam.pos = eye.astype(np.float32)
                cam.yaw = yaw
                cam.pitch = pitch
                cam._looking = False
                cam._last_mouse = None
            cam.mode = new_mode

def framebuffer_size_callback(window, width, height):
    glViewport(0, 0, width, height)

def setup_gl_state():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

def build_scene(meshes: dict, materials: dict):
    """Constructs scene graph from provided meshes and materials.

    Returns (root, nodes) where `nodes` is a dict with named node references
    for easy access (car, car_body, floor, wheels, skybox).
    """
    # ensure required meshes exist
    if 'car' not in meshes or 'plane' not in meshes:
        raise ValueError("meshes must include at least 'car' and 'plane'")

    car_mesh = meshes.get('car')
    plane_mesh = meshes.get('plane')
    wheel_mesh = meshes.get('wheel', None)

    root = Node('Root')

    # Floor
    floor_node = Node('Floor', local=translate(0, 0, 0) @ scale(50, 0.1, 50), mesh=plane_mesh)
    if 'floor' in materials:
        floor_node.material = materials['floor']

    # Car and body
    car_node = Node('Car')
    car_body = Node('CarBody', local=translate(0, 2.00, 0) @ scale(4.2, 0.9, 2.0), mesh=car_mesh)
    if 'car' in materials:
        car_body.material = materials['car']

    car_node.add(car_body)
    root.add(floor_node, car_node)

    # Wheels
    wheels = {}
    if wheel_mesh is not None:
        wheel_radius = 0.55
        wheel_width = 0.30
        x_off, z_off, y_off = 2.3, 1.2, wheel_radius

        def make_wheel(name, ox, oy, oz):
            n = Node(name, local=translate(ox, oy, oz) @ scale(wheel_radius, wheel_radius, wheel_width), mesh=wheel_mesh)
            if 'wheel' in materials:
                n.material = materials['wheel']
            return n

        wheels = {
            'Wheel_FL': make_wheel('Wheel_FL', +x_off, y_off, -z_off),
            'Wheel_FR': make_wheel('Wheel_FR', +x_off, y_y := y_off, +z_off),
            'Wheel_RL': make_wheel('Wheel_RL', -x_off, y_off, -z_off),
            'Wheel_RR': make_wheel('Wheel_RR', -x_off, y_off, +z_off),
        }
        car_node.add(*wheels.values())

    nodes = {
        'root': root,
        'car': car_node,
        'car_body': car_body,
        'floor': floor_node,
        'wheels': wheels,
        
    }
    return root, nodes

def main():
    window = Window(800, 600, "Carro na rua — Grafo de cena com Flat Shading (OpenGL 3.3)")
    glfw.set_cursor_pos_callback(window.win, cursor_pos_callback)
    glfw.set_key_callback(window.win, key_callback)
    glfw.set_framebuffer_size_callback(window.win, framebuffer_size_callback)
    setup_gl_state()

    # UBOs
    uboPV = UniformBuffer(np.zeros(2 * 16, dtype=np.float32), binding_point=0)  # espaço para 2 matrizes 4x4 (P e V)

    # Shaders
    vs_path = os.path.join(os.path.dirname(__file__), 'shaders', 'basic.vert')
    fs_path = os.path.join(os.path.dirname(__file__), 'shaders', 'basic.frag')
    shader = ShaderProgram.from_files(vs_path, fs_path)
    floor_shader = wrapperCreateShader('floor')
    floor_shader.setInt('texture1', 0)  # textura no slot 0
    
    skybox_shader = Shader(get_content_of_file_project('shaders/skybox.vert'),
                           get_content_of_file_project('shaders/skybox.frag'))

    # bind UBO to binding point 0 for both shaders
    uboPV.bind_shader_block(skybox_shader.prog, 'Matrices')
    uboPV.bind_shader_block(shader.prog, 'Matrices')
    uboPV.bind_shader_block(floor_shader.prog, 'Matrices')

    # criar meshes 
    #chao 
    inter, idx = gen_uv_plane_flat(size=1.0, divisions=10)
    texture_floor = Texture(os.path.join(os.path.dirname(__file__), 'textures', 'floor.jpg'))
    plane_mesh = MeshTextured(inter, idx, texture=texture_floor)
    # Mesh do carro 
    inter, idx = gen_uv_car_body(size=1.0)
    car_mesh = Mesh(inter, idx)
    # Mesh de roda cilíndrica (unitária). O anim aplica a escala real.
    inter_w, idx_w = gen_uv_cylinder_flat(radius=1.0, half_width=0.5, slices=32)
    wheel_mesh = Mesh(inter_w, idx_w)
    # small sphere mesh for visualizing lights
    inter_s, idx_s = gen_uv_sphere_flat(radius=1.0, stacks=12, slices=24)
    sphere_mesh = Mesh(inter_s, idx_s)
    # Skybox is handled externally (not part of scene graph)
    mesh_dict = {'car': car_mesh, 'plane': plane_mesh, 'wheel': wheel_mesh}
    resources = [car_mesh, plane_mesh, wheel_mesh, sphere_mesh]
    COL_CAR = (0.8,0.1,0.1)
    COL_WHEEL = (0.1,0.1,0.1)
    materials = {
        'car' : Material.from_color(shader,COL_CAR),
        'floor': Material.from_texture(floor_shader,texture_floor)
    }
    if wheel_mesh is not None:
        materials['wheel'] = Material.from_color(shader, COL_WHEEL)

    root, nodes = build_scene(meshes=mesh_dict, materials=materials)

    # locate the Car node and wheel nodes returned by builder
    car_node = nodes.get('car')
    if car_node is None:
        raise RuntimeError("Car node not found in scene after build_scene()")

    wheel_nodes = nodes.get('wheels', {})

    # assign materials to Car and Floor nodes so drawing is automatic via root.draw()
    car_node.material = Material.from_color(shader, COL_CAR)
    floor_node = root.find('Floor')
    if floor_node is not None:
        # floor is textured: create a material that references the floor shader and texture
        floor_mat = Material.from_texture(floor_shader, texture_floor)
        floor_node.material = floor_mat

    # Estado lógico do carro
    car_state = Car()

    # Criar animador único que trata chassis + rodas
    # attach the car animator to the Car node
    car_node.animator = anim.make_car_animators(
        win=window,
        car_state=car_state,
        car_node=car_node,
        wheel_nodes=wheel_nodes,
        translate=translate, rotate=rotate, scale=scale,
    )

    # Camera a seguir o carro (offset atrás e acima, com smoothing)
    # Camera atrás do carro (e um pouco acima). Para frente = +X, usar offset X negativo.
    global follow_cam
    follow_cam = anim.make_follow_camera(
        lambda: car_node.local.copy(),
        offset_local=(-12.0, 4.0, 0.0),
        look_ahead=8.0,
        lag_seconds=0.18,
    )
    
    # create a Sun node (visualized by a small sphere) and mark it as a light source
    sun_node = Node('Sun', local=translate(0.0, 9.0, 2.0) @ scale(0.6, 0.6, 0.6), mesh=sphere_mesh)
    sun_node.is_light = True
    sun_node.light_color = np.array([1.0, 0.95, 0.8], dtype=np.float32)
    sun_node.light_intensity = 2.0
    # give it a bright material so it's visible
    sun_node.material = Material.from_color(shader, (1.0, 0.95, 0.8))
    root.add(sun_node)

    # dados globais da cena, camara e luz
    up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    ambient       = np.array([0.38, 0.38, 0.32], dtype=np.float32)   
    light_diffuse = np.array([1.0, 1.0, 1.0], dtype=np.float32)  


    light_dir = np.array([0.45, 0.9, 0.35], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    # skybox loader (external to scene graph)
    try:
        sky = Skybox(shader_program=skybox_shader.prog)
    except Exception:
        sky = None

    # instantiate renderer once (avoid recreating per-frame)
    renderer = Renderer()

    # don't remove this variable (it's used for deltatime calculation)
    last_time = glfw.get_time()

    # Renderização
    while not window.should_close():
        glfw.poll_events()
        
        # deltatime
        current_time = glfw.get_time()
        deltaTime = current_time - last_time
        last_time = current_time

        fbw, fbh = window.get_framebuffer_size()

        # Seleciona modo de camera — atualizar a free cam se necessário
        if cam.mode == 'free':
            update_free_camera(window, cam, deltaTime)
            cam_eye, cam_ctr = get_view_free(cam)
        else:
            cam_eye, cam_ctr = follow_cam(deltaTime)

        # create projection and view matrices
        P = perspective(math.radians(45.0), float(fbw) / float(fbh), 0.1, 100.0)
        V = lookAt(cam_eye, cam_ctr, up)

        # upload projection and view into the UBO using column-major float arrays
        proj_arr = mat_to_column_major_floats(P)
        view_arr = mat_to_column_major_floats(V)
        proj_bytes = proj_arr.nbytes

        # Normal render mode: enable back-face culling and filled polygons
        glEnable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # clear buffers
        glClearColor(0.05, 0.05, 0.25, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   
        # Draw the entire scene via the Renderer; it minimizes shader switches
        # and calls node.on_draw when present.
        def _common_setup(shader_obj, lights=None):
            # upload per-shader globals: view/proj and lighting if API available
            try:
                if hasattr(shader_obj, 'set_common'):
                    # pass collected point lights (if any) to shader wrapper
                    shader_obj.set_common(V, P, light_dir, ambient, light_diffuse, lights)
                else:
                    # fallback: try to set uView/uProj uniforms directly
                    prog = getattr(shader_obj, 'prog', None)
                    if prog is not None:
                        # loc_p = glGetUniformLocation(prog, 'uProj')
                        # if loc_p != -1:
                        #     glUniformMatrix4fv(loc_p, 1, GL_TRUE, P.astype(np.float32))
                        # loc_v = glGetUniformLocation(prog, 'uView')
                        # if loc_v != -1:
                        #     glUniformMatrix4fv(loc_v, 1, GL_TRUE, V.astype(np.float32))
                        
                        # also try to set lighting uniforms for older shader wrappers
                        loc_amb = glGetUniformLocation(prog, 'uAmbient')
                        if loc_amb != -1:
                            glUniform3fv(loc_amb, 1, ambient.astype(np.float32))
                        loc_lcol = glGetUniformLocation(prog, 'uLightColor')
                        if loc_lcol != -1:
                            glUniform3fv(loc_lcol, 1, light_diffuse.astype(np.float32))
                        # try to set point-light arrays if present
                        loc_lcount = glGetUniformLocation(prog, 'uLightCount')
                        if loc_lcount != -1 and lights is not None:
                            count = min(len(lights), 4)
                            glUniform1i(loc_lcount, count)
                            # positions
                            loc_pos0 = glGetUniformLocation(prog, 'uLightPos[0]')
                            if loc_pos0 != -1 and count > 0:
                                flat = []
                                for i in range(count): flat.extend(list(lights[i]['pos']))
                                glUniform3fv(loc_pos0, count, (np.array(flat, dtype=np.float32)))
                            loc_col0 = glGetUniformLocation(prog, 'uLightCol[0]')
                            if loc_col0 != -1 and count > 0:
                                flatc = []
                                for i in range(count): flatc.extend(list(np.array(lights[i]['col']) * lights[i]['int']))
                                glUniform3fv(loc_col0, count, (np.array(flatc, dtype=np.float32)))
            except Exception:
                # let errors surface during development if desired
                raise
        
        # draw skybox first (if any)
        view_rot = V.copy()
        view_rot[0:3, 3] = 0.0
        proj_arr = mat_to_column_major_floats(P)
        view_arr = mat_to_column_major_floats(view_rot)
        uboPV.update_subdata(0, proj_arr)                  # proj @ offset 0
        uboPV.update_subdata(proj_arr.nbytes, view_arr)    # view @ offset 64 (16 floats * 4 bytes)
        draw_skybox_loader(sky, view_rot, P)

        proj_arr = mat_to_column_major_floats(P)
        view_arr = mat_to_column_major_floats(V)
        uboPV.update_subdata(0, proj_arr)                  # proj @ offset 0
        uboPV.update_subdata(proj_arr.nbytes, view_arr)    # view @ offset 64 (16 floats * 4 bytes)

        root.update(deltaTime) # update the scene graph (animations, transforms, etc)

        renderer.render(root, None, cam_eye, light_dir, ambient, default_shader=shader, common_setup=_common_setup)

        window.swap_buffers()

    # saída e limpeza
    for m in resources:
        try:
            m.destroy()
        except Exception:
            pass
    
    shader.destroy()
    glfw.terminate()


def find_node_by_name(node, name):
    if node.name == name:
        return node
    for c in node.children:
        r = find_node_by_name(c, name)
        if r is not None:
            return r
    return None

def apply_transform_local_by_name(root, node_name, M, pre=True):
    node = find_node_by_name(root, node_name)
    if node is None:
        raise RuntimeError(f"Node '{node_name}' não encontrado.")
    node.local = (M @ node.local) if pre else (node.local @ M)
    return node





if __name__ == "__main__":
    main()
