# OpenGL com Shaders (>3.3)
# Sistema solar simples : 2 planetas + 1 lua

import sys, math, ctypes, os
import numpy as np
import glfw
from OpenGL.GL import *
import anim  # módulo separado com animadores (src/anim.py)
from carro import Car  # estado físico do carro
from camera import Camera, update_free_camera, get_view_free
from node import Node
from gfx import MeshTextured, ShaderProgram, Mesh, wrapperCreateShader
from myglm import *
from geo import *
from skybox import Skybox
from texture import *

# iniciar o OpenGL e Janela usando o GLFW
def setup_window(w=1200, h=800, title="Solar System — Grafo de cena com Flat Shading (OpenGL 3.3)"):
    if not glfw.init():
        print("Failed to initialize GLFW", file=sys.stderr); sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(w, h, title, None, None)
    if not win:
        glfw.terminate(); print("Failed to create window", file=sys.stderr); sys.exit(1)
    glfw.make_context_current(win)
    # Usar vsync para evitar o loop a correr demasiado depressa saturando a CPU
    try:
        glfw.swap_interval(1)
    except Exception:
        pass
    return win

def setup_gl_state():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

# Forma extensível de construir a cena: cria internamente os meshes
def build_scene(meshes: dict = None):
    if meshes is None:
        meshes = {}
    # materiais simples
    COL_CAR   = (0.15, 0.15, 0.18)
    COL_FLOOR = (0.20, 0.80, 0.20)
    COL_TOP   = (0.8, 0.1, 0.1)
    car_mesh = meshes.get('car', None)
    if car_mesh is None:
        raise ValueError("Car mesh not provided in mesh dictionary")
    plane_mesh = meshes.get('plane', None)
    if plane_mesh is None:
        raise ValueError("Plane mesh not provided in mesh dictionary")
    car = Node("Car")
    # Orientação do carro: comprimento ao longo do eixo X, largura ao longo de Z
    # A geometria do corpo foi orientada por índices para X, por isso não é necessário rodar +90° em Y aqui.
    # Re-posicionar: manter chão em y=0 e corpo do carro elevando pelas rodas
    car_body = Node(
        "CarBody",
        local=translate(0, 0.55, 0) @ scale(4.2, 0.9, 2.0),
        mesh=car_mesh,
        albedo=COL_CAR
    )
    floor = Node("Floor", local=translate(0, 0, 0) @ scale(50, 0.1, 50), mesh=plane_mesh, albedo=COL_FLOOR)

    root = Node("Root")
    root.add(floor, car)
    car.add(car_body)
    return root

def main():
    win = setup_window()
    setup_gl_state()

    #skybox
    #skybox = Skybox()

    #aqui só teremos um shader (carregado dos ficheiros em src/shaders)
    vs_path = os.path.join(os.path.dirname(__file__), 'shaders', 'basic.vert')
    fs_path = os.path.join(os.path.dirname(__file__), 'shaders', 'basic.frag')

    shader = ShaderProgram.from_files(vs_path, fs_path)

    floor_shader = wrapperCreateShader('floor')
    floor_shader.setInt('texture1', 0)  # textura no slot 0
    
    inter, idx = gen_uv_plane_flat(size=1.0, divisions=10)
    texture_floor = Texture(os.path.join(os.path.dirname(__file__), 'textures', 'floor.jpg'))
    plane_mesh = MeshTextured(inter, idx, texture=texture_floor)

    inter, idx = gen_uv_car_body(size=1.0)
    car_mesh = Mesh(inter, idx)
    
    mesh_dict = {'car': car_mesh, 'plane': plane_mesh}
    
    resources = [car_mesh, plane_mesh]
    root = build_scene(meshes=mesh_dict)
    
    # --- ligar controlador do carro e criar rodas (placeholders) ---
    car = find_node_by_name(root, "Car")
    if car is None:
        raise RuntimeError("Node 'Car' não encontrado. Confirma o nome no build_scene().")

    # Mesh de roda cilíndrica (unitária). O anim aplica a escala real.
    inter, idx = gen_uv_cylinder_flat(radius=1.0, half_width=0.5, slices=32)
    wheel_mesh = Mesh(inter, idx)
    resources.append(wheel_mesh)

    # Offsets aproximados (X frente, Z direita, Y cima)
    wheel_radius = 0.55
    wheel_width  = 0.30
    # Offsets com Y positivo para pousar no chão (y ~= radius)
    x_off, z_off, y_off = 2.3, 1.2, wheel_radius

    def make_wheel(name, ox, oy, oz):
        return Node(
            name,
            local=translate(ox, oy, oz) @ scale(wheel_radius, wheel_radius, wheel_width),
            mesh=wheel_mesh,
            albedo=(0.05, 0.05, 0.05)
        )

    wheels = {
        "Wheel_FL": make_wheel("Wheel_FL", +x_off, y_off, -z_off),
        "Wheel_FR": make_wheel("Wheel_FR", +x_off, y_off, +z_off),
        "Wheel_RL": make_wheel("Wheel_RL", -x_off, y_off, -z_off),
        "Wheel_RR": make_wheel("Wheel_RR", -x_off, y_off, +z_off),
    }
    car.add(*wheels.values())

    # Estado lógico do carro
    car_state = Car()

    # Criar animador único que trata chassis + rodas
    car.animator = anim.make_car_animators(
        win=win,
        car_state=car_state,
        car_node=car,
        wheel_nodes=wheels,
        translate=translate, rotate=rotate, scale=scale,
    )

    # Camera a seguir o carro (offset atrás e acima, com smoothing)
    # Camera atrás do carro (e um pouco acima). Para frente = +X, usar offset X negativo.
    follow_cam = anim.make_follow_camera(
        lambda: car.local.copy(),
        offset_local=(-12.0, 4.0, 0.0),
        look_ahead=8.0,
        lag_seconds=0.18,
    )
    # Estado da camera livre
    cam = Camera()

    # dados globais da cena, camara e luz
    up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    ambient       = np.array([0.38, 0.38, 0.32], dtype=np.float32)   
    light_diffuse = np.array([1.0, 1.0, 1.0], dtype=np.float32)  


    light_dir = np.array([0.45, 0.9, 0.35], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)



    #controlo (com glfw)
    show_debug = False
    debug_accum = 0.0

    def on_key(win, key, sc, action, mods):
        nonlocal show_debug, cam
        if action in (glfw.PRESS, glfw.REPEAT):
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key == glfw.KEY_F1 and action == glfw.PRESS:
                show_debug = not show_debug
            elif key == glfw.KEY_P and action == glfw.PRESS:
                # Toggle camera mode. If switching to free, re-seed free cam from current follow view
                new_mode = 'free' if cam.mode != 'free' else 'follow'
                if new_mode == 'free':
                    # Amostrar a posição atual da follow cam e orientar a free cam igual
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

    glfw.set_key_callback(win, on_key) #definir o call back do teclado
    t_prev = glfw.get_time()

    # Renderização
    while not glfw.window_should_close(win):
        glfw.poll_events()

        # tempos
        t_now = glfw.get_time()
        dt = max(1e-6, t_now - t_prev); t_prev = t_now

        # actualização do grafo de cena
        root.update(dt)

        # Debug opcional do estado do carro (F1 para alternar)
        if show_debug:
            debug_accum += dt
            if debug_accum >= 0.25:  # imprime a cada ~0.25s
                debug_accum = 0.0
                print(f"Car: x={car_state.x:.2f} z={car_state.z:.2f} yaw={math.degrees(car_state.yaw):.1f}° v={car_state.v:.2f} steer={math.degrees(car_state.steer):.1f}°")


        #definição das transformações até ao viewport (perspectivca e vista)
        fbw, fbh = glfw.get_framebuffer_size(win)
        glViewport(0,0,fbw,fbh)
        P  = perspective(35.0, max(fbw,1) / float(max(fbh,1)), 0.1, 100.0)

        # Seleciona modo de camera
        if cam.mode == 'free':
            update_free_camera(win, cam, dt)
            cam_eye, cam_ctr = get_view_free(cam)
        else:
            cam_eye, cam_ctr = follow_cam(dt)
        
        
        V  = lookAt(cam_eye, cam_ctr, up)
        VP = P @ V

        glClearColor(0.05, 0.05, 0.25, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shader.use()
        shader.set_common(V, P, light_dir)

        # Desenhar objectos não texturados (carro)
        root.children[1].draw(shader, np.eye(4, dtype=np.float32), None, cam_eye, light_dir, ambient)

        # Desenhar o piso com o shader texturado: ativar o programa e definir os uniforms comuns
        floor_shader.use()
        floor_shader.set_common(V, P, light_dir)
        root.children[0].draw(floor_shader, np.eye(4, dtype=np.float32), None, cam_eye, light_dir, ambient)

        #skybox.draw(V, P)

        glfw.swap_buffers(win)

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
    try:
        # Modo rápido de validação sem abrir janela:
        #   python cenario.py --validate-cube
        if len(sys.argv) > 1 and sys.argv[1] == "--validate-cube":
            inter, idx = gen_uv_cube_flat(size=3.0)
            ok = validate_cube_mesh(inter, idx, expected_size=3.0, verbose=True)
            sys.exit(0 if ok else 2)
        else:
            main()
    except Exception as e:
        print("ERRO:", e); sys.exit(1)
