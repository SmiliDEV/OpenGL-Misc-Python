# Fazemos import explicito em alguns modulos para evitar possiveis conflitos e explicitar as funcoes usadas
import math, os
import numpy as np
import glfw
from OpenGL.GL import (
    glViewport, glEnable, glCullFace, glFrontFace, glClearColor, glClear, glPolygonMode,
    glGetUniformLocation, glUniform3fv, glUniform1i,
    GL_DEPTH_TEST, GL_CULL_FACE, GL_BACK, GL_CCW, GL_FRONT_AND_BACK, GL_FILL,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
)
import anim
from carro import Car
from camera import Camera, update_free_camera, get_view_free
from node import Node
from math3d import perspective, lookAt, mat_to_column_major_floats, translate, scale, rotate
from geo import gen_uv_plane_flat, gen_uv_cylinder_flat, gen_uv_sphere_flat, gen_uv_cube_flat
from skybox import Skybox, draw_skybox_loader
from renderer import Renderer
from window import Window
from glib import Shader, ShaderProgram, wrapperCreateShader, UniformBuffer, Texture, Mesh, MeshTextured
from material import Material
import random
from file import get_content_of_file_project
from obj import load_obj, load_obj_multi

width = 800
height = 600

# Ajustar posição inicial da câmara (Free Cam) para perto do carro (-15, 0, 0)
cam = Camera(pos=np.array([-28.0, 5.0, 0.0], dtype=np.float32), yaw=0.0)
last_x, last_y = width / 2, height / 2
first_mouse = True
follow_cam = None
follow_cam2 = None

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

    xoffset *= cam.mouse_sens
    yoffset *= cam.mouse_sens
    cam.yaw += xoffset
    cam.pitch += yoffset
    cam.pitch = max(math.radians(-85.0), min(math.radians(85.0), cam.pitch))

# Teclas e comportamentos
def key_callback(win, key, sc, action, mods):
    global follow_cam, cam, first_mouse
    if action in (glfw.PRESS, glfw.REPEAT):
        # terminar programa
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        # alternar fullscreen
        elif key == glfw.KEY_Z and action == glfw.PRESS:
            window = glfw.get_window_user_pointer(win)
            if window:
                window.toggle_fullscreen()
                first_mouse = True
        # alternar modos de camera
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            # Ciclo: free -> follow -> inside -> free...
            global follow_cam2
            modes = ['free', 'follow', 'inside']
            idx = modes.index(cam.mode) if cam.mode in modes else 0
            new_mode = modes[(idx + 1) % len(modes)]

            if new_mode == 'free':
                eye, ctr = None, None
                if cam.mode == 'follow' and follow_cam is not None:
                    eye, ctr = follow_cam(0.0)
                elif cam.mode == 'inside' and follow_cam2 is not None:
                    eye, ctr = follow_cam2(0.0)
                if eye is None and follow_cam is not None:
                    eye, ctr = follow_cam(0.0)
                if eye is not None and ctr is not None:
                    dirv = ctr - eye
                    n = np.linalg.norm(dirv)
                    if n > 1e-6:
                        dirv = dirv / n
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

def build_scene(window, meshes: dict, materials: dict):
    # -------------------------------------------------------------------------
    # 1. Inicializar Meshes
    # -------------------------------------------------------------------------
    car_mesh = meshes['car']
    grass_mesh = meshes['grass']
    ground_mesh = meshes['ground']
    road_mesh = meshes['road']
    wheel_left = meshes.get('wheel_left')
    wheel_right = meshes.get('wheel_right')
    wheel_rear = meshes.get('wheel_rear')
    wheel_single = meshes.get('wheel_rear') # Fallback
    door_left = meshes.get('door_left')
    door_right = meshes.get('door_right')
    steering_mesh = meshes.get('steering_wheel')
    pole_mesh = meshes.get('pole')
    light_pole_mesh = meshes.get('esfera')
    aviao_mesh = meshes.get('aviao')
    sun_mesh = meshes.get('esfera')
    go_mesh = meshes.get('go')
    gd_mesh = meshes.get('gd')
    cafe_mesh = meshes.get('cafe')
    tree_parts = meshes.get('tree_parts', {})

    # -------------------------------------------------------------------------
    # 2. Criar Nodes
    # -------------------------------------------------------------------------
    root = Node('Root')

    # --- Ambiente (Chão, Estrada, Relva) ---
    floor_node = Node('Floor', local=translate(0, 0, 0) @ scale(500, 0.1, 500), mesh=grass_mesh)
    floor_node.material = materials['floor']

    road_x = Node('RoadX', local=translate(0, 0.02, 0) @ scale(500, 1.0, 12), mesh=road_mesh)
    road_x.material = materials['road']
    
    road_z = Node('RoadZ', local=translate(0, 0.02, 0) @ rotate(math.radians(90), (0, 1, 0)) @ scale(500, 1.0, 12), mesh=road_mesh)
    road_z.material = materials['road']

    ground1 = Node('Ground1', local=translate(30, 0.02,-30) @ scale(30, 0.1,30), mesh=ground_mesh)
    ground1.material = materials['ground']
    
    ground2 = Node('Ground2', local=translate(30, 0.02,30) @ scale(30, 0.1,30), mesh=ground_mesh)
    ground2.material = materials['ground']

    # --- Carro ---
    x_car, y_car, z_car = -15.0, 0.0, 0.0
    car_node = Node('Car', local=translate(x_car, y_car, z_car))
    
    car_body = Node('CarBody', local=translate(0, 2.00, 0) @ scale(10.0, 10.0, 10.0) @ rotate(math.radians(90), (0, 1, 0)), mesh=car_mesh)
    car_body.material = materials['car']

    # Portas
    door_node_l = Node('Door_L', local=translate(-0.17, -0.06, -0.0375) @ scale(0.3, 0.3, 0.3), mesh=door_left)
    door_node_l.material = materials['car']
    door_node_l.animator = anim.make_door_anim(door_node_l, window, glfw.KEY_K, open_angle_deg=75.0, axis=(0,1,0), speed=10.0)

    door_node_r = Node('Door_R', local=translate(0.17, -0.07, 0.0375) @ scale(0.3, 0.3, 0.3), mesh=door_right)
    door_node_r.material = materials['car']
    door_node_r.animator = anim.make_door_anim(door_node_r, window, glfw.KEY_L, open_angle_deg=-75.0, axis=(0,1,0), speed=10.0)

    # Volante
    sw_node = Node('SteeringWheel', local=translate(0.1, -0.01, 0.07) @ rotate(0.0, (1,0,0)) @ scale(0.10, 0.10, 0.10), mesh=steering_mesh)
    sw_node.material = materials.get('steering', materials.get('wheel', materials['car']))

    # Rodas
    def make_wheel(name):
        if name.endswith('FL'): mesh_for_wheel = wheel_left or wheel_rear or wheel_single
        elif name.endswith('FR'): mesh_for_wheel = wheel_right or wheel_rear or wheel_single
        elif name.endswith('RL'): mesh_for_wheel = wheel_left or wheel_rear or wheel_single
        elif name.endswith('RR'): mesh_for_wheel = wheel_right or wheel_rear or wheel_single
        else: mesh_for_wheel = wheel_rear or wheel_single
        
        n = Node(name, mesh=mesh_for_wheel)
        n.material = materials['wheel']
        return n

    wheels = {
        'Wheel_FL': make_wheel('Wheel_FL'),
        'Wheel_FR': make_wheel('Wheel_FR'),
        'Wheel_RL': make_wheel('Wheel_RL'),
        'Wheel_RR': make_wheel('Wheel_RR'),
    }

    # Animador do Carro
    car_state = Car(x=x_car, z=z_car)
    car_node.animator = anim.make_car_animators(
        win=window,
        car_state=car_state,
        car_node=car_node,
        wheel_nodes=wheels,
        translate=translate, rotate=rotate, scale=scale,
    )

    # --- Poste de Luz ---
    pole_node = Node('Pole', local=translate(12.0, 0.0, -6.0) @ scale(2.0, 2.0, 2.0) @ rotate(90, (0, 1, 0)), mesh=pole_mesh)
    pole_node.material = materials.get('pole', None)
    
    bulb_local_pos = translate(-0.46, 2.57, 0.0) @ scale(0.14, 0.14, 0.14)
    light_pole_node = Node('LightPole', local=bulb_local_pos, mesh=light_pole_mesh)
    light_pole_node.is_light = True
    light_pole_node.light_color = np.array([1.0, 1.0, 0.9], dtype=np.float32)
    light_pole_node.light_intensity = 1.3
    light_pole_node.material = materials.get('light_pole', None)
    pole_node.add(light_pole_node)

    # --- Avião ---
    aviao_node = Node('Aviao', mesh=aviao_mesh)
    aviao_node.animator = anim.make_plane_animator(
        node=aviao_node,
        center_pos=(20.0, 0.0, 20.0),
        axis=(0.2, 1.0, 0.2),
        radius=20.0,
        speed=1.5,
        height=20.0,
        scale_factor=5.0,
        spin_speed=1.0 
    )
    aviao_node.material = materials.get('aviao', None)

    # --- Sol ---
    sun_node = Node('Sun', local=translate(0.0, 9.0, 2.0) @ scale(0.6, 0.6, 0.6), mesh=sun_mesh)
    sun_node.is_light = True
    sun_node.light_color = np.array([1.0, 0.95, 0.8], dtype=np.float32)
    sun_node.light_intensity = 2.0
    sun_node.material = materials.get('sun', None)
    sun_node.animator = anim.make_sun_animator(sun_node, translate=translate, rotate=rotate, scale=scale, orbit_radius=50.0, orbit_period=80.0, tilt_angle_deg=23.5)

    # --- Garagem ---
    go_node = Node('GO', local=translate(30.0, 2.3, 30.0) @ scale(15.0, 14.0, 15.0) @ rotate(math.radians(270), (0, 1, 0)), mesh=go_mesh)
    go_node.material = materials['go']
    
    gd_node = Node('GD', local=translate(0, 0.0, 0.475) @ scale(0.7, 0.7, 0.7), mesh=gd_mesh)
    gd_node.material = materials['gd']
    gd_node.animator = anim.make_garage_door_animator(gd_node, win=window, key=glfw.KEY_F, open_offset_y=-0.4, speed=8.0)
    go_node.add(gd_node)

    # --- Café ---
    cafe_scale = 1.5
    cafe_node = Node('Cafe', local=translate(30, 2.7+ 0.5 * cafe_scale, -30) @ scale(cafe_scale, cafe_scale, cafe_scale), mesh=cafe_mesh)
    cafe_node.material = materials.get('cafe_textura', materials.get('cafe_padrao'))

    # -------------------------------------------------------------------------
    # 3. Construir Hierarquia (Adicionar Nodes)
    # -------------------------------------------------------------------------
    
    # Montar Carro
    car_body.add(door_node_l)
    car_body.add(door_node_r)
    car_body.add(sw_node)
    car_body.add(*wheels.values())
    car_node.add(car_body)

    # Adicionar tudo à Root
    root.add(floor_node)
    root.add(road_x, road_z)
    root.add(ground1, ground2)
    root.add(car_node)
    root.add(pole_node)
    root.add(aviao_node)
    root.add(sun_node)
    root.add(go_node)
    root.add(cafe_node)

    # Funções auxiliares que já adicionam à root internamente
    add_random_trees(tree_parts=tree_parts, min_dist=8.0, num_trees=15, root=root, materials=materials)
    add_park(root, meshes, materials)

    return root

def main():
    window = Window(800, 600, "Carro na rua — Grafo de cena com Flat Shading (OpenGL 3.3)", fullscreen=False)
    glfw.set_cursor_pos_callback(window.win, cursor_pos_callback)
    glfw.set_key_callback(window.win, key_callback)
    glfw.set_framebuffer_size_callback(window.win, framebuffer_size_callback)
    setup_gl_state()

    # UBOs
    uboPV = UniformBuffer(np.zeros(2 * 16, dtype=np.float32), binding_point=0)  # espaço para 2 matrizes 4x4 (P e V)

    # Shaders_SC
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

    # criar meshes_SC
    # relva
    inter, idx = gen_uv_plane_flat(size=1.0, divisions=10)
    texture_grass = Texture(os.path.join(os.path.dirname(__file__), 'textures', 'grass.jpg'))
    grass_mesh = MeshTextured(inter, idx, texture=texture_grass)
    # estrada
    texture_road = Texture(os.path.join(os.path.dirname(__file__), 'textures', 'road.jpg'))
    texture_ground = Texture(os.path.join(os.path.dirname(__file__), 'textures', 'ground.jpg'))
    
    # Tentar carregar a textura do edificio se existir

    ground_mesh = MeshTextured(inter, idx, texture=texture_ground)
    # textura estrada usa mesma malha do plano
    # criar mesh texturizada para estrada
    inter[6::8] *= 40.0 
    road_mesh = MeshTextured(inter, idx, texture=texture_road)
    # Mesh do carro 
    car_obj_path = os.path.join(os.path.dirname(__file__), 'objects', 'car.obj')
    car_mesh = load_obj(car_obj_path, normalize=True, target_max=1.0)

    # Mesh de roda cilíndrica (unitária). O anim aplica a escala real.
    inter_w, idx_w = gen_uv_cylinder_flat(radius=1.0, half_width=0.5, slices=32)
    # create reusable cylinder mesh used for rear wheels and as fallback
    wheel_cyl_mesh = Mesh(inter_w, idx_w)

    # Load wheel meshes: try front right/left specific objects (rfe/rfd). Rear wheels will use the cylinder.
    wheel_left_path = os.path.join(os.path.dirname(__file__), 'objects', 'rfe.obj')
    wheel_right_path = os.path.join(os.path.dirname(__file__), 'objects', 'rfd.obj')
    wheel_mesh_left =  load_obj(wheel_left_path, normalize=True, target_max=1.0)
    wheel_mesh_right = load_obj(wheel_right_path, normalize=True, target_max=1.0)
    # Portas 
    doorl_path = os.path.join(os.path.dirname(__file__), 'objects', 'de.obj')
    doord_path = os.path.join(os.path.dirname(__file__), 'objects', 'dd.obj')
    doorl_mesh = load_obj(doorl_path, normalize=True, target_max=1.0)
    doord_mesh = load_obj(doord_path, normalize=True, target_max=1.0)
    steering_path = os.path.join(os.path.dirname(__file__), 'objects', 'wheel.obj')
    steering_mesh = load_obj(steering_path, normalize=True, target_max=1.0)
    
    # Mesh esférica genérica (raio 1.0) para ser usada em tudo (Sol, Luz, Pedras)
    inter_s, idx_s = gen_uv_sphere_flat(radius=1.0, stacks=12, slices=24)
    sphere_mesh = Mesh(inter_s, idx_s)

    pole_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'objects', 'pole2.obj'))
    pole_mesh = load_obj(pole_path, normalize=False)

    aviao_path = os.path.join(os.path.dirname(__file__), 'objects', 'aviao.obj')
    aviao_mesh = load_obj(aviao_path, normalize=True, target_max=1.0)


    go_mesh = None
    gd_mesh = None
    go_path = os.path.join(os.path.dirname(__file__), 'objects', 'go.obj')
    gd_path = os.path.join(os.path.dirname(__file__), 'objects', 'gd.obj')
    go_mesh = load_obj(go_path, normalize=True, target_max=1.0)
    gd_mesh = load_obj(gd_path, normalize=True, target_max=1.0)

    # Tree
    tree_path = os.path.join(os.path.dirname(__file__), 'objects', 'tree.obj')
    tree_parts = load_obj_multi(tree_path, normalize=True, target_max=1.0)

    # Cafe
    cafe_path = os.path.join(os.path.dirname(__file__), 'objects', 'building.obj')
    cafe_mesh = load_obj(cafe_path, normalize=True, target_max=10.0)

    # Primitivas para o parque e armazém
    inter_cube, idx_cube = gen_uv_cube_flat(size=1.0)
    cube_mesh = Mesh(inter_cube, idx_cube)
    
    # Skybox é tratada de forma separada no renderer
    mesh_dict = {
        'car': car_mesh,
        'grass': grass_mesh,
        'aviao': aviao_mesh,
        'road': road_mesh,
        'ground': ground_mesh,
        'tree_parts': tree_parts,
        'cafe': cafe_mesh,
        'door_left': doorl_mesh,
        'door_right': doord_mesh,
        'wheel_left': wheel_mesh_left,
        'wheel_right': wheel_mesh_right,
        'wheel_rear': wheel_cyl_mesh,
        'steering_wheel': steering_mesh,
        'esfera': sphere_mesh,
        'cubo': cube_mesh,
        'pole': pole_mesh,
        'go': go_mesh,
        'gd': gd_mesh,
    }
    resources = [
        car_mesh, grass_mesh, ground_mesh, road_mesh, 
        doorl_mesh, doord_mesh, 
        wheel_mesh_left, wheel_mesh_right, wheel_cyl_mesh, 
        steering_mesh, 
        sphere_mesh, pole_mesh, 
        go_mesh, gd_mesh,
        cube_mesh,
        cafe_mesh,
        aviao_mesh
    ]
    resources.extend(tree_parts.values())
    
    # Materiais
    COL_CAR = (0.8, 0.1, 0.1)
    COL_WHEEL = (0.1, 0.1, 0.1)
    materials = {}
    # carro
    materials['car'] = Material.from_color(shader, COL_CAR)
    materials['car'].shininess = 100.0
    materials['car'].specular_color = (0.8, 0.8, 0.8)
    # aviao 
    materials['aviao'] = Material.from_color(shader, (0.9, 0.9, 0.9))
    materials['aviao'].shininess = 50.0
    materials['aviao'].specular_color = (0.5, 0.5, 0.5)
    # relva
    materials['floor'] = Material.from_texture(floor_shader, texture_grass)
    materials['floor'].specular_color = (0.0,0.0,0.0)
    # estrada
    materials['road'] = Material.from_texture(floor_shader, texture_road)
    materials['road'].specular_color = (0.0,0.0,0.0)

    #chao 
    materials['ground'] = Material.from_texture(floor_shader, texture_ground)
    materials['ground'].specular_color = (0.0,0.0,0.0)
    materials['ground'].shininess = 3.0
    materials['ground'].diffuse_scale = 0.6


    # rodas
    materials['wheel'] = Material.from_color(shader, COL_WHEEL)
    materials['wheel'].shininess = 10.0
    materials['wheel'].specular_color = (0.3, 0.3, 0.3)
    # Volante: Preto
    materials['steering'] = Material(shader, albedo_color=(0.05, 0.05, 0.05), shininess=10.0, specular_color=(0.1, 0.1, 0.1))
    # Poste de luz
    materials['pole'] = Material.from_color(shader, (0.0, 0.0, 0.0))
    materials['pole'].specular_color = (0.1, 0.1, 0.1)
    # Sol
    # O emissive faz o sol ter apenas a cor dele 
    materials['sun'] = Material.from_color(shader, (1.0, 0.95, 0.8))
    materials['sun'].emissive = True    
    # Luz do poste de luz
    materials['light_pole'] = Material.from_color(shader, (1.0, 0.95, 0.8))
    materials['light_pole'].emissive = True
    # Garagem
    materials['go'] = Material.from_color(shader, (0.2, 0.2, 0.2))
    materials['go'].shininess = 3.0
    materials['go'].specular_color = (0.0,0.0,0.0)
    # Porta da garagem
    materials['gd'] = Material.from_color(shader, (0.5, 0.5, 0.6))
    materials['gd'].shininess = 32.0
    materials['gd'].specular_color = (0.2, 0.2, 0.2)
    # Portas do carro
    materials['door'] = Material.from_color(shader, COL_CAR)
    materials['door'].shininess = 100.0
    materials['door'].specular_color = (0.8, 0.8, 0.8)
    
    # Arvore
    materials['bark'] = Material.from_color(shader, (0.4, 0.25, 0.1)) # Castanho
    materials['bark'].shininess = 5.0
    materials['bark'].specular_color = (0.0, 0.0, 0.0)

    materials['bf_wood'] = Material.from_color(shader, (0.1, 0.4, 0.1)) # Verde escuro
    materials['bf_wood'].shininess = 1.0
    materials['bf_wood'].specular_color = (0.0, 0.0, 0.0)

    # Materiais para o parque  
    materials['pedra'] = Material.from_color(shader, (0.5, 0.5, 0.55)) # Cinzento
    materials['madeira_banco'] = Material.from_color(shader, (0.6, 0.4, 0.2)) # Madeira clara
    materials['cafe_textura'] = Material.from_color(shader, (0.7, 0.5, 0.3)) # Castanho claro
    materials['cafe_textura'].specular_color = (0.1, 0.1, 0.1)
    materials['cafe_textura'].shininess = 10.0
    root = build_scene(window, meshes=mesh_dict, materials=materials)

    # Necessario para a camara
    car_node = root.find('Car')

    # Camera a seguir o carro (offset atrás e acima, com smoothing)
    # Camera atrás do carro (e um pouco acima). Para frente = +X, usar offset X negativo.
    global follow_cam
    follow_cam = anim.make_follow_camera(
        lambda: car_node.local.copy(),
        offset_local=(-12.0, 4.0, 0.0),
        look_ahead=8.0,
        lag_seconds=0.18,
    )

    # Orbital camera that circles the car — user can tweak with keys
    global follow_cam2
    follow_cam2 = anim.make_follow_camera_2(
        lambda: car_node.local.copy(),
        offset_local=(-1.1, 3.0, -0.75),
        look_ahead=8.0,
    )



    # dados globais da cena, camara e luz
    up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    ambient       = np.array([0.38, 0.38, 0.32], dtype=np.float32)   
    light_diffuse = np.array([1.0, 1.0, 1.0], dtype=np.float32)  


    light_dir = np.array([0.45, 0.9, 0.35], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    # Configuração da Skybox
    # Mudar o nome da pasta aqui para trocar de skybox (ex: 'skybox', 'skybox_night', etc.)
    # Opcoes: 'skybox' (original), 'fantasy_day', 'fantasy_sunless', 'fantasy_night'
    SKYBOX_COLLECTION = 'fantasy_sunless' 
    skybox_path = os.path.join(os.path.dirname(__file__), 'textures', SKYBOX_COLLECTION)
    
    sky = Skybox(shader_program=skybox_shader.prog, texture_folder=skybox_path)

    renderer = Renderer()

    last_time = glfw.get_time()

    # Renderização
    while not window.should_close():
        glfw.poll_events()
        
        current_time = glfw.get_time()
        deltaTime = current_time - last_time
        last_time = current_time
        fbw, fbh = window.get_framebuffer_size()

        # Seleciona modo de camera — atualizar a free cam se necessário
        if cam.mode == 'free':
            update_free_camera(window, cam, deltaTime)
            cam_eye, cam_ctr = get_view_free(cam)
        elif cam.mode == 'follow':
            cam_eye, cam_ctr = follow_cam(deltaTime)
        elif cam.mode == 'inside':
            cam_eye, cam_ctr = follow_cam2(deltaTime)
        else:
            cam_eye, cam_ctr = follow_cam(deltaTime)

        P = perspective(math.radians(45.0), float(fbw) / float(fbh), 0.1, 100.0)
        V = lookAt(cam_eye, cam_ctr, up)

        proj_arr = mat_to_column_major_floats(P)
        view_arr = mat_to_column_major_floats(V)

        glEnable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glClearColor(0.05, 0.05, 0.25, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   
        def _common_setup(shader_obj, lights=None):
            # pass collected point lights (if any) to shader wrapper
            shader_obj.set_common(V, P, light_dir, ambient, light_diffuse, lights)
        
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

    # cleanup
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


def add_random_trees(root,materials,min_dist, num_trees,tree_parts):
    tree_positions = []
    for i in range(num_trees):
        attempts = 0
        found_pos = False
        tx, tz = 0, 0
        # mudança de while true pra attempts pra evitar um caso rarissimo de loop eterno
        while attempts < 100:
            attempts += 1
            tx = random.uniform(-20, 50)
            tz = random.uniform(-50, 50)
            
            # Nao meter na estrada
            if abs(tx) < 7 or abs(tz) < 7: 
                continue
            # Nao meter na garagem
            if (5 < tx < 45) and (20 < tz < 40):
                continue
            # Nao meter no chao
            if (15 < tx < 45) and ((-45 < tz < -15) or (15 < tz < 45)):
                continue
            
            # Ver se tao perto de outras arvores
            too_close = False
            for (ex, ez) in tree_positions:
                if math.sqrt((tx - ex)**2 + (tz - ez)**2) < min_dist:
                    too_close = True
                    break
            if too_close:
                continue
                
            found_pos = True
            tree_positions.append((tx, tz))
            break
        
        if not found_pos:
            continue
                
        s = random.uniform(1.5, 2.5)
        r = random.uniform(0, 360)
            
        t_node = Node(f'Tree_{i}', local=translate(tx, 1.5 * s, tz) @ rotate(r, (0, 1, 0)) @ scale(2*s, 3*s, 2*s))
            
        for mtl_name, mesh_part in tree_parts.items():
            part_node = Node(f'Tree_{i}_{mtl_name}', mesh=mesh_part)
            part_node.material = materials.get(mtl_name, materials.get('bark'))
            t_node.add(part_node)
                
        root.add(t_node)



# tira um pouco da poluicao do build_scene
def add_park(root, meshes, materials):
    sphere_mesh = meshes.get('esfera')
    cube_mesh = meshes.get('cubo')

    park_node = Node('ParkElements', local=translate(30, 0, -30))

    # Pedras (espalhadas mais longe)
    for i in range(5):
        angle = random.uniform(0, 360)
        dist = random.uniform(10, 14)
        px = dist * math.sin(math.radians(angle))
        pz = dist * math.cos(math.radians(angle))
        s = random.uniform(0.3, 0.7)
        pedra = Node(f'Pedra_{i}', local=translate(px, s*0.4, pz) @ scale(s, s*0.6, s), mesh=sphere_mesh)
        pedra.material = materials['pedra']
        park_node.add(pedra)
    # Bancos (simples: 1 corpo + 2 pernas)
    def criar_banco(name, x, z, rot):
        banco = Node(name, local=translate(x, 0, z) @ rotate(math.radians(rot), (0,1,0)) @ scale(1.5, 1.5, 1.5))
        assento = Node('Assento', local=translate(0, 0.6, 0) @ scale(2.0, 0.1, 0.6), mesh=cube_mesh)
        perna1 = Node('Perna1', local=translate(-0.8, 0.3, 0) @ scale(0.2, 0.6, 0.5), mesh=cube_mesh)
        perna2 = Node('Perna2', local=translate(0.8, 0.3, 0) @ scale(0.2, 0.6, 0.5), mesh=cube_mesh)
        assento.material = materials['madeira_banco']
        perna1.material = materials['madeira_banco']
        perna2.material = materials['madeira_banco']
        banco.add(assento, perna1, perna2)
        return banco

    # Bancos à volta do edifício, virados para ele
    dist_banco = 10.0
    park_node.add(criar_banco('Banco1', 0, dist_banco, 180))   # Frente (Z+), virado para trás
    park_node.add(criar_banco('Banco2', 0, -dist_banco, 0))    # Trás (Z-), virado para frente
    park_node.add(criar_banco('Banco3', -dist_banco, 0, -90))  # Esquerda (X-), virado para direita
    park_node.add(criar_banco('Banco4', dist_banco, 0, 90))    # Direita (X+), virado para esquerda
    
    root.add(park_node)


if __name__ == "__main__":
    main()
