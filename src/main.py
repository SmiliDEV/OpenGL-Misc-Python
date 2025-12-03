import math, os
import numpy as np
import glfw
from OpenGL.GL import *
import anim
from carro import Car
from camera import Camera, update_free_camera, get_view_free
from node import Node
from math3d import *
from geo import *
from skybox import Skybox, draw_skybox_loader
from renderer import Renderer
from window import Window
from glib import *
from material import Material
from file import *
from obj import *

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

def key_callback(win, key, sc, action, mods):
    global follow_cam, cam
    if action in (glfw.PRESS, glfw.REPEAT):
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            # Cycle camera modes: free -> follow -> orbit -> free
            global follow_cam2
            modes = ['free', 'follow', 'orbit']
            idx = modes.index(cam.mode) if cam.mode in modes else 0
            new_mode = modes[(idx + 1) % len(modes)]

            # If switching to free, seed the free camera from the previous mode's view
            if new_mode == 'free':
                eye, ctr = None, None
                if cam.mode == 'follow' and follow_cam is not None:
                    eye, ctr = follow_cam(0.0)
                elif cam.mode == 'orbit' and follow_cam2 is not None:
                    eye, ctr = follow_cam2(0.0)
                # fall back to follow_cam for initialization if no other source
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
    """Constructs scene graph from provided meshes and materials.

    `materials` must be a dict mapping keys like 'car','wheel','floor','pole'
    to `Material` instances. The function will attach those materials to
    corresponding nodes. Returns the scene `root` Node.
    """
    # ensure required meshes exist
    if 'car' not in meshes or 'plane' not in meshes:
        raise ValueError("meshes must include at least 'car' and 'plane'")

    car_mesh = meshes.get('car')
    plane_mesh = meshes.get('plane')
    # wheels can be provided as either a single 'wheel' mesh or separate 'wheel_left'/'wheel_right'
    wheel_single = meshes.get('wheel', None)
    wheel_left = meshes.get('wheel_left', None)
    wheel_right = meshes.get('wheel_right', None)
    wheel_rear = meshes.get('wheel_rear', None)
    if wheel_left is None and wheel_right is None and wheel_single is not None:
        wheel_left = wheel_right = wheel_single

    root = Node('Root')

    # Floor 
    floor_node = Node('Floor', local=translate(0, 0, 0) @ scale(500, 0.1, 500), mesh=plane_mesh)
    floor_node.material = materials['floor']

    # Car and body
    x_car= -15.0
    y_car =  0.0 
    z_car= 0.0
    car_node = Node('Car', local=translate(x_car, y_car, z_car))
    car_body = Node('CarBody', local=translate(0, 2.00, 0) @ scale(10.0, 10.0, 10.0) @ rotate(math.radians(90), (0, 1, 0)), mesh=car_mesh)
    car_body.material = materials['car']

    # optional left/right door meshes: attach as children of car_body so they inherit body transforms
    door_left = meshes.get('door_left', None)
    door_right = meshes.get('door_right', None)
    door_node_l = Node('Door_L', local=translate(-0.17, -0.06, -0.0375) @ scale(0.3, 0.3, 0.3), mesh=door_left)
    door_node_l.material = materials['door']
    door_node_l.material = materials['car']
    door_node_l.animator = anim.make_door_anim(door_node_l, window, glfw.KEY_K, open_angle_deg=75.0, axis=(0,1,0), speed=10.0)
    car_body.add(door_node_l)

    door_node_r = Node('Door_R', local=translate(0.17, -0.07, 0.0375) @ scale(0.3, 0.3, 0.3), mesh=door_right)
    door_node_r.material = materials['door']
    door_node_r.material = materials['car']
    door_node_r.animator = anim.make_door_anim(door_node_r, window, glfw.KEY_L, open_angle_deg=-75.0, axis=(0,1,0), speed=10.0)
    car_body.add(door_node_r)

    # roda de direção
    steering_mesh = meshes.get('steering_wheel', None)
    sw_node = Node('SteeringWheel', local=translate(0.1, -0.01, 0.07) @ rotate(0.0, (1,0,0)) @ scale(0.10, 0.10, 0.10), mesh=steering_mesh)
    sw_node.material = materials.get('steering', materials.get('wheel', materials['car']))
    car_body.add(sw_node)

    # Wheels
    wheels = {}
    if (wheel_left is not None) or (wheel_right is not None) or (wheel_rear is not None) or (wheel_single is not None):
        
        def make_wheel(name):
            if name.endswith('FL'):
                mesh_for_wheel = wheel_left if wheel_left is not None else (wheel_rear or wheel_single)
            elif name.endswith('FR'):
                mesh_for_wheel = wheel_right if wheel_right is not None else (wheel_rear or wheel_single)
            elif name.endswith('RL'):
                mesh_for_wheel = wheel_left if wheel_left is not None else (wheel_rear or wheel_single)
            elif name.endswith('RR'):
                mesh_for_wheel = wheel_right if wheel_right is not None else (wheel_rear or wheel_single)
            else:
                mesh_for_wheel = wheel_rear if wheel_rear is not None else wheel_single
            
            # create wheel node with no initial transform — animator will set per-frame transforms
            n = Node(name, mesh=mesh_for_wheel)
            
            n.material = materials['wheel']
            return n

        wheels = {
            'Wheel_FL': make_wheel('Wheel_FL'),
            'Wheel_FR': make_wheel('Wheel_FR'),
            'Wheel_RL': make_wheel('Wheel_RL'),
            'Wheel_RR': make_wheel('Wheel_RR'),
        }
        
        car_body.add(*wheels.values())

    car_node.add(car_body)

    # Estado lógico do carro
    car_state = Car(x=x_car, z=z_car)

    # Criar animador único que trata chassis + rodas
    # attach the car animator to the Car node
    car_node.animator = anim.make_car_animators(
        win=window,
        car_state=car_state,
        car_node=car_node,
        wheel_nodes=wheels,
        translate=translate, rotate=rotate, scale=scale,
    )

    root.add(floor_node, car_node)
    pole_mesh = meshes.get('pole', None)
    if pole_mesh is not None:
        pole_node = Node('Pole', local=translate(12.0, 0.0, -6.0) @ scale(2.0, 2.0, 2.0) @ rotate(90, (0, 1, 0)), mesh=pole_mesh)
        pole_node.material = materials.get('pole', None)
        root.add(pole_node)


    light_pole_mesh = meshes.get('light_pole', None)

    bulb_local_pos = translate(-0.46, 2.57, 0.0) 
    light_pole_node = Node('LightPole', local=bulb_local_pos, mesh=light_pole_mesh)
    light_pole_node.is_light = True
    light_pole_node.light_color = np.array([1.0, 1.0, 0.9], dtype=np.float32)
    light_pole_node.light_intensity = 1.1
    light_pole_node.material = materials.get('light_pole', None)

    pole_node.add(light_pole_node)
   


    sun_mesh = meshes.get('sun', None)
    sun_node = Node('Sun', local=translate(0.0, 9.0, 2.0), mesh=sun_mesh)
    sun_node.is_light = True
    sun_node.light_color = np.array([1.0, 0.95, 0.8], dtype=np.float32)
    sun_node.light_intensity = 1.3
    sun_node.material = materials.get('sun', None)
    sun_node.animator = anim.make_sun_animator(sun_node, translate=translate, rotate=rotate, scale=scale, orbit_radius=30.0, orbit_period=80.0, tilt_angle_deg=23.5)
    root.add(sun_node)


    go_mesh = meshes.get('go', None)
    gd_mesh = meshes.get('gd', None)
    go_node = Node('GO', local=translate(-15.0, 2.3, 0.0) @ scale(10.0, 10.0, 10.0) @ rotate(math.radians(90), (0, 1, 0)), mesh=go_mesh)
    go_node.material = materials['go']
    gd_node = Node('GD', local=translate(0, 0.0, 0.475) @ scale(0.7, 0.7, 0.7), mesh=gd_mesh)
    gd_node.material = materials['gd']
    gd_node.animator = anim.make_garage_door_animator(gd_node, win=window, key=glfw.KEY_F, open_offset_y=-0.4, speed=8.0)
    
    go_node.add(gd_node)
    root.add(go_node)

    return root

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
    # Para o mini-sol
    inter_s, idx_s = gen_uv_sphere_flat(radius=0.6, stacks=12, slices=24)
    sun_mesh = Mesh(inter_s, idx_s)

    inter_s, idx_s = gen_uv_sphere_flat(radius=0.14, stacks=12, slices=24)
    idx_s = idx_s[::-1] # Invert indices to fix winding order (CW -> CCW)
    light_pole_mesh = Mesh(inter_s, idx_s)

    pole_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'objects', 'pole2.obj'))
    pole_mesh = load_obj(pole_path, normalize=False)


    go_mesh = None
    gd_mesh = None
    go_path = os.path.join(os.path.dirname(__file__), 'objects', 'go.obj')
    gd_path = os.path.join(os.path.dirname(__file__), 'objects', 'gd.obj')
    go_mesh = load_obj(go_path, normalize=True, target_max=1.0)
    gd_mesh = load_obj(gd_path, normalize=True, target_max=1.0)

    # Skybox é tratada de forma separada no renderer
    mesh_dict = {
        'car': car_mesh,
        'plane': plane_mesh,
        'door_left': doorl_mesh,
        'door_right': doord_mesh,
        'wheel_left': wheel_mesh_left,
        'wheel_right': wheel_mesh_right,
        'wheel_rear': wheel_cyl_mesh,
        'steering_wheel': steering_mesh,
        'sun': sun_mesh,
        'light_pole': light_pole_mesh,
        'pole': pole_mesh,
        'go': go_mesh,
        'gd': gd_mesh,
    }
    resources = [car_mesh, plane_mesh, wheel_mesh_left, wheel_mesh_right, wheel_cyl_mesh, sun_mesh,light_pole_mesh ,pole_mesh]
    if steering_mesh is not None:
        resources.append(steering_mesh)
    # add loaded door meshes to resources (keeping insertion order)
    if doorl_mesh is not None:
        resources.insert(2, doorl_mesh)
    if doord_mesh is not None:
        resources.insert(3, doord_mesh)
    
    # create materials up-front and pass them to build_scene
    COL_CAR = (0.8, 0.1, 0.1)
    COL_WHEEL = (0.1, 0.1, 0.1)
    materials = {}
    materials['car'] = Material.from_color(shader, COL_CAR)
    materials['floor'] = Material.from_texture(floor_shader, texture_floor)
    materials['wheel'] = Material.from_color(shader, COL_WHEEL)
    # Volante: Preto mate (pouco brilho)
    materials['steering'] = Material(shader, albedo_color=(0.05, 0.05, 0.05), shininess=10.0, specular_color=(0.1, 0.1, 0.1))
    materials['pole'] = Material.from_color(shader, (0.0, 0.0, 0.0))
    materials['sun'] = Material.from_color(shader, (1.0, 0.95, 0.8))
    materials['light_pole'] = Material.from_color(shader, (1.0, 0.8, 0.0))
    materials['go'] = Material.from_color(shader, (0.6, 0.6, 0.6))
    materials['gd'] = Material.from_color(shader, (0.7, 0.7, 0.7))
    materials['door'] = Material.from_color(shader, COL_CAR)
    
    root = build_scene(window, meshes=mesh_dict, materials=materials)

    # locate the Car node and wheel nodes from the constructed scene
    car_node = root.find('Car')
    if car_node is None:
        raise RuntimeError("Car node not found in scene after build_scene()")



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

    sky = Skybox(shader_program=skybox_shader.prog)

    renderer = Renderer()

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
        elif cam.mode == 'follow':
            cam_eye, cam_ctr = follow_cam(deltaTime)
        elif cam.mode == 'orbit':
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
            # upload per-shader globals: view/proj and lighting if API available
            try:
                if hasattr(shader_obj, 'set_common'):
                    # pass collected point lights (if any) to shader wrapper
                    shader_obj.set_common(V, P, light_dir, ambient, light_diffuse, lights)
                else:
                    # fallback: try to set uView/uProj uniforms directly
                    prog = getattr(shader_obj, 'prog', None)

                    if prog is not None:                        
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



if __name__ == "__main__":
    main()
