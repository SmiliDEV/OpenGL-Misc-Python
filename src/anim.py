"""Animadores e aplicação de estado aos Nodes.

Este módulo NÃO conhece OpenGL diretamente; só transforma o estado
(`Car` de carro.py) em matrizes locais de `Node`.
"""

import math
from typing import Dict, Callable

import glfw
import numpy as np

from .math3d import translate, rotate, scale
from .carro import Car, update_car

# Helpers esperados serão injectados (translate, rotate, scale). Mantemos aqui só nomes.
def make_car_animators(
    *,
    win,
    car_state: Car,
    car_node,
    wheel_nodes: Dict[str, any],  # {'FL':Node, 'FR':Node, ...}
    translate, rotate, scale,
) -> Callable[[float], None]:
    # aceitar wrapper Window (com .win) ou ponteiro GLFWwindow*
    gw = getattr(win, "win", win)

    def apply_transforms():
        m_car = translate(car_state.x, 0.0, car_state.z) @ rotate(-car_state.yaw, (0, 1, 0))
        car_node.local = m_car

        for w in car_state.wheels:
            # Preferir correspondência exata pelo nome completo ('Wheel_FL') ou sufixo curto ('FL').
            # Se o nó não estiver presente na cena, pular a atualização dessa roda
            # em vez de recorrer a outra roda (o que duplicaria as transformações).
            node = wheel_nodes.get(w.name) or wheel_nodes.get(w.name.split()[-1])
            if node is None:
                # nó da roda não presente na cena (usuário removeu rodas traseiras, etc.)
                continue

            r_steer = rotate(car_state.steer if w.is_front else 0.0, (0, -1, 0))
            r_x = rotate(math.pi / 2.0, (1, 0, 0))
            # fazer as rodas traseiras girarem visualmente um pouco mais rápido que as dianteiras
            rear_spin_multiplier = 1.5
            spin_val = w.spin * (rear_spin_multiplier if not w.is_front else 1.0) * 0.2  # para não rodar tão rápido
            r_spin = rotate(spin_val, (1, 0, 0))

            # aplicar escala por roda para que o tamanho visual corresponda a wheel.radius e wheel.width
            # ordem: translação -> direção -> rotação X para o frame da roda -> giro -> escala
            m = (
                translate(w.ox, w.oy, w.oz) @ r_steer @ r_x @ r_spin @ scale(w.radius, w.radius, w.width)
            )

            node.local = m

        # Volante: encontrar um nó chamado 'SteeringWheel' em qualquer lugar sob o nó do carro
        # e rotacioná-lo em relação à sua transformação local base de acordo com car_state.steer.
        sw = car_node.find('SteeringWheel')
        if sw is not None:
            try:
                base = sw.local.copy()
                t = base[:3, 3].copy()
                S = base[:3, :3]
                sx, sy, sz = np.linalg.norm(S, axis=0).tolist()
                # o volante normalmente gira mais do que o ângulo de direção da roda
                STEER_WHEEL_RATIO = 6.0
                angle = -car_state.steer * STEER_WHEEL_RATIO
                # eixo de rotação alinhado com o vetor frontal do carro (+X)
                r_sw = rotate(angle, (0, 0, -1))
                sw.local = translate(float(t[0]), float(t[1]), float(t[2])) @ r_sw @ scale(float(sx), float(sy), float(sz))
            except Exception:
                # manter a transformação original se algo der errado
                pass

    def anim(node, dt: float):
        fwd = glfw.get_key(gw, glfw.KEY_UP) in (glfw.PRESS, glfw.REPEAT)
        rev = glfw.get_key(gw, glfw.KEY_DOWN) in (glfw.PRESS, glfw.REPEAT)
        left = glfw.get_key(gw, glfw.KEY_LEFT) in (glfw.PRESS, glfw.REPEAT)
        right = glfw.get_key(gw, glfw.KEY_RIGHT) in (glfw.PRESS, glfw.REPEAT)

        update_car(car_state, dt, fwd=fwd, rev=rev, left=left, right=right)
        apply_transforms()

    return anim



def make_follow_camera(
    get_car_matrix: Callable[[], np.ndarray], *,
    offset_local=(-10.0, 4.0, 0.0), look_ahead=6.0,
    lag_seconds=0.15
):
    """Retorna função dt->(eye,center) que segue o carro suavemente."""
    offset_local = np.array(offset_local, dtype=np.float32)
    prev_eye = None

    def follow(dt: float):
        nonlocal prev_eye
        m = get_car_matrix()
        fwd = m[:3, 0]
        fwd = fwd / np.linalg.norm(fwd)
        pos = m[:3, 3]
        off_world = pos + m[:3, :3] @ offset_local
        target = pos + fwd * look_ahead
        if prev_eye is None:
            prev_eye = off_world
        alpha = 1.0 - math.exp(-dt / max(1e-6, lag_seconds))
        eye = prev_eye + (off_world - prev_eye) * alpha
        prev_eye = eye
        return eye, target

    return follow


def make_sun_animator(
    sun_node,
    *,
    translate,
    rotate,
    scale,
    orbit_radius=100.0,
    orbit_period=4.0,
    tilt_angle_deg=0.0,
) -> Callable[[float], None]:
    def anim(node, dt: float):
        time = glfw.get_time()
        angle = (time / orbit_period) * 2.0 * math.pi
        # Orbitar em torno do eixo Z para simular nascer/pôr do sol (altera altura Y)
        orbit_rot = rotate(angle, (0, 0, 1))
        # Posição inicial no eixo X
        pos = orbit_rot @ np.array([orbit_radius, 0.0, 0.0, 1.0], dtype=np.float32)
        m = translate(pos[0], pos[1], pos[2]) @ scale(2.0, 2.0, 2.0)
        sun_node.local = m

    return anim


def make_garage_door_animator(
    node,
    *,
    win,
    key=glfw.KEY_F,
    open_offset_y: float = -1.5,
    speed: float = 6.0,
):
    gw = getattr(win, 'win', win)

    # Capturar transformação base e calcular translação base
    base_local = node.local.copy()
    base_t = base_local[:3, 3].copy()

    # Variáveis de estado no closure
    is_open = False
    prev_key = False
    current_y = float(base_t[1])

    def anim(n, dt: float):
        nonlocal is_open, prev_key, current_y, base_local

        # Ler estado atual da tecla
        cur_pressed = glfw.get_key(gw, key) in (glfw.PRESS, glfw.REPEAT)

        # detectar borda de subida: tecla pressionada agora, mas não no quadro anterior
        if cur_pressed and not prev_key:
            is_open = not is_open

        prev_key = cur_pressed

        target_y = float(base_t[1] + (open_offset_y if is_open else 0.0))

        # suavizar em direção ao alvo
        if dt > 0:
            alpha = 1.0 - pow(2.718281828, -speed * dt)
        else:
            alpha = 1.0

        current_y += (target_y - current_y) * alpha

        # reconstruir node.local com deslocamento Y atualizado preservando rotação/escala
        M = base_local.copy()
        M[:3, 3] = np.array([float(base_t[0]), -current_y, float(base_t[2])], dtype=np.float32)
        n.local = M

    return anim

def make_door_anim(node, win, key, open_angle_deg=70.0, axis=(0,1,0), speed=8.0):
    if node is None:
        return None
    gw = getattr(win, 'win', win)
    # capturar translação base e escala do local inicial
    base = node.local.copy()
    t = base[:3, 3].copy()
    S = base[:3, :3]
    sx, sy, sz = np.linalg.norm(S, axis=0).tolist()
    current = 0.0
    target_open = math.radians(open_angle_deg)

    def anim_fn(n, dt: float):
        nonlocal current
        pressed = glfw.get_key(gw, key) in (glfw.PRESS, glfw.REPEAT)
        target = target_open if pressed else 0.0
        alpha = 1.0 - math.exp(-speed * dt) if dt > 0 else 1.0
        current += (target - current) * alpha
        n.local = translate(float(t[0]), float(t[1]), float(t[2])) @ rotate(current, axis) @ scale(float(sx), float(sy), float(sz))

    return anim_fn
def make_plane_animator(node, center_pos, axis, radius, speed, height, scale_factor=5.0, spin_speed=0.0):
    angle = 0.0
    spin_angle = 0.0
    axis = np.array(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis)

    # Inicializar Rv_prev arbitrário (direita inicial do avião)
    Rv_prev = np.array([0, 0, 1], dtype=np.float32)

    def animator(n, dt):
        nonlocal angle, spin_angle, Rv_prev
        angle += dt * speed
        spin_angle += dt * spin_speed * 4

        # --- 1. Posição na órbita ---
        p0 = np.array([radius, 0, 0, 1], dtype=np.float32)
        R_orbit = rotate(angle, axis)
        p = R_orbit @ p0
        P = np.array(center_pos, dtype=np.float32) + np.array([p[0], p[1]+height, p[2]], dtype=np.float32)

        # --- 2. Tangente ---
        eps = 1e-3
        R_orbit2 = rotate(angle + eps, axis)
        p2 = R_orbit2 @ p0
        P2 = np.array(center_pos, dtype=np.float32) + np.array([p2[0], p2[1]+height, p2[2]], dtype=np.float32)

        T = P2 - P
        T /= np.linalg.norm(T)
        T = -T  # inverter para frente do avião

        # --- 3. Construir frame ortonormal ---
        U = axis

        # Right (Rv) perpendicular a tangente e up
        Rv = np.cross(U, T)
        norm = np.linalg.norm(Rv)
        if norm < 1e-6:
            # se muito pequeno, usar eixo anterior
            Rv = Rv_prev
        else:
            Rv /= norm

        # Garantir consistência: inverter se mudou de sinal
        if np.dot(Rv, Rv_prev) < 0:
            Rv = -Rv
        Rv_prev = Rv.copy()

        # Up corrigido
        Uv = np.cross(T, Rv)

        # --- 4. Spin sobre eixo local ---
        R_spin = rotate(spin_angle, Rv)  # spin sobre eixo direito (perpendicular à direção)

        # --- 5. Matriz de orientação ---
        orient = np.array([
            [T[0], Uv[0], Rv[0], 0],
            [T[1], Uv[1], Rv[1], 0],
            [T[2], Uv[2], Rv[2], 0],
            [0,    0,    0,    1]
        ], dtype=np.float32)

        # --- 6. Escala e posição final ---
        S = scale(scale_factor, scale_factor, scale_factor)
        T_pos = translate(P[0], P[1], P[2])

        # Rotação sobre o eixo Z local (roll) para o avião girar sobre si mesmo
        R_spin_local = rotate(spin_angle, np.array([1,0,0], dtype=np.float32))

        # --- 7. Combinar matrizes ---
        # Ordem: Escala -> Spin Local -> Orientação na Trajetória -> Posição no Mundo
        n.local = T_pos @ orient @ R_spin_local @ S

    return animator