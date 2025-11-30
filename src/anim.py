"""Animadores e aplicação de estado aos Nodes.

Este módulo NÃO conhece OpenGL diretamente; só transforma o estado
(`Car` de carro.py) em matrizes locais de `Node`.
"""

import math
from typing import Dict, Callable

import glfw  # para ler inputs de teclado na janela
import numpy as np

from carro import Car, update_car

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
            # Prefer exact-match by full name ('Wheel_FL') or short suffix ('FL').
            # If the node isn't present in the scene, skip updating that wheel
            # instead of falling back to another wheel (which would duplicate transforms).
            node = wheel_nodes.get(w.name) or wheel_nodes.get(w.name.split()[-1])
            if node is None:
                # wheel node not present in the scene (user removed rear wheels, etc.)
                continue

            r_steer = rotate(car_state.steer if w.is_front else 0.0, (0, 1, 0))
            r_x = rotate(math.pi / 2.0, (1, 0, 0))
            r_spin = rotate(w.spin, (0, -1, 0))
            m = (
                translate(w.ox, w.oy, w.oz) @ r_steer @ r_x @ r_spin @ scale(w.radius, w.radius, w.width)
            )
            node.local = m

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
    orbit_radius=15.0,
    orbit_period=80.0,
    tilt_angle_deg=23.5
) -> Callable[[float], None]:
    """Retorna animador que faz o sol orbitar em círculo inclinado."""
    tilt_rad = math.radians(tilt_angle_deg)
    tilt_rot = rotate(tilt_rad, (0, 0, 1))

    def anim(node, dt: float):
        time = glfw.get_time()
        angle = (time / orbit_period) * 2.0 * math.pi
        orbit_rot = rotate(angle, (0, 0, 1))
        pos = orbit_rot @ tilt_rot @ np.array([orbit_radius, 1.0, 0.0, 1.0], dtype=np.float32)
        m = translate(pos[0], pos[1], pos[2]) @ scale(2.0, 2.0, 2.0)
        sun_node.local = m

    return anim
