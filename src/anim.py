"""Animadores e aplicação de estado aos Nodes.

Este módulo NÃO conhece OpenGL diretamente; só transforma o estado
(`Car` de carro.py) em matrizes locais de `Node`.
"""

import math
from typing import Dict, Callable

import glfw  # para ler inputs de teclado na janela
import numpy as np

from math3d import translate, rotate, scale

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

            r_steer = rotate(car_state.steer if w.is_front else 0.0, (0, -1, 0))
            r_x = rotate(math.pi / 2.0, (1, 0, 0))
            # make rear wheels rotate visually a bit faster than front wheels
            rear_spin_multiplier = 1.5
            spin_val = w.spin * (rear_spin_multiplier if not w.is_front else 1.0)
            r_spin = rotate(spin_val, (1, 0, 0))

            # apply per-wheel scale so visual size matches wheel.radius and wheel.width
            # order: translate -> steer -> rotate X to wheel frame -> spin -> scale
            m = (
                translate(w.ox, w.oy, w.oz) @ r_steer @ r_x @ r_spin @ scale(w.radius, w.radius, w.width)
            )
            
            node.local = m

        # Steering wheel: find a node named 'SteeringWheel' anywhere under the car node
        # and rotate it relative to its base local transform according to car_state.steer.
        sw = car_node.find('SteeringWheel')
        if sw is not None:
            try:
                base = sw.local.copy()
                t = base[:3, 3].copy()
                S = base[:3, :3]
                sx, sy, sz = np.linalg.norm(S, axis=0).tolist()
                # steering wheel typically rotates more than wheel steer angle
                STEER_WHEEL_RATIO = 8.0
                angle = -car_state.steer * STEER_WHEEL_RATIO
                # rotation axis aligned with forward vector of car (+X)
                r_sw = rotate(angle, (0, 0, -1))
                sw.local = translate(float(t[0]), float(t[1]), float(t[2])) @ r_sw @ scale(float(sx), float(sy), float(sz))
            except Exception:
                # keep the original transform if anything goes wrong
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


def make_follow_camera_2(
    get_car_matrix: Callable[[], np.ndarray], *,
    offset_local=(-10.0, 4.0, 0.0), look_ahead=6.0,
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
            
        eye = prev_eye + (off_world - prev_eye)
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


def make_garage_door_animator(
    node,
    *,
    win,
    key=glfw.KEY_F,
    open_offset_y: float = -1.5,
    speed: float = 6.0,
):
    """Return an animator that toggles a vertical (Y) translation of `node` when `key` is pressed.

    Behavior:
    - The animator stores the node's base local matrix and applies an additional Y translation
      when the door is 'open'. Pressing the key toggles open/closed (rising-edge detection).
    - Movement is smoothed; `speed` controls responsiveness (higher is snappier).

    Parameters:
    - node: Node instance to animate
    - win: window or glfw window pointer (used to read keyboard state)
    - key: GLFW key to toggle with (default F)
    - open_offset_y: Y translation amount applied when opened (negative slides down by default)
    - speed: smoothing speed in units/sec
    """

    gw = getattr(win, 'win', win)

    # Capture base transform and compute base translation
    base_local = node.local.copy()
    base_t = base_local[:3, 3].copy()

    # State variables in closure
    is_open = False
    prev_key = False
    current_y = float(base_t[1])

    def anim(n, dt: float):
        nonlocal is_open, prev_key, current_y, base_local

        # Read current key state
        cur_pressed = glfw.get_key(gw, key) in (glfw.PRESS, glfw.REPEAT)

        # detect rising edge: key pressed now, but not in previous frame
        if cur_pressed and not prev_key:
            is_open = not is_open

        prev_key = cur_pressed

        target_y = float(base_t[1] + (open_offset_y if is_open else 0.0))

        # smooth towards target
        if dt > 0:
            alpha = 1.0 - pow(2.718281828, -speed * dt)
        else:
            alpha = 1.0

        current_y += (target_y - current_y) * alpha

        # rebuild node.local with updated Y offset while preserving rotation/scale
        M = base_local.copy()
        M[:3, 3] = np.array([float(base_t[0]), -current_y, float(base_t[2])], dtype=np.float32)
        n.local = M

    return anim

def make_door_anim(node, win, key, open_angle_deg=70.0, axis=(0,1,0), speed=8.0):
    if node is None:
        return None
    gw = getattr(win, 'win', win)
    # capture base translation and scale from initial local
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