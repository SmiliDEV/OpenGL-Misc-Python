from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import glfw
import numpy as np


@dataclass
class Camera:
    mode: str = "free"
    pos: np.ndarray = None
    yaw: float = math.radians(35.0)
    pitch: float = math.radians(-15.0)
    speed: float = 8.0
    mouse_sens: float = math.radians(0.12)
    _last_mouse: Optional[Tuple[float, float]] = None
    _looking: bool = False

    def __post_init__(self):
        if self.pos is None:
            self.pos = np.array([-6.0, 4.0, -8.0], dtype=np.float32)


def _glfw_window_ptr(win):
    """Aceita wrapper Window (com .win) ou o ponteiro GLFWwindow* diretamente."""
    return getattr(win, "win", win)


def _clamp(x: float, a: float, b: float) -> float:
    if x < a:
        return a
    if x > b:
        return b
    return x


def _forward_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    cx, sx = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    fx = cx * cp
    fy = sp
    fz = sx * cp
    v = np.array([fx, fy, fz], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0, 0.0], dtype=np.float32)


def update_free_camera(win, cam: Camera, dt: float) -> None:
    """Atualiza a `cam` no modo livre. Aceita `win` como wrapper ou ponteiro."""
    dt = min(dt, 0.05)
    gw = _glfw_window_ptr(win)

    rmb = glfw.get_mouse_button(gw, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    x, y = glfw.get_cursor_pos(gw)
    if rmb and not cam._looking:
        cam._looking = True
        cam._last_mouse = (x, y)
    elif not rmb and cam._looking:
        cam._looking = False
        cam._last_mouse = (x, y)
    if cam._looking and cam._last_mouse is not None:
        dx = x - cam._last_mouse[0]
        dy = y - cam._last_mouse[1]
        cam._last_mouse = (x, y)
        cam.yaw += cam.mouse_sens * dx
        cam.pitch -= cam.mouse_sens * dy
        cam.pitch = _clamp(cam.pitch, math.radians(-85.0), math.radians(85.0))

    # movimento horizontal + strafing
    fwd = _forward_from_yaw_pitch(cam.yaw, 0.0)
    right = np.array([math.cos(cam.yaw + math.pi / 2.0), 0.0, math.sin(cam.yaw + math.pi / 2.0)], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)

    if glfw.get_key(gw, glfw.KEY_W) in (glfw.PRESS, glfw.REPEAT):
        v += fwd
    if glfw.get_key(gw, glfw.KEY_S) in (glfw.PRESS, glfw.REPEAT):
        v -= fwd
    if glfw.get_key(gw, glfw.KEY_D) in (glfw.PRESS, glfw.REPEAT):
        v += right
    if glfw.get_key(gw, glfw.KEY_A) in (glfw.PRESS, glfw.REPEAT):
        v -= right
    if glfw.get_key(gw, glfw.KEY_E) in (glfw.PRESS, glfw.REPEAT):
        v += up
    if glfw.get_key(gw, glfw.KEY_Q) in (glfw.PRESS, glfw.REPEAT):
        v -= up

    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
        spd = cam.speed * (2.0 if glfw.get_key(gw, glfw.KEY_LEFT_SHIFT) in (glfw.PRESS, glfw.REPEAT) else 1.0)
        cam.pos = cam.pos + v * (spd * dt)


def get_view_free(cam: Camera):
    fwd = _forward_from_yaw_pitch(cam.yaw, cam.pitch)
    eye = cam.pos.astype(np.float32)
    center = eye + fwd
    return eye, center

