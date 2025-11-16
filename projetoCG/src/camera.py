from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import glfw
import numpy as np

# remove this
import glm

@dataclass
class Camera:
    mode: str = "free"
    
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
    
    def update(self, xoffset, yoffset):
        self.yaw += xoffset
        self.pitch += yoffset
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0
        self._update_vectors()


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
    dt = min(dt, 0.05)
    rmb = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    x, y = glfw.get_cursor_pos(win)
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
        cam.yaw   += cam.mouse_sens * dx
        cam.pitch -= cam.mouse_sens * dy
        cam.pitch = _clamp(cam.pitch, math.radians(-85.0), math.radians(85.0))

    fwd = _forward_from_yaw_pitch(cam.yaw, 0.0)  
    right = np.array([math.cos(cam.yaw + math.pi/2.0), 0.0, math.sin(cam.yaw + math.pi/2.0)], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = np.zeros(3, dtype=np.float32)
    
    if glfw.get_key(win, glfw.KEY_W) in (glfw.PRESS, glfw.REPEAT): v += fwd
    if glfw.get_key(win, glfw.KEY_S) in (glfw.PRESS, glfw.REPEAT): v -= fwd
    if glfw.get_key(win, glfw.KEY_D) in (glfw.PRESS, glfw.REPEAT): v += right
    if glfw.get_key(win, glfw.KEY_A) in (glfw.PRESS, glfw.REPEAT): v -= right
    if glfw.get_key(win, glfw.KEY_E) in (glfw.PRESS, glfw.REPEAT): v += up
    if glfw.get_key(win, glfw.KEY_Q) in (glfw.PRESS, glfw.REPEAT): v -= up

    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
        spd = cam.speed * (2.0 if glfw.get_key(win, glfw.KEY_LEFT_SHIFT) in (glfw.PRESS, glfw.REPEAT) else 1.0)
        cam.pos = cam.pos + v * (spd * dt)


def get_view_free(cam: Camera):
    fwd = _forward_from_yaw_pitch(cam.yaw, cam.pitch)
    eye = cam.pos.astype(np.float32)
    center = eye + fwd
    return eye, center
