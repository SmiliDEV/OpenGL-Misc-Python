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
	def apply_transforms():
		M_car = translate(car_state.x, 0.0, car_state.z) @ rotate(-car_state.yaw, (0,1,0))
		car_node.local = M_car 
		for w in car_state.wheels:
			node = wheel_nodes.get(w.name.split()[-1]) or wheel_nodes.get(w.name) or next(iter(wheel_nodes.values()))

			Rsteer = rotate(car_state.steer if w.is_front else 0.0, (0,1,0))
			Rx = rotate(math.pi/2.0, (1,0,0))
			Rspin = rotate(w.spin, (0,0,1))
			M = translate(w.ox, w.oy, w.oz) @ Rsteer @ Rx @ Rspin @ scale(w.radius, w.radius, w.width)
			node.local = M

	def anim(node, dt: float): 
		fwd  = glfw.get_key(win, glfw.KEY_UP)    in (glfw.PRESS, glfw.REPEAT)
		rev  = glfw.get_key(win, glfw.KEY_DOWN)  in (glfw.PRESS, glfw.REPEAT)
		left = glfw.get_key(win, glfw.KEY_LEFT)  in (glfw.PRESS, glfw.REPEAT)
		right= glfw.get_key(win, glfw.KEY_RIGHT) in (glfw.PRESS, glfw.REPEAT)

		update_car(car_state, dt, fwd=fwd, rev=rev, left=left, right=right)
		apply_transforms()

	return anim


def make_follow_camera(get_car_matrix: Callable[[], np.ndarray], *,
					   offset_local=(-10.0, 4.0, 0.0), look_ahead=6.0,
					   lag_seconds=0.15):
	"""Retorna função dt->(eye,center) que segue o carro suavemente."""
	offset_local = np.array(offset_local, dtype=np.float32)
	prev_eye = None

	def follow(dt: float):
		nonlocal prev_eye
		M = get_car_matrix()
		fwd = M[:3,0]; fwd = fwd/np.linalg.norm(fwd)
		pos = M[:3,3]
		off_world = pos + M[:3,:3] @ offset_local
		target = pos + fwd * look_ahead
		if prev_eye is None:
			prev_eye = off_world
		alpha = 1.0 - math.exp(-dt / max(1e-6, lag_seconds))
		eye = prev_eye + (off_world - prev_eye) * alpha
		prev_eye = eye
		return eye, target

	return follow

