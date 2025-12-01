from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class Wheel:
    name: str
    ox: float
    oy: float
    oz: float
    radius: float
    width: float
    is_front: bool
    spin: float = 0.0  # rad (rotação acumulada visual)


@dataclass
class Car:
    # Posição e orientação no plano XZ (Y para cima)
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0          # rad (0 -> +X)
    # Vetor frente normalizado (mantém referência estável da direção "para a frente")
    fx: float = 1.0           # inicial +X
    fz: float = 0.0

    # Estado principal
    v: float = 0.0            # m/s (+ frente / - trás)
    steer: float = 0.0        # rad (ângulo das rodas da frente)

    # Geometria
    L: float = 2.6            # distância entre eixos
    wheel_radius: float = 1.0

    # Visual
    wheel_spin: float = 0.0   # rad

    # Tuning
    accel: float = 14.0
    drag: float = 1.5
    vmax: float = 20.0
    max_steer: float = math.radians(30.0)
    steer_rate: float = math.radians(90.0)
    steer_return: float = math.radians(120.0)

    # Bitolas (futuro Ackermann)
    front_track: float = 1.5
    rear_track: float = 1.5

    # Rodas (se não fornecidas, cria 4 genéricas)
    wheels: List[Wheel] = field(default_factory=list)

    def __post_init__(self):
        if not self.wheels:
            self.wheels = [
                Wheel(name="Wheel_FL", ox=0.175, oy=-0.11, oz=0.35, radius=0.13, width=0.13, is_front=True),
                Wheel(name="Wheel_FR", ox=-0.175, oy=-0.11, oz=0.35, radius=0.13, width=0.13, is_front=True),
                Wheel(name="Wheel_RL", ox=0.175, oy=-0.10, oz=-0.30, radius=0.16, width=0.16, is_front=False),
                Wheel(name="Wheel_RR", ox=-0.175, oy=-0.10, oz=-0.30, radius=0.16, width=0.16, is_front=False),
            ]



def step_controls(car: Car, dt: float, fwd: bool, rev: bool, left: bool, right: bool) -> None:
    """Atualiza velocidade e esterço a partir dos inputs (sem mexer em posição)."""
    # Velocidade: aceleração positiva ou negativa + drag proporcional
    dt = min(dt, 0.05)  # evitar saltos grandes
    v_prev = car.v
    a = (1.0 if fwd else 0.0) - (1.0 if rev else 0.0)
    a *= car.accel
    a -= car.drag * car.v
    car.v += a * dt
    # Anti-overshoot: se o arrasto inverter sinal sem comando, trava em zero
    if not fwd and not rev and v_prev * car.v < 0.0:
        car.v = 0.0
    # Bloqueia marcha-atrás se não estiver a carregar reverse
    if not rev and car.v < 0.0:
        car.v = 0.0
    # Limites
    if car.v > car.vmax: car.v = car.vmax
    elif car.v < -car.vmax: car.v = -car.vmax


    # Esterço (RIGHT aumenta yaw se frente for -X, ajustar se invertido). Aqui queremos que RIGHT rode para a direita.
    s_in = (1.0 if right else 0.0) - (1.0 if left else 0.0)
    if abs(s_in) > 1e-9:
        car.steer += s_in * car.steer_rate * dt
    else:
        # retorno para zero (centragem)
        car.steer -= car.steer * car.steer_return * dt
    # Clamp
    if car.steer > car.max_steer: car.steer = car.max_steer
    elif car.steer < -car.max_steer: car.steer = -car.max_steer


def step_pose(car: Car, dt: float) -> None:
    """Atualiza posição usando yaw como única fonte de direção (frente = +X).

    1. Integra yaw via modelo bicicleta (omega = v * tan(steer) / L).
    2. Deriva vetor frente diretamente de yaw (fx, fz).
    3. Move ao longo desse vetor.
    """
    dt = min(dt, 0.05)
    steer = car.steer
    if abs(steer) < 1e-5:
        omega = 0.0
    else:
        kappa = math.tan(steer) / car.L
        # Permite rotação mínima parado para orientar frente
        eff_v = car.v if abs(car.v) >= 0.15 else (0.15 if car.v >= 0.0 else -0.15)
        omega = eff_v * kappa
    car.yaw += omega * dt
    # Normaliza yaw
    if car.yaw > math.pi:
        car.yaw -= 2 * math.pi
    elif car.yaw < -math.pi:
        car.yaw += 2 * math.pi
    # Frente do modelo = -X (inverte vector para movimento desejado se estava a ir ao contrário)
    car.fx = math.cos(car.yaw)
    car.fz = math.sin(car.yaw)
    # Avança
    car.x += car.v * dt * car.fx
    car.z += car.v * dt * car.fz


def step_wheels(car: Car, dt: float) -> None:
    """Atualiza giro visual das rodas."""
    dt = min(dt,0.05)  # evitar saltos grandes
    
    if car.wheel_radius > 0.0:
        car.wheel_spin += (car.v / car.wheel_radius) * dt

    for w in car.wheels:
        if w.radius > 0.0:
            w.spin += (car.v / w.radius) * dt


def update_car(car: Car, dt: float, *, fwd: bool, rev: bool, left: bool, right: bool) -> None:
    """Pipeline completo de atualização de estado do carro num frame."""
    step_controls(car, dt, fwd=fwd, rev=rev, left=left, right=right)
    step_pose(car, dt)
    step_wheels(car, dt)


# Conveniência para compatibilidade antiga (se alguém chamar carro())
carro = Car

