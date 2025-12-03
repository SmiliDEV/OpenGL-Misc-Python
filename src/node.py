import numpy as np
import os

class Node:
    def __init__(self, name="Node", local=None, mesh=None,animator=None,
                 material=None, vs_path: str = None, fs_path: str = None, on_draw=None):
        self.name = name
        # Aqui coloca-se a matriz de transformação local
        self.local = np.array(local if local is not None else np.eye(4, dtype=np.float32), dtype=np.float32)
        self.children = []
        self.mesh = mesh
        self.animator = animator  # função de animação
        self.material = material
        self.on_draw = on_draw
        # light properties: if True this node represents a point light
        self.is_light = False
        # light color (rgb) and scalar intensity
        self.light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.light_intensity = 1.0
        # optional range (for attenuation); if None, no attenuation
        self.light_range = None
    
    def add(self, *children):
        for c in children: self.children.append(c)
        return self

    def update(self, dt):
        if self.animator:
            self.animator(self, dt)
        for c in self.children: c.update(dt)

    def draw(self, parent_world=None):
        """Compute world transforms and recurse children.

        Note: this method intentionally does not perform any GL work or
        shader activation. The `Renderer` is responsible for binding shaders
        and issuing draw calls.
        """
        if parent_world is None:
            parent_world = np.eye(4, dtype=np.float32)
        world = parent_world @ self.local

        # recurse to children; actual drawing happens in Renderer
        for c in self.children:
            c.draw(world)

    def find(self, name: str):
        """Recursively find a child node by name (including self). Returns the Node or None."""
        if self.name == name:
            return self
        for c in self.children:
            r = c.find(name)
            if r is not None:
                return r
        return None