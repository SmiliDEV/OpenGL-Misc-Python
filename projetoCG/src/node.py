import numpy as np

class Node:
    def __init__(self, name="Node", local=None, mesh=None, albedo=(1,1,1), animator=None):
        self.name = name
        #Aqui coloca-se a matriz de transformação local
        self.local = np.array(local if local is not None else np.eye(4, dtype=np.float32), dtype=np.float32)
        self.children = []
        self.mesh = mesh
        self.albedo = np.array(albedo, dtype=np.float32)
        self.animator = animator  # função de animação

    def add(self, *children):
        for c in children: self.children.append(c)
        return self

    def update(self, dt):
        if self.animator: self.animator(self, dt)
        for c in self.children: c.update(dt)

    def draw(self, shader, parent_world, vp, view_pos, light_dir, ambient):
        #parent_world é a matriz anterior existente
        world = parent_world @ self.local  
        if self.mesh is not None:
            shader.set_per_object(world, self.albedo)
            self.mesh.draw()
        for c in self.children:
            c.draw(shader, world, vp, view_pos, light_dir, ambient)