import glfw
import sys

class Window:
    def __init__(self, width: int, height: int, title: str):
        if not glfw.init():
            print("Failed to initialize GLFW", file=sys.stderr); sys.exit(1)
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.win = glfw.create_window(width, height, title, None, None)
        glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        if not self.win:
            glfw.terminate(); print("Failed to create window", file=sys.stderr); sys.exit(1)
        
        glfw.make_context_current(self.win)

        # Usar vsync para evitar o loop a correr demasiado depressa saturando a CPU
        try:
            glfw.swap_interval(1)
        except Exception:
            pass

    def should_close(self) -> bool:
        return glfw.window_should_close(self.win)
    
    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.win)

    def get_framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.win)