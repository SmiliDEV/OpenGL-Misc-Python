import glfw
import sys
from event import MouseCallbackRouter, KeyCallbackRouter

class Window:
    def __init__(self, width: int, height: int, title: str):
        self.mouse_router = MouseCallbackRouter()
        self.key_router = KeyCallbackRouter()

        if not glfw.init():
            print("Failed to initialize GLFW", file=sys.stderr); sys.exit(1)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.win = glfw.create_window(width, height, title, None, None)
        if not self.win:
            glfw.terminate(); print("Failed to create window", file=sys.stderr); sys.exit(1)
        glfw.make_context_current(self.win)

        # Usar vsync para evitar o loop a correr demasiado depressa saturando a CPU
        try:
            glfw.swap_interval(1)
        except Exception:
            pass
        
        # wrappers simples que podem ser usados como callbacks únicos do GLFW
        def _glfw_mouse_wrapper(window, xpos, ypos):
            # dispara os inscritos do router, seguro a exceções
            try:
                self.mouse_router.dispatch(window, xpos, ypos)
            except Exception:
                pass

        def _glfw_key_wrapper(window, key, scancode, action, mods):
            try:
                self.key_router.dispatch(window, key, scancode, action, mods)
            except Exception:
                pass

        glfw.set_key_callback(self.win, _glfw_key_wrapper)
        glfw.set_cursor_pos_callback(self.win, _glfw_mouse_wrapper)
        glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_DISABLED)

    def should_close(self) -> bool:
        return glfw.window_should_close(self.win)
    
    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.win)

    def get_framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.win)