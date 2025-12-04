import glfw
import sys

class Window:
    def __init__(self, width: int, height: int, title: str, fullscreen=False):
        if not glfw.init():
            print("Failed to initialize GLFW", file=sys.stderr); sys.exit(1)
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        monitor = None
        self.fullscreen = fullscreen
        self._saved_pos = (100, 100)
        self._saved_size = (width, height)

        if fullscreen:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            width = mode.size.width
            height = mode.size.height

        self.win = glfw.create_window(width, height, title, monitor, None)
        glfw.set_input_mode(self.win, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        if not self.win:
            glfw.terminate(); print("Failed to create window", file=sys.stderr); sys.exit(1)
        
        glfw.make_context_current(self.win)
        glfw.set_window_user_pointer(self.win, self)

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

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # Save current window params before switching
            self._saved_pos = glfw.get_window_pos(self.win)
            self._saved_size = glfw.get_window_size(self.win)
            
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(self.win, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        else:
            # Restore
            x, y = self._saved_pos
            w, h = self._saved_size
            glfw.set_window_monitor(self.win, None, x, y, w, h, 0)