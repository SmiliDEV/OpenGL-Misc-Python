import sys

class MouseCallbackRouter:
    """
    Regista funções que querem receber triggers do mouse_callback do GLFW.
    Uso:
      router = MouseCallbackRouter()
      router.register(func)           # func(window, xpos, ypos)
      router.unregister(func)
      router.list_subscribers() -> list[str]
      router.dispatch(window,xpos,ypos)  # chama os inscritos (seguro a exceções)
    """
    def __init__(self):
        self._subs = []

    def register(self, func):
        if callable(func) and func not in self._subs:
            self._subs.append(func)

    def unregister(self, func):
        try:
            self._subs.remove(func)
        except ValueError:
            pass

    def list_subscribers(self):
        # devolve nomes amigáveis para inspeção
        out = []
        for f in self._subs:
            name = getattr(f, "__name__", None) or getattr(f, "__qualname__", None) or repr(f)
            out.append(name)
        return out

    def dispatch(self, window, xpos, ypos):
        for f in list(self._subs):
            try:
                f(window, xpos, ypos)
            except Exception as e:
                # evita quebrar o loop caso um subscriber falhe
                print(f"Mouse subscriber error in {getattr(f,'__name__',repr(f))}: {e}", file=sys.stderr)

class KeyCallbackRouter:
    """
    Regista funções que querem receber triggers de teclado do GLFW.
    Uso:
      key_router = KeyCallbackRouter()
      key_router.register(func)   # func(window, key, scancode, action, mods)
      key_router.unregister(func)
      key_router.list_subscribers() -> list[str]
      key_router.dispatch(window, key, scancode, action, mods)
    """
    def __init__(self):
        self._subs = []

    def register(self, func):
        if callable(func) and func not in self._subs:
            self._subs.append(func)

    def unregister(self, func):
        try:
            self._subs.remove(func)
        except ValueError:
            pass

    def list_subscribers(self):
        out = []
        for f in self._subs:
            name = getattr(f, "__name__", None) or getattr(f, "__qualname__", None) or repr(f)
            out.append(name)
        return out

    def dispatch(self, window, key, scancode, action, mods):
        for f in list(self._subs):
            try:
                f(window, key, scancode, action, mods)
            except Exception as e:
                print(f"Key subscriber error in {getattr(f,'__name__',repr(f))}: {e}", file=sys.stderr)
