# Skybox Example (Python + OpenGL + GLFW)

Este projeto mostra uma skybox simples usando OpenGL, GLFW e PyGLM.

Requisitos
- Python 3.8+
- Dependências: `glfw`, `PyOpenGL`, `PyGLM`, `Pillow` (veja `requirements.txt`).

Instalação

Abra um terminal (PowerShell) e rode:

```powershell
python -m pip install -r requirements.txt
```

Uso

Coloque 6 imagens na pasta `textures/skybox` com nomes: `right`, `left`, `top`, `bottom`, `front`, `back` e extensões `.jpg` ou `.png`.

Execute:

```powershell
python main.py
```

Controles
- Mouse: olhar
- W/A/S/D: mover
- ESC: sair

Notas
- Se não quiser nomear as imagens exatamente, o script tenta casar arquivos que começam com `right`, `left`, etc.
