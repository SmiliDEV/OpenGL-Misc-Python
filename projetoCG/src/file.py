import os

def get_content_of_file_project(file_path: str) -> str:
    file_path = os.path.join(os.path.dirname(__file__), file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return content