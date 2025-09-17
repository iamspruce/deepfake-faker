import os
import platform

is_windows = platform.system() == "Windows"

def get_venv_python(backend_path: str):
    if is_windows:
        return os.path.join(backend_path, "venv", "Scripts", "python.exe")
    else:
        return os.path.join(backend_path, "venv", "bin", "python")


