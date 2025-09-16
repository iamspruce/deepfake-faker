import os, zipfile, platform, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["face", "voice"]

def run(cmd, cwd=None):
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True, cwd=cwd)

def write_start_files(build_dir, backend_name, binary_name):
    start_sh = build_dir / f"start_{backend_name}.sh"
    start_bat = build_dir / f"start_{backend_name}.bat"
    start_sh.write_text(f"#!/bin/bash\ncd $(dirname $0)\nchmod +x ./{binary_name}\n./{binary_name}\n")
    start_bat.write_text(f"@echo off\ncd %~dp0\n{binary_name}.exe\n")

def write_version_file(build_dir, version):
    (build_dir / "VERSION").write_text(version)

def package_backend(backend, version, device, os_name):
    backend_dir = ROOT / "backends" / backend
    build_dir = ROOT / f"build_{backend}_{os_name}_{device}"
    
    # Clean build folder
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # Copy src/, models/, and the main entry script
    if (backend_dir / "src").exists():
        shutil.copytree(backend_dir / "src", build_dir / "src")
    if (backend_dir / "models").exists():
        shutil.copytree(backend_dir / "models", build_dir / "models")
    shutil.copy(backend_dir / f"{backend}_main.py", build_dir / f"{backend}_main.py")

    # Install Python and dependencies in a temp venv for PyInstaller
    venv_dir = build_dir / "venv"
    run(f"\"{sys.executable}\" -m venv \"{venv_dir}\"")
    
    # Create OS-specific paths for pip and pyinstaller
    if os_name == "windows":
        pip = venv_dir / "Scripts" / "pip.exe"
        pyinstaller = venv_dir / "Scripts" / "pyinstaller.exe"
    else:
        pip = venv_dir / "bin" / "pip"
        pyinstaller = venv_dir / "bin" / "pyinstaller"

    run(f"\"{pip}\" install --upgrade pip setuptools wheel pyinstaller")

    # Install backend-specific dependencies
    requirements_file = backend_dir / "requirements.txt"
    run(f"\"{pip}\" install -r \"{requirements_file}\"")

    # Backend-specific installation order for Torch
    if backend == "voice":
        if device == "cpu":
            run(f"\"{pip}\" install torch==2.0.1")
            run(f"\"{pip}\" install fairseq==0.12.2 faiss-cpu==1.7.3")
        else:
            run(f"\"{pip}\" install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
            run(f"\"{pip}\" install fairseq==0.12.2 faiss-gpu==1.7.3")
    elif backend == "face":
        if device == "cpu":
            run(f"\"{pip}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            run(f"\"{pip}\" install onnxruntime")
        else:
            run(f"\"{pip}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            run(f"\"{pip}\" install onnxruntime-gpu")

    # Build executable with PyInstaller
    entry_script = build_dir / f"{backend}_main.py"
    binary_name = f"{backend}_backend"
    run(f"\"{pyinstaller}\" --onefile --name {binary_name} \"{entry_script}\"", cwd=build_dir)

    # Define paths for zipped contents
    dist_dir = build_dir / "dist"
    final_build_dir = build_dir / "package"
    final_build_dir.mkdir()

    # Move executable
    binary_suffix = ".exe" if os_name == "windows" else ""
    shutil.move(dist_dir / f"{binary_name}{binary_suffix}", final_build_dir / f"{binary_name}{binary_suffix}")

    # Copy other assets to the final package directory
    if (build_dir / "src").exists():
        shutil.copytree(build_dir / "src", final_build_dir / "src")
    if (build_dir / "models").exists():
        shutil.copytree(build_dir / "models", final_build_dir / "models")
    
    write_version_file(final_build_dir, version)
    write_start_files(final_build_dir, backend, binary_name)

    # Prepare zip from the clean 'final_build_dir'
    zip_path = ROOT / f"{backend}-{os_name}-{device}-v{version.lstrip('v')}.zip"
    shutil.make_archive(zip_path.with_suffix(''), 'zip', final_build_dir)

    print(f"[DONE] {zip_path}")

if __name__ == "__main__":
    version = os.environ.get("GITHUB_REF_NAME", "dev")
    if not version:
        version = "dev"

    # Example: run for a specific target for faster testing
    # package_backend("face", version, "cpu", "windows")
    # sys.exit(0)

    systems = ["windows", "linux", "macos"]
    devices = ["cpu", "gpu"]

    for backend in BACKENDS:
        for os_name in systems:
            # Skip unsupported combinations
            if os_name == "macos" and device == "gpu":
                print(f"[SKIP] Skipping {backend} for {os_name}/{device} (unsupported).")
                continue
            
            for device in devices:
                try:
                    package_backend(backend, version, device, os_name)
                except Exception as e:
                    print(f"[FAIL] Failed to package {backend} for {os_name}/{device}: {e}")