import os, zipfile, platform, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["face", "voice"]
RUNNER_TO_OS = {
    'ubuntu-latest': 'linux',
    'windows-latest': 'windows',
    'macos-latest': 'macos'
}

IGNORE_DIRS = {"venv", "__pycache__", "models", "rvc_models"}

def run(cmd, cwd=None):
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    print(f"[SUCCESS] Command completed: {cmd}")
    return result

def write_start_files(build_dir, backend_name, binary_name):
    start_sh = build_dir / f"start_{backend_name}.sh"
    start_bat = build_dir / f"start_{backend_name}.bat"
    start_sh.write_text(f"#!/bin/bash\ncd $(dirname $0)\nchmod +x ./{binary_name}\n./{binary_name}\n")
    start_bat.write_text(f"@echo off\ncd %~dp0\n{binary_name}.exe\n")

def write_version_file(build_dir, version):
    (build_dir / "VERSION").write_text(version)

def copy_backend_files(backend_dir, build_dir):
    for item in backend_dir.iterdir():
        if item.name in IGNORE_DIRS:
            continue
        target = build_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy(item, target)


def package_backend(backend, version, device, os_name):
    backend_dir = ROOT / "backends" / backend
    build_dir = ROOT / f"build_{backend}_{os_name}_{device}"
    
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    copy_backend_files(backend_dir, build_dir)

    venv_dir = build_dir / "venv"
    run(f"\"{sys.executable}\" -m venv \"{venv_dir}\"")
    
    actual_os = platform.system().lower()
    if actual_os == "windows":
        pip = venv_dir / "Scripts" / "pip.exe"
        pyinstaller = venv_dir / "Scripts" / "pyinstaller.exe"
    else:
        pip = venv_dir / "bin" / "pip"
        pyinstaller = venv_dir / "bin" / "pyinstaller"

    run(f"\"{pip}\" install pyinstaller")
    run(f'"{pyinstaller}" --version')

    # Install backend requirements
    requirements_file = backend_dir / "requirements.txt"
    run(f"\"{pip}\" install -r \"{requirements_file}\" --no-cache-dir")

    # Install device-specific dependencies
    if backend == "voice":
        if device == "cpu":
            run(f"\"{pip}\" install torch==2.0.1")
            run(f"\"{pip}\" install fairseq==0.12.2 faiss-cpu==1.7.3")
        else:
            if actual_os in ["windows", "linux"]:
                run(f"\"{pip}\" install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir")
                run(f"\"{pip}\" install fairseq==0.12.2 faiss-gpu==1.7.2")
            else:
                print(f"[SKIP] Skipping GPU dependencies for {backend} on {actual_os}")
                return
    elif backend == "face":
        if device == "cpu":
            run(f"\"{pip}\" install torch torchvision torchaudio")
            run(f"\"{pip}\" install onnxruntime")
        else:
            if actual_os in ["windows", "linux"]:
                run(f"\"{pip}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                run(f"\"{pip}\" install onnxruntime-gpu")
            else:
                print(f"[SKIP] Skipping GPU dependencies for {backend} on {actual_os}")
                return

    # Build binary
    entry_script = build_dir / f"{backend}_main.py"
    binary_name = f"{backend}_backend"
    run(f"\"{pyinstaller}\" --onefile --name {binary_name} \"{entry_script}\"", cwd=build_dir)

    # Prepare final package
    dist_dir = build_dir / "dist"
    final_build_dir = build_dir / "package"
    final_build_dir.mkdir(parents=True, exist_ok=True)

    binary_suffix = ".exe" if os_name == "windows" else ""
    shutil.move(dist_dir / f"{binary_name}{binary_suffix}", final_build_dir / f"{binary_name}{binary_suffix}")

    if (build_dir / "src").exists():
        shutil.copytree(build_dir / "src", final_build_dir / "src")
    if (build_dir / "models").exists():
        shutil.copytree(build_dir / "models", final_build_dir / "models")

    write_version_file(final_build_dir, version)
    write_start_files(final_build_dir, backend, binary_name)

    # Zip final package
    zip_path = ROOT / f"{backend}-{os_name}-{device}-v{version.lstrip('v')}.zip"
    shutil.make_archive(zip_path.with_suffix(''), 'zip', final_build_dir)
    print(f"[DONE] {zip_path}")

if __name__ == "__main__":
    version = os.environ.get("GITHUB_REF_NAME", "dev")
    runner = os.environ.get("GITHUB_RUNNER", "ubuntu-latest")
    os_name = RUNNER_TO_OS.get(runner, 'linux')
    device = os.environ.get("GITHUB_DEVICE", "cpu")
    backend = os.environ.get("GITHUB_BACKEND")

    if os_name == "macos" and device == "gpu":
        print(f"[SKIP] Skipping for {os_name}/gpu (unsupported).")
        sys.exit(0)

    if backend:
        package_backend(backend, version, device, os_name)
    else:
        for b in BACKENDS:
            package_backend(b, version, device, os_name)
