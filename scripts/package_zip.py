import os, zipfile, platform, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["face", "voice"]
RUNNER_TO_OS = {
    'ubuntu-latest': 'linux',
    'windows-latest': 'windows',
    'macos-latest': 'macos'
}

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
    print(f"[DEBUG] Writing start files to {build_dir}")
    start_sh = build_dir / f"start_{backend_name}.sh"
    start_bat = build_dir / f"start_{backend_name}.bat"
    start_sh.write_text(f"#!/bin/bash\ncd $(dirname $0)\nchmod +x ./{binary_name}\n./{binary_name}\n")
    start_bat.write_text(f"@echo off\ncd %~dp0\n{binary_name}.exe\n")
    print(f"[DEBUG] Wrote {start_sh} and {start_bat}")

def write_version_file(build_dir, version):
    print(f"[DEBUG] Writing VERSION file to {build_dir}")
    (build_dir / "VERSION").write_text(version)
    print(f"[DEBUG] Wrote VERSION: {version}")
    
IGNORE_DIRS = {"venv", "__pycache__", "tests", "models", "rvc_models"}

def copy_backend_files(backend_dir, build_dir, backend):
    print(f"[DEBUG] Copying files from {backend_dir} to {build_dir} (ignoring models/tests/venv)")
    for item in backend_dir.iterdir():
        if item.name in IGNORE_DIRS:
            print(f"[SKIP] {item} (ignored)")
            continue
        target = build_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
            print(f"[DEBUG] Copied directory {item} -> {target}")
        else:
            shutil.copy(item, target)
            print(f"[DEBUG] Copied file {item} -> {target}")

def package_backend(backend, version, device, os_name):
    print(f"[DEBUG] Packaging {backend} for {os_name}/{device}, version: {version}")
    print(f"[DEBUG] ROOT directory: {ROOT}")
    backend_dir = ROOT / "backends" / backend
    build_dir = ROOT / f"build_{backend}_{os_name}_{device}"
    
    print(f"[DEBUG] Backend dir: {backend_dir}, Build dir: {build_dir}")
    if build_dir.exists():
        print(f"[DEBUG] Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    copy_backend_files(backend_dir, build_dir, backend)

    venv_dir = build_dir / "venv"
    print(f"[DEBUG] Creating venv at {venv_dir}")
    run(f"\"{sys.executable}\" -m venv \"{venv_dir}\"")
    
    actual_os = platform.system().lower()
    if actual_os == "windows":
        pip = venv_dir / "Scripts" / "pip.exe"
        pyinstaller = venv_dir / "Scripts" / "pyinstaller.exe"
    else:
        pip = venv_dir / "bin" / "pip"
        pyinstaller = venv_dir / "bin" / "pyinstaller"
    print(f"[DEBUG] Actual OS: {actual_os}, pip: {pip}, pyinstaller: {pyinstaller}")

    run(f"\"{pip}\" install pyinstaller")
    
    run(f'"{pyinstaller}" --version')

    requirements_file = backend_dir / "requirements.txt"
    print(f"[DEBUG] Installing requirements from {requirements_file}")
    run(f"\"{pip}\" install -r \"{requirements_file}\" --no-cache-dir")

    if backend == "voice":
        if device == "cpu":
            run(f"\"{pip}\" install torch==2.0.1")
            run(f"\"{pip}\" install fairseq==0.12.2 faiss-cpu==1.7.3")
        else:
            if actual_os == "windows" or actual_os == "linux":
                run(f"\"{pip}\" install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir")
                run(f"\"{pip}\" install fairseq==0.12.2 faiss-gpu==1.7.4")
            else:
                print(f"[SKIP] Skipping GPU dependencies for {backend} on {actual_os} (unsupported)")
                return
    elif backend == "face":
        if device == "cpu":
            run(f"\"{pip}\" install torch torchvision torchaudio")
            run(f"\"{pip}\" install onnxruntime")
        else:
            if actual_os == "windows" or actual_os == "linux":
                run(f"\"{pip}\" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                run(f"\"{pip}\" install onnxruntime-gpu")
            else:
                print(f"[SKIP] Skipping GPU dependencies for {backend} on {actual_os} (unsupported)")
                return

    entry_script = build_dir / f"{backend}_main.py"
    binary_name = f"{backend}_backend"
    print(f"[DEBUG] Running PyInstaller on {entry_script} to create {binary_name}")
    run(f"\"{pyinstaller}\" --onefile --name {binary_name} \"{entry_script}\"", cwd=build_dir)

    dist_dir = build_dir / "dist"
    final_build_dir = build_dir / "package"
    print(f"[DEBUG] Creating final build directory: {final_build_dir}")
    final_build_dir.mkdir(parents=True, exist_ok=True)

    binary_suffix = ".exe" if os_name == "windows" else ""
    print(f"[DEBUG] Moving {dist_dir / f'{binary_name}{binary_suffix}'} to {final_build_dir / f'{binary_name}{binary_suffix}'}")
    shutil.move(dist_dir / f"{binary_name}{binary_suffix}", final_build_dir / f"{binary_name}{binary_suffix}")

    if (build_dir / "src").exists():
        print(f"[DEBUG] Copying {build_dir / 'src'} to {final_build_dir / 'src'}")
        shutil.copytree(build_dir / "src", final_build_dir / "src")
    if (build_dir / "models").exists():
        print(f"[DEBUG] Copying {build_dir / 'models'} to {final_build_dir / 'models'}")
        shutil.copytree(build_dir / "models", final_build_dir / "models")
    
    write_version_file(final_build_dir, version)
    write_start_files(final_build_dir, backend, binary_name)

    zip_path = ROOT / f"{backend}-{os_name}-{device}-v{version.lstrip('v')}.zip"
    print(f"[DEBUG] Creating zip: {zip_path}")
    shutil.make_archive(zip_path.with_suffix(''), 'zip', final_build_dir)
    print(f"[DONE] {zip_path}")

if __name__ == "__main__":
    version = os.environ.get("GITHUB_REF_NAME", "dev")
    runner = os.environ.get("GITHUB_RUNNER", "ubuntu-latest")
    os_name = RUNNER_TO_OS.get(runner, 'linux')
    device = os.environ.get("GITHUB_DEVICE", "cpu")
    backend = os.environ.get("GITHUB_BACKEND")  # NEW

    if os_name == "macos" and device == "gpu":
        print(f"[SKIP] Skipping for {os_name}/gpu (unsupported).")
        sys.exit(0)

    if backend:
        try:
            package_backend(backend, version, device, os_name)
        except Exception as e:
            print(f"[FAIL] Failed to package {backend} for {os_name}/{device}: {e}")
    else:
        # Fallback: build all (like before)
        for b in BACKENDS:
            try:
                package_backend(b, version, device, os_name)
            except Exception as e:
                print(f"[FAIL] Failed to package {b} for {os_name}/{device}: {e}")