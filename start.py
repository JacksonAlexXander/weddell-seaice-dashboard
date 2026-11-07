#!/usr/bin/env python3
"""Create a project-local environment, install dependencies, then run the dashboard."""
from pathlib import Path
import argparse
import hashlib
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "dashboard/dist/index.html",
    "data/processed/weddell_icemotion_daily.zarr/.zgroup",
    "data/processed/dashboard/metrics.json",
    "data/processed/dashboard/climatology.npz",
    "data/processed/dashboard/simvp_day1_fields.npz",
    "models/ml/ridge.joblib", "models/ml/rf.joblib",
    "models/simvp-24h/ckpt_best.pt",
]

def main():
    ap = argparse.ArgumentParser(description="Weddell Sea Ice Dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--setup-only", action="store_true")
    args = ap.parse_args()
    if sys.version_info[:2] != (3, 13):
        print("Please install Python 3.13 (64-bit): https://www.python.org/downloads/")
        return 1
    if not 1 <= args.port <= 65535:
        print("Port must be between 1 and 65535.")
        return 1
    missing = [f for f in REQUIRED if not (ROOT / f).is_file()]
    if missing:
        print("Missing runtime assets. Verify the data, model weights and dashboard build are present.")
        print("\n".join(missing))
        return 1
    env_dir = ROOT / ".venv"
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(ROOT / ".cache/pip")
    env["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    env["TORCH_HOME"] = str(ROOT / ".cache/torch")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    os.chdir(ROOT)
    if not py.exists():
        print("Creating Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(env_dir)
    requirements = ROOT / "requirements.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = env_dir / ".dashboard-requirements"
    if not marker.exists() or marker.read_text().strip() != fingerprint:
        print("Installing dependencies (first launch requires internet and may take several minutes)...", flush=True)
        subprocess.run([str(py), "-m", "pip", "install", "-r", str(requirements)], check=True, env=env)
        marker.write_text(fingerprint)
    subprocess.run([str(py), "-c", "import numpy,scipy,sklearn,joblib,zarr,numcodecs,pandas,pyarrow,torch,fastapi,uvicorn"], check=True, env=env)
    if args.setup_only:
        print("Setup complete.")
        return 0
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", args.port))
        except OSError:
            print(f"Port {args.port} is busy. Retry with --port {args.port + 1}.")
            return 1
    url = f"http://127.0.0.1:{args.port}/"
    proc = subprocess.Popen([str(py), str(ROOT / "scripts/serve_dashboard.py"), "--port", str(args.port), "--host", "127.0.0.1"], cwd=ROOT, env=env)
    if not args.no_browser:
        def open_when_ready():
            for _ in range(120):
                if proc.poll() is not None:
                    return
                try:
                    with urllib.request.urlopen(url + "api/health", timeout=2) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            return
                except OSError:
                    time.sleep(1)
        threading.Thread(target=open_when_ready, daemon=True).start()
    print(f"Dashboard: {url}  |  Keep this window open. Ctrl+C stops the server.", flush=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Setup failed (exit {exc.returncode}). Check your network and retry.")
        raise SystemExit(exc.returncode)
