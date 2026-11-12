#!/usr/bin/env python
"""Kaggle launcher for serve.

Push with:  kaggle kernels push -p .kaggle/
The kernel clones the repo, installs the dashboard dependencies, injects
Kaggle Secrets, then hands the process to ``python -m serve.main``.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_URL = "https://github.com/Igidn/serve.git"
BRANCH = "main"
WORK = "/kaggle/working/serve"
MODEL_DATASET = "katherinameyerkat/qwen3-8-27b-gptq-4bit"
MODEL_MOUNTS = (
    "/kaggle/input/datasets/katherinameyerkat/qwen3-8-27b-gptq-4bit",
    "/kaggle/input/qwen3-8-27b-gptq-4bit",
)


def run(argv: list[str], **kwargs) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, check=True, **kwargs)


def clone() -> None:
    if os.path.isdir(os.path.join(WORK, ".git")):
        run(["git", "-C", WORK, "fetch", "--depth", "1", "origin", BRANCH])
        run(["git", "-C", WORK, "reset", "--hard", f"origin/{BRANCH}"])
    else:
        run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, WORK])


def secret(name: str) -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def main() -> None:
    print(f"serve launcher: python={sys.version.split()[0]}", flush=True)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except OSError:
        pass

    clone()
    os.chdir(WORK)
    sys.path.insert(0, WORK)

    mounted = next(
        (path for path in MODEL_MOUNTS if os.path.isfile(os.path.join(path, "config.json"))),
        None,
    )
    print(
        f"model dataset {MODEL_DATASET}: "
        f"{'mounted at ' + mounted if mounted else 'NOT mounted'}",
        flush=True,
    )

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--quiet",
            "-r",
            os.path.join(WORK, "requirements.txt"),
        ]
    )

    env = dict(os.environ)
    env["SERVE_WORK_DIR"] = "/kaggle/working"
    env["PYTHONUNBUFFERED"] = "1"

    hf = secret("HUGGINGFACE_TOKEN")
    if hf:
        env["HF_TOKEN"] = hf
        env["HUGGINGFACE_TOKEN"] = hf
        print("injected HUGGINGFACE_TOKEN", flush=True)

    identity = secret("OPENTUNNEL_IDENTITY_B64")
    if identity:
        env["OPENTUNNEL_IDENTITY_B64"] = identity
        print("restoring the saved opentunnel identity", flush=True)

    idle = secret("SERVE_IDLE_TIMEOUT")
    if idle:
        env["SERVE_IDLE_TIMEOUT"] = str(idle)
        print(f"idle timeout override: {idle}s", flush=True)

    # Keep this process alive. Replacing the Jupyter kernel with os.exec* makes
    # Kaggle think the notebook died and restart it, which kills the gateway.
    # The child exits when the idle watchdog or /api/shutdown ends the run;
    # then this parent exits too, so Kaggle stops metering GPU time.
    argv = [sys.executable, "-m", "serve.main"]
    print("+", " ".join(argv), flush=True)
    try:
        code = subprocess.call(argv, env=env)
    except KeyboardInterrupt:
        code = 130
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
