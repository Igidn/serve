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
MODEL_MOUNT = "/kaggle/input/qwen3-8-27b-gptq-4bit"


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

    mounted = os.path.isfile(os.path.join(MODEL_MOUNT, "config.json"))
    print(
        f"model dataset {MODEL_DATASET}: "
        f"{'mounted at ' + MODEL_MOUNT if mounted else 'NOT mounted'}",
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

    os.execvpe(sys.executable, [sys.executable, "-m", "serve.main"], env)


if __name__ == "__main__":
    main()
