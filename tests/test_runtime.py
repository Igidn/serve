from __future__ import annotations

import logging
import sys

from serve.runtime import RuntimeManager
from serve.util import exception_summary

CRASH = """\
engine exited with code 1
Traceback (most recent call last):
  File "/usr/local/lib/python3.13/dist-packages/vllm/entrypoints/cli/main.py", line 43, in main
    import vllm.entrypoints.cli.benchmark.main
  File "/usr/local/lib/python3.13/dist-packages/vllm/config/model.py", line 21, in <module>
    from vllm.config.multimodal import (
ImportError: cannot import name 'MMTensorIPC' from 'vllm.config.multimodal'
"""


def test_exception_summary_keeps_the_cause() -> None:
    assert exception_summary(CRASH) == (
        "ImportError: cannot import name 'MMTensorIPC' from 'vllm.config.multimodal'"
    )


def test_exception_summary_handles_plain_errors() -> None:
    assert exception_summary("engine exited with code 1\n") == "engine exited with code 1"


def test_exception_summary_keeps_one_line_messages() -> None:
    message = "this model needs 21.0 GiB but only 3.0 GiB is free."
    assert exception_summary(message) == message


def test_reinstall_adds_reinstall_flag(cfg) -> None:
    runtime = RuntimeManager(cfg, logging.getLogger("test"))
    plain = runtime._commands("vllm")
    forced = runtime._commands("vllm", reinstall=True)
    assert len(plain) == 1
    assert "--reinstall" not in plain[0]
    assert f"vllm=={cfg.vllm_version}" in plain[0]

    # A reinstall first removes the old files, then installs with --reinstall.
    assert forced[0][:5] == [sys.executable, "-m", "pip", "uninstall", "-y"]
    assert "vllm" in forced[0]
    assert "--reinstall" in forced[-1]
    assert f"vllm=={cfg.vllm_version}" in forced[-1]
