from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_root() -> Path:
    override = os.environ.get("SERVE_WORK_DIR")
    if override:
        return Path(override).expanduser().resolve()
    kaggle = Path("/kaggle/working")
    if kaggle.is_dir():
        return kaggle
    return (Path.cwd() / ".serve").resolve()


@dataclass(frozen=True)
class Config:
    """Everything the two halves of serve need to know about their host.

    On Kaggle the root is ``/kaggle/working``. On the Raspberry Pi it is a
    local ``.serve`` directory. Models, logs and the opentunnel identity all
    live under ``<root>/serve-state``.
    """

    root: Path
    state_dir: Path
    models_dir: Path
    logs_dir: Path
    hf_home: Path
    engine_port: int
    idle_timeout: float
    vllm_version: str
    omni_version: str
    llamacpp_build: str
    llamacpp_cuda: str
    hf_token: str | None
    allow_exit: bool
    max_model_len_cap: int
    stage_models: bool

    # The model the Kaggle node installs and serves without a dashboard.
    model_key: str
    # The Pi's persistent opentunnel WebSocket URL (wss://.../kaggle).
    relay_url: str
    # Optional shared secret for the public WebSocket endpoint. This is not
    # the harness API key; the harness talks to the Pi over the LAN only.
    relay_token: str | None

    # Pi-side listeners: HTTP for the local harness, WebSocket for Kaggle.
    pi_http_host: str
    pi_http_port: int
    pi_ws_host: str
    pi_ws_port: int
    pi_ws_path: str
    pi_tunnel: bool
    pi_tunnel_name: str
    pi_tunnel_url_file: Path

    @classmethod
    def from_env(cls) -> "Config":
        root = _default_root()
        state = root / "serve-state"
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
        on_kaggle = Path("/kaggle/working").is_dir()
        return cls(
            root=root,
            state_dir=state,
            models_dir=state / "models",
            logs_dir=state / "logs",
            hf_home=Path(_str("HF_HOME", str(state / "hf"))).expanduser(),
            engine_port=_int("SERVE_ENGINE_PORT", 8001),
            idle_timeout=_float("SERVE_IDLE_TIMEOUT", 600.0),
            vllm_version=_str("SERVE_VLLM_VERSION", "0.32.0"),
            omni_version=_str("SERVE_OMNI_VERSION", "0.32.0"),
            llamacpp_build=_str("SERVE_LLAMACPP_BUILD", "b12318"),
            llamacpp_cuda=_str("SERVE_LLAMACPP_CUDA", "cuda-12.8"),
            hf_token=hf_token,
            allow_exit=_bool("SERVE_ALLOW_EXIT", on_kaggle),
            max_model_len_cap=_int("SERVE_MAX_MODEL_LEN", 32768),
            stage_models=_bool("SERVE_STAGE_MODELS", True),
            model_key=_str("SERVE_MODEL", "qwen3.8-27b-gguf"),
            relay_url=_str("SERVE_RELAY_URL", "ws://127.0.0.1:8787/kaggle"),
            relay_token=os.environ.get("SERVE_RELAY_TOKEN") or None,
            pi_http_host=_str("SERVE_PI_HTTP_HOST", "0.0.0.0"),
            pi_http_port=_int("SERVE_PI_HTTP_PORT", 8080),
            pi_ws_host=_str("SERVE_PI_WS_HOST", "0.0.0.0"),
            pi_ws_port=_int("SERVE_PI_WS_PORT", 8787),
            pi_ws_path=_str("SERVE_PI_WS_PATH", "/kaggle"),
            pi_tunnel=_bool("SERVE_PI_TUNNEL", True),
            pi_tunnel_name=_str("SERVE_PI_TUNNEL_NAME", "serve-relay"),
            pi_tunnel_url_file=Path(
                _str("SERVE_PI_TUNNEL_URL_FILE", str(state / "pi-tunnel-url.txt"))
            ).expanduser(),
        )

    # -- derived paths -----------------------------------------------------

    @property
    def custom_models_file(self) -> Path:
        return self.state_dir / "custom-models.json"

    @property
    def opentunnel_dir(self) -> Path:
        return self.state_dir / "opentunnel"

    @property
    def opentunnel_data(self) -> Path:
        return self.opentunnel_dir / "data"

    @property
    def opentunnel_config(self) -> Path:
        return self.opentunnel_dir / "config"

    @property
    def opentunnel_state(self) -> Path:
        return self.opentunnel_dir / "state"

    @property
    def opentunnel_runtime(self) -> Path:
        return self.opentunnel_dir / "runtime"

    def ensure(self) -> "Config":
        for path in (
            self.state_dir,
            self.models_dir,
            self.logs_dir,
            self.hf_home,
            self.opentunnel_data,
            self.opentunnel_config,
            self.opentunnel_state,
            self.opentunnel_runtime,
        ):
            path.mkdir(parents=True, exist_ok=True)
        # opentunnel refuses a runtime directory that is not private.
        try:
            self.opentunnel_runtime.chmod(0o700)
        except OSError:
            pass
        return self

    def opentunnel_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "XDG_DATA_HOME": str(self.opentunnel_data),
                "XDG_CONFIG_HOME": str(self.opentunnel_config),
                "XDG_STATE_HOME": str(self.opentunnel_state),
                "XDG_RUNTIME_DIR": str(self.opentunnel_runtime),
                "PATH": os.pathsep.join(
                    [str(Path.home() / ".local" / "bin"), env.get("PATH", "")]
                ),
            }
        )
        return env
