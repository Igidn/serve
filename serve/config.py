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
    """Everything the gateway needs to know about its environment.

    On Kaggle the root is ``/kaggle/working``. Models, logs and the tunnel
    identity all live under ``<root>/serve-state`` so they survive until the
    kernel run ends and show up in the kernel's output files.
    """

    root: Path
    state_dir: Path
    models_dir: Path
    logs_dir: Path
    hf_home: Path
    app_port: int
    engine_port: int
    idle_timeout: float
    route_name: str
    profile: str
    vllm_version: str
    omni_version: str
    hf_token: str | None
    tunnel_enabled: bool
    tunnel_identity: str | None
    allow_exit: bool
    max_model_len_cap: int

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
            app_port=_int("SERVE_PORT", 8000),
            engine_port=_int("SERVE_ENGINE_PORT", 8001),
            idle_timeout=_float("SERVE_IDLE_TIMEOUT", 600.0),
            route_name=_str("SERVE_ROUTE_NAME", "serve"),
            profile=_str("OPENTUNNEL_PROFILE", "default"),
            vllm_version=_str("SERVE_VLLM_VERSION", "0.32.0"),
            omni_version=_str("SERVE_OMNI_VERSION", "0.32.0"),
            hf_token=hf_token,
            tunnel_enabled=_bool("SERVE_TUNNEL", True),
            tunnel_identity=os.environ.get("OPENTUNNEL_IDENTITY_B64")
            or os.environ.get("OPENTUNNEL_IDENTITY")
            or None,
            allow_exit=_bool("SERVE_ALLOW_EXIT", on_kaggle),
            max_model_len_cap=_int("SERVE_MAX_MODEL_LEN", 32768),
        )

    # -- derived paths -----------------------------------------------------

    @property
    def settings_file(self) -> Path:
        return self.state_dir / "settings.json"

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

    @property
    def opentunnel_profile_dir(self) -> Path:
        return self.opentunnel_data / "opentunnel" / self.profile

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
                "OPENTUNNEL_PROFILE": self.profile,
                "PATH": os.pathsep.join(
                    [str(Path.home() / ".local" / "bin"), env.get("PATH", "")]
                ),
            }
        )
        return env
