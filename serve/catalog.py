from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


KINDS = ("chat", "image")
ENGINES = ("vllm", "vllm-omni")


@dataclass
class ModelSpec:
    """A model the gateway knows how to serve.

    ``size_gb`` is an estimate used for the disk preflight. The downloader
    asks Hugging Face for the real size and refuses to start when it will not
    fit, so a stale estimate cannot fill the disk silently.
    """

    key: str
    label: str
    repo: str
    kind: str
    engine: str
    size_gb: float = 0.0
    tensor_parallel: int = 1
    dtype: str | None = "float16"
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    trust_remote_code: bool = False
    extra_args: list[str] = field(default_factory=list)
    local_paths: list[str] = field(default_factory=list)
    min_vram_gb: float = 0.0
    notes: str = ""
    builtin: bool = True

    def validate(self) -> "ModelSpec":
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", self.key):
            raise ValueError("key must be lowercase letters, numbers, dot, dash or underscore")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.repo):
            raise ValueError("repo must look like 'owner/name'")
        if not 1 <= self.tensor_parallel <= 8:
            raise ValueError("tensor_parallel must be between 1 and 8")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}
        spec = cls(**{k: v for k, v in data.items() if k in known})
        spec.builtin = bool(data.get("builtin", False))
        spec.extra_args = [str(a) for a in (spec.extra_args or [])]
        spec.local_paths = [str(p) for p in (spec.local_paths or [])]
        return spec.validate()


def _t4_chat(**kwargs) -> ModelSpec:
    # T4 is compute capability 7.5: float16, no bf16, no FP8.
    return ModelSpec(dtype="float16", **kwargs)


PRESETS: tuple[ModelSpec, ...] = (
    _t4_chat(
        key="qwen3.8-27b-gptq",
        label="Qwen3.8 27B (GPTQ INT4)",
        repo="btbtyler09/Qwen3.8-27B-GPTQ-4bit",
        kind="chat",
        engine="vllm",
        size_gb=21.02,
        min_vram_gb=28.0,
        tensor_parallel=2,
        max_model_len=16384,
        gpu_memory_utilization=0.9,
        extra_args=[
            "--enforce-eager",
            "--skip-mm-profiling",
            "--limit-mm-per-prompt",
            '{"image": 2}',
            # The chat template renders tool calls as <function=name> XML, so
            # the XML parser is the one that turns them into tool_calls.
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
            "--reasoning-parser",
            "qwen3",
            # Thinking defaults to xhigh, which on two T4s means minutes of
            # <think> before the first useful token. Agent harnesses want it
            # off unless the request asks for it.
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
        ],
        local_paths=[
            "/kaggle/input/datasets/katherinameyerkat/qwen3-8-27b-gptq-4bit",
            "/kaggle/input/qwen3-8-27b-gptq-4bit",
        ],
        notes=(
            "21 GB of weights: it does not fit in Kaggle's 20 GB working disk. "
            "Attach it as a Kaggle Dataset (auto-detected under /kaggle/input) "
            "or it will refuse to download. GPTQ 4-bit group size 32 without "
            "act-order is supported on Turing; vLLM selects the GPTQ kernel."
        ),
    ),
    _t4_chat(
        key="qwen3-8b-awq",
        label="Qwen3 8B (AWQ) — smoke test",
        repo="Qwen/Qwen3-8B-AWQ",
        kind="chat",
        engine="vllm",
        size_gb=5.6,
        min_vram_gb=9.0,
        max_model_len=32768,
        gpu_memory_utilization=0.9,
        notes="Small enough to download inside the kernel. Useful to verify the tunnel and API.",
    ),
    ModelSpec(
        key="sdxl-base",
        label="Stable Diffusion XL",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        kind="image",
        engine="vllm-omni",
        size_gb=6.9,
        min_vram_gb=12.0,
        dtype="float16",
        notes="Text-to-image via /v1/images/generations. Fits one T4 comfortably.",
    ),
    ModelSpec(
        key="sd35-medium",
        label="Stable Diffusion 3.5 Medium",
        repo="stabilityai/stable-diffusion-3.5-medium",
        kind="image",
        engine="vllm-omni",
        size_gb=11.0,
        min_vram_gb=14.0,
        dtype="float16",
        notes="Text-to-image. Tighter fit on a 16 GB T4.",
    ),
    ModelSpec(
        key="z-image-turbo",
        label="Z-Image Turbo",
        repo="Tongyi-MAI/Z-Image-Turbo",
        kind="image",
        engine="vllm-omni",
        size_gb=12.0,
        min_vram_gb=20.0,
        dtype="float16",
        notes="Few-step 1024px model. Its recipe peaks near 20 GB, so a single T4 is unlikely to hold it.",
    ),
    ModelSpec(
        key="flux2-klein-4b",
        label="FLUX.2 klein 4B",
        repo="black-forest-labs/FLUX.2-klein-4B",
        kind="image",
        engine="vllm-omni",
        size_gb=8.5,
        min_vram_gb=16.0,
        dtype="float16",
        notes="Text-to-image and editing. 16 GB is the lower bound; expect a tight fit.",
    ),
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


class Catalog:
    def __init__(self, custom_file: Path):
        self.custom_file = custom_file
        self._custom: dict[str, ModelSpec] = {}
        self.reload_custom()

    def reload_custom(self) -> None:
        self._custom = {}
        try:
            data = json.loads(self.custom_file.read_text())
        except (OSError, ValueError):
            data = []
        for item in data if isinstance(data, list) else []:
            try:
                self._custom[item["key"]] = ModelSpec.from_dict(item)
            except (KeyError, TypeError, ValueError):
                continue

    def save_custom(self) -> None:
        payload = [spec.to_dict() for spec in self._custom.values()]
        self.custom_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.custom_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.custom_file)

    def all(self) -> list[ModelSpec]:
        return [*PRESETS, *self._custom.values()]

    def get(self, key: str) -> ModelSpec:
        for spec in self.all():
            if spec.key == key:
                return spec
        raise KeyError(key)

    def add_custom(self, spec: ModelSpec) -> ModelSpec:
        if any(existing.key == spec.key for existing in self.all()):
            raise ValueError(f"a model with key '{spec.key}' already exists")
        spec.builtin = False
        self._custom[spec.key] = spec
        self.save_custom()
        return spec

    def remove_custom(self, key: str) -> None:
        if key not in self._custom:
            raise ValueError(f"'{key}' is not a custom model")
        del self._custom[key]
        self.save_custom()

    def local_source(self, spec: ModelSpec) -> Path | None:
        for raw in spec.local_paths:
            path = Path(raw)
            if (path / "config.json").is_file():
                return path
        return self._scan_kaggle_input(spec.repo)

    def download_dir(self, spec: ModelSpec, models_dir: Path) -> Path:
        return models_dir / spec.key

    def source_for(self, spec: ModelSpec, models_dir: Path) -> tuple[str, Path | None]:
        """Return the value to hand to vLLM and, if local, its path.

        A local/Kaggle-input copy wins over a downloaded one so attaching a
        dataset avoids the download entirely.
        """
        local = self.local_source(spec)
        if local is not None:
            return str(local), local
        cached = self.download_dir(spec, models_dir)
        if (cached / "config.json").is_file():
            return str(cached), cached
        return spec.repo, None

    @staticmethod
    def _scan_kaggle_input(repo: str) -> Path | None:
        root = Path("/kaggle/input")
        if not root.is_dir():
            return None
        wanted = _slug(repo.split("/", 1)[-1]).replace("qwen3-", "qwen3")
        flattened = _slug(repo)
        for candidate in root.glob("*/"):
            try:
                if not (candidate / "config.json").is_file():
                    continue
            except OSError:
                continue
            name = _slug(candidate.name)
            if name in {wanted, flattened} or wanted in name or name in wanted:
                return candidate
        # Older kernels mount datasets at /kaggle/input/<slug>; newer ones use
        # /kaggle/input/datasets/<owner>/<slug>.
        for candidate in (*root.glob("*/*/"), *root.glob("datasets/*/*/")):
            try:
                if (candidate / "config.json").is_file():
                    name = _slug(candidate.name)
                    if wanted in name or name in wanted or flattened in _slug(str(candidate)):
                        return candidate
            except OSError:
                continue
        return None


def vllm_executable(python: str = sys.executable) -> str | None:
    candidate = Path(python).with_name("vllm")
    if candidate.is_file():
        return str(candidate)
    return shutil.which("vllm")


def build_argv(spec: ModelSpec, source: str, port: int, python: str = sys.executable) -> list[str]:
    executable = vllm_executable(python)
    if executable:
        argv = [executable, "serve", source]
    elif spec.engine == "vllm-omni":
        raise RuntimeError(
            "vLLM-Omni needs the 'vllm' command installed in the same environment "
            "(the module entry point does not accept --omni)"
        )
    else:
        argv = [python, "-m", "vllm.entrypoints.openai.api_server", source]

    argv += ["--host", "127.0.0.1", "--port", str(port)]
    if spec.engine == "vllm-omni":
        argv.append("--omni")
    if spec.engine == "vllm" and spec.kind == "chat":
        argv += ["--served-model-name", spec.key]
    if spec.tensor_parallel > 1:
        argv += ["--tensor-parallel-size", str(spec.tensor_parallel)]
    if spec.dtype:
        argv += ["--dtype", spec.dtype]
    if spec.kind == "chat" and spec.max_model_len:
        argv += ["--max-model-len", str(spec.max_model_len)]
    if spec.kind == "chat" and spec.gpu_memory_utilization:
        argv += ["--gpu-memory-utilization", f"{spec.gpu_memory_utilization:g}"]
    if spec.trust_remote_code:
        argv.append("--trust-remote-code")
    argv += list(spec.extra_args)
    return argv
