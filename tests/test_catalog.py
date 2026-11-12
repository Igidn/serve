from __future__ import annotations

import pytest

from serve.catalog import PRESETS, Catalog, ModelSpec, build_argv


def test_every_preset_is_valid() -> None:
    for spec in PRESETS:
        spec.validate()


def test_chat_command_carries_t4_settings(monkeypatch) -> None:
    monkeypatch.setattr("serve.catalog.shutil.which", lambda name: "/usr/local/bin/vllm")
    spec = next(s for s in PRESETS if s.key == "qwen3.8-27b-awq")
    argv = build_argv(spec, "some/path", 8001, python="/usr/bin/python3")
    joined = " ".join(argv)

    assert argv[:3] == ["/usr/local/bin/vllm", "serve", "some/path"]
    assert "--tensor-parallel-size 2" in joined
    assert "--dtype float16" in joined
    assert "--max-model-len 16384" in joined
    assert "--served-model-name qwen3.8-27b-awq" in joined
    assert "--enforce-eager" in argv
    assert "--omni" not in argv


def test_image_command_uses_omni_without_served_name(monkeypatch) -> None:
    monkeypatch.setattr("serve.catalog.shutil.which", lambda name: "/usr/local/bin/vllm")
    spec = next(s for s in PRESETS if s.key == "sdxl-base")
    argv = build_argv(spec, "stabilityai/stable-diffusion-xl-base-1.0", 8001)
    joined = " ".join(argv)

    assert "--omni" in argv
    assert "--served-model-name" not in joined
    assert "--max-model-len" not in joined


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError):
        ModelSpec(
            key="thing",
            label="Thing",
            repo="owner/thing",
            kind="chat",
            engine="ollama",
        ).validate()


def test_custom_models_round_trip(tmp_path) -> None:
    path = tmp_path / "custom.json"
    catalog = Catalog(path)
    spec = ModelSpec(
        key="my-model",
        label="My model",
        repo="owner/model",
        kind="image",
        engine="vllm-omni",
        extra_args=["--enforce-eager"],
    )
    catalog.add_custom(spec)

    assert catalog.get("my-model").key == "my-model"
    assert "my-model" in path.read_text()
    assert Catalog(path).get("my-model").extra_args == ["--enforce-eager"]

    catalog.remove_custom("my-model")
    assert all(spec.key != "my-model" for spec in Catalog(path).all())


def test_custom_key_collision_is_refused(tmp_path) -> None:
    catalog = Catalog(tmp_path / "custom.json")
    catalog.add_custom(
        ModelSpec(key="x", label="X", repo="a/b", kind="chat", engine="vllm")
    )
    with pytest.raises(ValueError):
        catalog.add_custom(
            ModelSpec(key="x", label="X2", repo="a/c", kind="chat", engine="vllm")
        )
