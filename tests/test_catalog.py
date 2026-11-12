from __future__ import annotations

import pytest

from serve.catalog import PRESETS, Catalog, ModelSpec, build_argv


def test_every_preset_is_valid() -> None:
    for spec in PRESETS:
        spec.validate()


def test_chat_command_carries_t4_settings(monkeypatch) -> None:
    monkeypatch.setattr("serve.catalog.shutil.which", lambda name: "/usr/local/bin/vllm")
    spec = next(s for s in PRESETS if s.key == "qwen3.8-27b-gptq")
    argv = build_argv(spec, "some/path", 8001, python="/usr/bin/python3")
    joined = " ".join(argv)

    assert argv[:3] == ["/usr/local/bin/vllm", "serve", "some/path"]
    assert "--tensor-parallel-size 2" in joined
    assert "--dtype float16" in joined
    assert "--max-model-len 16384" in joined
    assert "--served-model-name qwen3.8-27b-gptq" in joined
    assert "--enforce-eager" in argv
    assert "--omni" not in argv
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert argv[argv.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking": false}'
    )


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


def test_llamacpp_command_uses_the_gguf_and_jinja(tmp_path) -> None:
    state = tmp_path / "serve-state"
    binary = state / "runtime" / "llamacpp" / "b12318" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "Qwen3.8-27B-UD-Q4_K_M.gguf").write_bytes(b"x" * 64)
    (weights / "mmproj-F16.gguf").write_bytes(b"x" * 8)

    spec = next(s for s in PRESETS if s.key == "qwen3.8-27b-gguf")
    argv = build_argv(spec, str(weights), 8001, state_dir=state)

    assert argv[0] == str(binary)
    assert argv[argv.index("--model") + 1] == str(weights / "Qwen3.8-27B-UD-Q4_K_M.gguf")
    assert argv[argv.index("--mmproj") + 1] == str(weights / "mmproj-F16.gguf")
    assert argv[argv.index("--ctx-size") + 1] == "131072"
    assert argv[argv.index("--tensor-split") + 1] == "1,1"
    assert argv[argv.index("--alias") + 1] == "qwen3.8-27b-gguf"
    assert argv[argv.index("--chat-template-kwargs") + 1] == '{"enable_thinking": false}'
    assert "--jinja" in argv
    assert "--served-model-name" not in argv


def test_llamacpp_without_a_binary_explains_itself(tmp_path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.gguf").write_bytes(b"x")
    spec = next(s for s in PRESETS if s.key == "qwen3.8-27b-gguf")
    with pytest.raises(RuntimeError, match="not installed"):
        build_argv(spec, str(weights), 8001, state_dir=tmp_path / "serve-state")


def test_gguf_counts_as_a_checkpoint(tmp_path) -> None:
    spec = next(s for s in PRESETS if s.key == "qwen3.8-27b-gguf")
    empty = tmp_path / "empty"
    empty.mkdir()
    assert Catalog.is_checkpoint(empty, spec) is False
    (empty / "model.Q4_K_M.gguf").write_bytes(b"x")
    assert Catalog.is_checkpoint(empty, spec) is True
    # A Python checkpoint still needs config.json, not a stray .gguf.
    vllm = next(s for s in PRESETS if s.key == "qwen3.8-27b-gptq")
    assert Catalog.is_checkpoint(empty, vllm) is False


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
