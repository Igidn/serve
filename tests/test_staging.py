from __future__ import annotations

import os
from pathlib import Path

from serve import staging


def make_checkpoint(root: Path) -> Path:
    src = root / "mount"
    (src / "sub").mkdir(parents=True)
    (src / "config.json").write_text("{}")
    (src / "model-00001.safetensors").write_bytes(bytes(range(256)) * 40)
    (src / "sub" / "tokenizer.json").write_text("x" * 100)
    (src / "weights.gguf").write_bytes(b"not copied")
    return src


def test_copy_file_parallel_chunks_match(tmp_path) -> None:
    src = tmp_path / "in.bin"
    src.write_bytes(os.urandom(100_000))
    dst = tmp_path / "out.bin"
    seen = []

    staging.copy_file(src, dst, seen.append, workers=4, chunk=4096)

    assert dst.read_bytes() == src.read_bytes()
    assert sum(seen) == 100_000
    assert not (tmp_path / "out.bin.part").exists()


def test_stage_copies_checkpoint_and_marks_it(tmp_path) -> None:
    src = make_checkpoint(tmp_path)
    dest = tmp_path / "models" / "qwen"
    progress = []

    staging.stage(src, dest, lambda done, total: progress.append((done, total)))

    assert (dest / "model-00001.safetensors").read_bytes() == (
        src / "model-00001.safetensors"
    ).read_bytes()
    assert (dest / "sub" / "tokenizer.json").read_text() == "x" * 100
    assert not (dest / "weights.gguf").exists()
    assert progress[-1][0] == progress[-1][1]
    assert staging.ready(dest)
    assert (dest / staging.MARKER).is_file()


def test_ready_rejects_truncated_copy(tmp_path) -> None:
    src = make_checkpoint(tmp_path)
    dest = tmp_path / "models" / "qwen"
    staging.stage(src, dest)
    assert staging.ready(dest) is True

    (dest / "model-00001.safetensors").write_bytes(b"short")
    assert staging.ready(dest) is False


def test_ready_rejects_partial_staging(tmp_path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "config.json").write_text("{}")
    (dest / staging.PARTIAL).write_text("staging\n")
    assert staging.ready(dest) is False


def test_ready_accepts_plain_hugging_face_download(tmp_path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "config.json").write_text("{}")
    assert staging.ready(dest) is True


def test_stage_skips_files_already_copied(tmp_path, monkeypatch) -> None:
    src = make_checkpoint(tmp_path)
    dest = tmp_path / "models" / "qwen"
    staging.stage(src, dest)
    (dest / staging.MARKER).unlink()

    copies = []
    monkeypatch.setattr(staging, "copy_file", lambda *a, **k: copies.append(a))

    progress = []
    staging.stage(src, dest, lambda done, total: progress.append((done, total)))

    assert copies == []
    assert progress[-1][0] == progress[-1][1]
    assert staging.ready(dest)


def test_local_disk_is_not_a_network_mount(tmp_path) -> None:
    assert staging.is_network_mount(tmp_path) is False
