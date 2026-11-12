from __future__ import annotations

from serve import llamacpp


def test_cuda_major_comes_from_the_build_flavour(cfg) -> None:
    assert llamacpp.cuda_major(cfg) == "12"


def test_cuda_libraries_need_the_matching_major(tmp_path) -> None:
    libs = tmp_path / "lib"
    libs.mkdir()
    # A cu13 image is not good enough for a build linked against 12.8.
    (libs / "libcudart.so.13").write_text("")
    (libs / "libcublas.so.13").write_text("")
    assert llamacpp.cuda_libraries_in([libs], "12") is False

    (libs / "libcudart.so.12.8.90").write_text("")
    (libs / "libcublas.so.12").write_text("")
    assert llamacpp.cuda_libraries_in([libs], "12") is True


def test_cuda_libraries_need_both_runtime_and_blas(tmp_path) -> None:
    libs = tmp_path / "lib"
    libs.mkdir()
    (libs / "libcudart.so.12").write_text("")
    assert llamacpp.cuda_libraries_in([libs], "12") is False
    (libs / "libcublas.so.12").write_text("")
    assert llamacpp.cuda_libraries_in([libs], "12") is True


def test_release_url_points_at_the_pinned_build() -> None:
    assert llamacpp.release_url("b12318", "x.tar.gz") == (
        "https://github.com/ggml-org/llama.cpp/releases/download/b12318/x.tar.gz"
    )
