# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import importlib
import json
import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest


def _clear_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


@pytest.fixture
def importing_module():
    _clear_modules("vllm.triton_utils", "vllm.triton_utils.importing")
    module = importlib.import_module("vllm.triton_utils.importing")
    try:
        yield module
    finally:
        _clear_modules("vllm.triton_utils", "vllm.triton_utils.importing")


class _FakeFileCacheManager:
    def __init__(self, key, override=False, dump=False):
        from triton import knobs

        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = os.path.join(knobs.cache.dump_dir, key)
        elif override:
            self.cache_dir = os.path.join(knobs.cache.override_dir, key)
        else:
            self.cache_dir = os.path.join(knobs.cache.dir, key)
        self.lock_path = os.path.join(self.cache_dir, "lock")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _make_path(self, filename):
        return os.path.join(self.cache_dir, filename)

    def get_file(self, filename):
        path = self._make_path(filename)
        return path if os.path.exists(path) else None

    def put(self, data, filename, binary=True):
        payload = data if isinstance(data, bytes) else str(data).encode("utf-8")
        path = Path(self._make_path(filename))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    def get_group(self, filename):
        path = Path(self._make_path(f"__grp__{filename}"))
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data["child_paths"]

    def put_group(self, filename, group):
        return self.put(json.dumps({"child_paths": group}), f"__grp__{filename}")


@pytest.fixture
def cache_manager_module(monkeypatch, tmp_path):
    _clear_modules(
        "triton",
        "triton.runtime",
        "triton.runtime.cache",
        "vllm.triton_utils",
        "vllm.triton_utils.cache_manager",
        "vllm.triton_utils.importing",
    )

    triton_module = types.ModuleType("triton")
    triton_module.__spec__ = ModuleSpec("triton", loader=None)
    triton_module.knobs = types.SimpleNamespace(
        cache=types.SimpleNamespace(
            dir=str(tmp_path / "triton"),
            dump_dir=str(tmp_path / "dump"),
            override_dir=str(tmp_path / "override"),
        )
    )
    runtime_module = types.ModuleType("triton.runtime")
    runtime_module.__spec__ = ModuleSpec("triton.runtime", loader=None)
    cache_module = types.ModuleType("triton.runtime.cache")
    cache_module.__spec__ = ModuleSpec("triton.runtime.cache", loader=None)
    cache_module.FileCacheManager = _FakeFileCacheManager
    runtime_module.cache = cache_module

    monkeypatch.setitem(sys.modules, "triton", triton_module)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "triton.runtime.cache", cache_module)

    module = importlib.import_module("vllm.triton_utils.cache_manager")
    try:
        yield module
    finally:
        _clear_modules(
            "triton",
            "triton.runtime",
            "triton.runtime.cache",
            "vllm.triton_utils.cache_manager",
        )


def test_importing_sets_hierarchical_cache_manager(monkeypatch, importing_module):
    monkeypatch.delenv("TRITON_CACHE_MANAGER", raising=False)
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")

    importing_module.maybe_set_triton_cache_manager()

    assert os.environ["TRITON_CACHE_MANAGER"] == (
        "vllm.triton_utils.cache_manager:HierarchicalFileCacheManager"
    )


def test_importing_respects_user_cache_manager_override(
    monkeypatch, importing_module
):
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "custom:Manager")
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")

    importing_module.maybe_set_triton_cache_manager()

    assert os.environ["TRITON_CACHE_MANAGER"] == "custom:Manager"


def test_put_group_writes_local_and_shared_manifests(
    monkeypatch, tmp_path, cache_manager_module
):
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")
    monkeypatch.setenv("VLLM_TRITON_LOCAL_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("VLLM_TRITON_SHARED_CACHE_DIR", str(tmp_path / "shared"))

    manager = cache_manager_module.HierarchicalFileCacheManager("TESTKEY")
    local_artifact = manager.put(b"cubin-bytes", "kernel.cubin")
    manager.put_group("kernel.json", {"kernel.cubin": local_artifact})

    local_manifest = json.loads(
        (manager.local_cache_dir / "__grp__kernel.json").read_text()
    )
    shared_manifest = json.loads(
        (manager.shared_cache_dir / "__grp__kernel.json").read_text()
    )

    assert local_manifest["child_paths"]["kernel.cubin"] == local_artifact
    assert shared_manifest["child_paths"]["kernel.cubin"] == str(
        manager.shared_cache_dir / "kernel.cubin"
    )


def test_get_group_materializes_shared_artifacts_to_local(
    monkeypatch, tmp_path, cache_manager_module
):
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")
    monkeypatch.setenv("VLLM_TRITON_LOCAL_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("VLLM_TRITON_SHARED_CACHE_DIR", str(tmp_path / "shared"))

    manager = cache_manager_module.HierarchicalFileCacheManager("TESTKEY")
    shared_artifact = manager.shared_cache_dir / "kernel.cubin"
    shared_artifact.parent.mkdir(parents=True, exist_ok=True)
    shared_artifact.write_bytes(b"shared-cubin")
    (manager.shared_cache_dir / "__grp__kernel.json").write_text(
        json.dumps({"child_paths": {"kernel.cubin": str(shared_artifact)}})
    )

    group = manager.get_group("kernel.json")

    assert group == {
        "kernel.cubin": str(manager.local_cache_dir / "kernel.cubin")
    }
    assert (manager.local_cache_dir / "kernel.cubin").read_bytes() == b"shared-cubin"


def test_get_group_returns_none_for_corrupt_shared_manifest(
    monkeypatch, tmp_path, cache_manager_module
):
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")
    monkeypatch.setenv("VLLM_TRITON_LOCAL_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("VLLM_TRITON_SHARED_CACHE_DIR", str(tmp_path / "shared"))

    manager = cache_manager_module.HierarchicalFileCacheManager("TESTKEY")
    manager.shared_cache_dir.mkdir(parents=True, exist_ok=True)
    (manager.shared_cache_dir / "__grp__kernel.json").write_text("{not-json")

    assert manager.get_group("kernel.json") is None


def test_get_group_treats_estale_as_cache_miss(
    monkeypatch, tmp_path, cache_manager_module
):
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")
    monkeypatch.setenv("VLLM_TRITON_LOCAL_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("VLLM_TRITON_SHARED_CACHE_DIR", str(tmp_path / "shared"))

    manager = cache_manager_module.HierarchicalFileCacheManager("TESTKEY")
    manager.shared_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manager.shared_cache_dir / "__grp__kernel.json"
    manifest_path.write_text(
        json.dumps({"child_paths": {"kernel.cubin": "ignored-for-estale"}})
    )

    original_open = cache_manager_module.Path.open

    def _raise_estale(self, *args, **kwargs):
        if self == manifest_path:
            raise OSError(getattr(errno, "ESTALE", 116), "Stale file handle")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(cache_manager_module.Path, "open", _raise_estale)

    assert manager.get_group("kernel.json") is None


def test_shared_publish_failure_is_best_effort(
    monkeypatch, tmp_path, cache_manager_module
):
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")
    monkeypatch.setenv("VLLM_TRITON_LOCAL_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("VLLM_TRITON_SHARED_CACHE_DIR", str(tmp_path / "shared"))

    manager = cache_manager_module.HierarchicalFileCacheManager("TESTKEY")
    original_atomic_write = cache_manager_module._atomic_write_bytes

    def _fail_shared_write(path, payload):
        if str(path).startswith(str(manager.shared_cache_dir)):
            raise OSError(errno.EIO, "shared write failed")
        return original_atomic_write(path, payload)

    monkeypatch.setattr(cache_manager_module, "_atomic_write_bytes", _fail_shared_write)

    local_path = manager.put(b"ptx", "kernel.ptx")

    assert Path(local_path).read_bytes() == b"ptx"
    assert not (manager.shared_cache_dir / "kernel.ptx").exists()


def test_local_write_failure_raises(monkeypatch, tmp_path, cache_manager_module):
    monkeypatch.setenv("VLLM_TRITON_CACHE_MODE", "hierarchical")
    monkeypatch.setenv("VLLM_TRITON_LOCAL_CACHE_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("VLLM_TRITON_SHARED_CACHE_DIR", str(tmp_path / "shared"))

    manager = cache_manager_module.HierarchicalFileCacheManager("TESTKEY")

    def _fail_local_write(path, payload):
        if str(path).startswith(str(manager.local_cache_dir)):
            raise OSError(errno.EIO, "local write failed")
        return None

    monkeypatch.setattr(cache_manager_module, "_atomic_write_bytes", _fail_local_write)

    with pytest.raises(OSError):
        manager.put(b"ptx", "kernel.ptx")
