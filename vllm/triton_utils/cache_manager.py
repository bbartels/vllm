# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import json
import os
import shutil
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Optional

from triton import knobs
from triton.runtime.cache import FileCacheManager

from vllm.logger import init_logger

logger = init_logger(__name__)

_EXPECTED_CACHE_READ_ERRNOS = {errno.ENOENT}
if hasattr(errno, "ESTALE"):
    _EXPECTED_CACHE_READ_ERRNOS.add(errno.ESTALE)

_HIERARCHICAL_MANAGER = (
    "vllm.triton_utils.cache_manager:HierarchicalFileCacheManager"
)
_PROCESS_MANAGER = "vllm.triton_utils.cache_manager:ProcessFileCacheManager"
_VALID_CACHE_MODES = {"default", "process", "hierarchical", "local"}
_DEFAULT_LOCAL_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "vllm-triton-cache")


class ProcessFileCacheManager(FileCacheManager):
    """A per-process Triton cache manager.

    This preserves the old vLLM behavior of isolating Triton writes by PID to
    avoid same-node multiprocessing collisions.
    """

    def __init__(self, key, override: bool = False, dump: bool = False):
        self.key = key
        self.lock_path = None
        if dump or override:
            super().__init__(key, override=override, dump=dump)
            return

        cache_root = os.getenv("TRITON_CACHE_DIR", "").strip() or knobs.cache.dir
        if not cache_root:
            raise RuntimeError("Could not create or locate cache dir")

        self.cache_dir = os.path.join(f"{cache_root}_{os.getpid()}", self.key)
        self.lock_path = os.path.join(self.cache_dir, "lock")
        os.makedirs(self.cache_dir, exist_ok=True)


class HierarchicalFileCacheManager(FileCacheManager):
    """Use a local live cache with an optional shared read/publish cache.

    Shared cache read failures are treated as cache misses. Shared cache hits are
    materialized into the local cache before their paths are returned to Triton.
    """

    def __init__(self, key, override: bool = False, dump: bool = False):
        self.key = key
        self.lock_path = None
        self._passthrough = dump or override
        if self._passthrough:
            super().__init__(key, override=override, dump=dump)
            return

        local_root = _resolve_local_cache_root()
        shared_root = _resolve_shared_cache_root()

        self.local_root = local_root / f"pid-{os.getpid()}"
        self.local_cache_dir = self.local_root / key
        self.shared_root = shared_root
        self.shared_cache_dir = shared_root / key if shared_root is not None else None
        self.publish_enabled = _publish_enabled() and self.shared_cache_dir is not None

        self.cache_dir = str(self.local_cache_dir)
        self.lock_path = str(self.local_cache_dir / "lock")
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_path(self, filename) -> str:
        return str(self.local_cache_dir / filename)

    def get_file(self, filename) -> Optional[str]:
        if self._passthrough:
            return super().get_file(filename)

        local_path = self.local_cache_dir / filename
        if _safe_exists(local_path):
            return str(local_path)

        if self.shared_cache_dir is None:
            return None

        shared_path = self.shared_cache_dir / filename
        if not _safe_exists(shared_path):
            return None

        return self._materialize_shared_file(filename, shared_path)

    def put(self, data, filename, binary=True) -> str:
        if self._passthrough:
            return super().put(data, filename, binary=binary)

        payload = data if isinstance(data, bytes) else str(data).encode("utf-8")
        local_path = self.local_cache_dir / filename
        _atomic_write_bytes(local_path, payload)

        if self.publish_enabled and self.shared_cache_dir is not None:
            shared_path = self.shared_cache_dir / filename
            try:
                _atomic_write_bytes(shared_path, payload)
            except OSError as exc:
                logger.warning(
                    "Failed to publish Triton cache artifact %s to shared cache %s: %s",
                    filename,
                    shared_path,
                    exc,
                )

        return str(local_path)

    def get_group(self, filename: str) -> Optional[dict[str, str]]:
        if self._passthrough:
            return super().get_group(filename)

        local_group = _read_group_from_dir(self.local_cache_dir, filename)
        if local_group is not None:
            return local_group

        if self.shared_cache_dir is None:
            return None

        shared_group = _read_group_from_dir(self.shared_cache_dir, filename)
        if shared_group is None:
            return None

        localized_group = self._materialize_group(shared_group)
        if localized_group is None:
            return None

        _write_group_manifest(self.local_cache_dir, filename, localized_group)
        return localized_group

    def put_group(self, filename: str, group: dict[str, str]) -> str:
        if self._passthrough:
            return super().put_group(filename, group)

        _write_group_manifest(self.local_cache_dir, filename, group)

        if self.publish_enabled and self.shared_cache_dir is not None:
            shared_group = {
                child_name: str(self.shared_cache_dir / child_name)
                for child_name in group
            }
            try:
                _write_group_manifest(self.shared_cache_dir, filename, shared_group)
            except OSError as exc:
                logger.warning(
                    "Failed to publish Triton cache group %s to shared cache %s: %s",
                    filename,
                    self.shared_cache_dir,
                    exc,
                )

        return str(self.local_cache_dir / f"__grp__{filename}")

    def _materialize_shared_file(
        self, filename: str, shared_path: Path
    ) -> Optional[str]:
        local_path = self.local_cache_dir / filename
        if _safe_exists(local_path):
            return str(local_path)

        try:
            _atomic_copy_file(shared_path, local_path)
        except OSError as exc:
            if _is_cache_read_error(exc):
                return None
            raise

        return str(local_path)

    def _materialize_group(
        self, shared_group: dict[str, str]
    ) -> Optional[dict[str, str]]:
        localized_group: dict[str, str] = {}
        for child_name, shared_path_str in shared_group.items():
            localized_path = self._materialize_shared_file(
                child_name, Path(shared_path_str)
            )
            if localized_path is None:
                return None
            localized_group[child_name] = localized_path
        return localized_group


def get_configured_triton_cache_manager() -> Optional[str]:
    mode = os.getenv("VLLM_TRITON_CACHE_MODE", "default").strip().lower()
    if mode not in _VALID_CACHE_MODES:
        logger.warning(
            "Ignoring unknown VLLM_TRITON_CACHE_MODE=%r. Expected one of %s.",
            mode,
            sorted(_VALID_CACHE_MODES),
        )
        return None
    if mode == "process":
        return _PROCESS_MANAGER
    if mode in {"hierarchical", "local"}:
        return _HIERARCHICAL_MANAGER
    return None


def _resolve_local_cache_root() -> Path:
    local_root = os.getenv("VLLM_TRITON_LOCAL_CACHE_DIR", "").strip()
    if not local_root:
        local_root = _DEFAULT_LOCAL_CACHE_ROOT
    return Path(local_root)


def _resolve_shared_cache_root() -> Optional[Path]:
    mode = os.getenv("VLLM_TRITON_CACHE_MODE", "default").strip().lower()
    if mode == "local":
        return None

    shared_root = os.getenv("VLLM_TRITON_SHARED_CACHE_DIR", "").strip()
    if not shared_root and mode == "hierarchical":
        shared_root = os.getenv("TRITON_CACHE_DIR", "").strip()
    return Path(shared_root) if shared_root else None


def _publish_enabled() -> bool:
    return bool(int(os.getenv("VLLM_TRITON_CACHE_PUBLISH", "1")))


def _is_cache_read_error(exc: OSError) -> bool:
    return exc.errno in _EXPECTED_CACHE_READ_ERRNOS


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        if _is_cache_read_error(exc):
            return False
        raise


def _read_group_from_dir(
    cache_dir: Path, filename: str
) -> Optional[dict[str, str]]:
    manifest_path = cache_dir / f"__grp__{filename}"
    if not _safe_exists(manifest_path):
        return None

    try:
        with manifest_path.open(encoding="utf-8") as f:
            group_data = json.load(f)
    except json.JSONDecodeError:
        return None
    except OSError as exc:
        if _is_cache_read_error(exc):
            return None
        raise

    child_paths = group_data.get("child_paths")
    if not isinstance(child_paths, dict) or not child_paths:
        return None

    result: dict[str, str] = {}
    for child_name, child_path in child_paths.items():
        if not isinstance(child_name, str) or not isinstance(child_path, str):
            return None
        if not _safe_exists(Path(child_path)):
            return None
        result[child_name] = child_path

    return result


def _write_group_manifest(
    cache_dir: Path, filename: str, group: dict[str, str]
) -> Path:
    manifest_path = cache_dir / f"__grp__{filename}"
    payload = json.dumps({"child_paths": group}, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(manifest_path, payload)
    return manifest_path


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = path.parent / f"tmp.pid_{os.getpid()}_{uuid.uuid4()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / path.name
    try:
        temp_path.write_bytes(payload)
        os.replace(temp_path, path)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        shutil.rmtree(temp_dir, ignore_errors=True)


def _atomic_copy_file(source: Path, dest: Path) -> None:
    if _safe_exists(dest):
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = dest.parent / f"tmp.pid_{os.getpid()}_{uuid.uuid4()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / dest.name
    try:
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, dest)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        shutil.rmtree(temp_dir, ignore_errors=True)
