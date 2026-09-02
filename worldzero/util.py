from __future__ import annotations
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
import sqlite3
import stat
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any


@dataclass
class _MutationCapability:
    fd: int
    logical: Path
    physical: Path
    process_id: int
    thread_id: int
    task_id: int | None
    active: bool = True


_MUTATION_TOKEN: ContextVar[str | None] = ContextVar(
    "worldzero_mutation_token", default=None
)
_CAPABILITY_LOCK = threading.RLock()
_CAPABILITIES: dict[str, _MutationCapability] = {}


def _fd_path(fd: int) -> Path:
    if hasattr(fcntl, "F_GETPATH"):
        raw = fcntl.fcntl(fd, fcntl.F_GETPATH, b"\0" * 1024)
        return Path(os.fsdecode(raw.split(b"\0", 1)[0]))
    return Path(os.readlink(f"/proc/self/fd/{fd}"))


@contextmanager
def anchored_mutations(directory_fd: int, logical_root: Path):
    """Anchor opted-in I/O to a lifetime- and owner-checked capability."""
    fd = os.dup(directory_fd)
    key = uuid.uuid4().hex
    capability = _MutationCapability(
        fd=fd,
        logical=Path(logical_root).absolute(),
        physical=_fd_path(fd),
        process_id=os.getpid(),
        thread_id=threading.get_ident(),
        task_id=_current_task_id(),
    )
    with _CAPABILITY_LOCK:
        _CAPABILITIES[key] = capability
    context_token = _MUTATION_TOKEN.set(key)
    try:
        yield
    finally:
        with _CAPABILITY_LOCK:
            capability.active = False
            _CAPABILITIES.pop(key, None)
        _MUTATION_TOKEN.reset(context_token)
        os.close(fd)


def _current_task_id() -> int | None:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        return None
    return id(task) if task is not None else None


def _relative_parts(path: Path, root: _MutationCapability) -> tuple[str, ...]:
    path = Path(path)
    if path.is_absolute():
        for base in (root.logical, root.physical):
            try:
                path = path.relative_to(base)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Mutation path escapes its anchored output root")
    parts = path.parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Mutation path is not canonical beneath its output root")
    return parts


def _contains(path: Path, root: _MutationCapability) -> bool:
    try:
        _relative_parts(Path(path), root)
    except ValueError:
        return False
    return True


def _validate_capability(capability: _MutationCapability) -> None:
    if not capability.active:
        raise RuntimeError("Protected I/O capability is closed or expired")
    if capability.process_id != os.getpid():
        raise RuntimeError("Protected I/O capability belongs to another process")
    if capability.thread_id != threading.get_ident():
        raise RuntimeError("Protected I/O capability belongs to another thread")
    if capability.task_id != _current_task_id():
        raise RuntimeError("Protected I/O capability belongs to another async task")


def _capability_for_path(path: Path) -> _MutationCapability | None:
    token = _MUTATION_TOKEN.get()
    if token is not None:
        with _CAPABILITY_LOCK:
            capability = _CAPABILITIES.get(token)
        if capability is None:
            raise RuntimeError("Protected I/O capability is closed or expired")
        _validate_capability(capability)
        if _contains(Path(path), capability):
            return capability

    with _CAPABILITY_LOCK:
        active = tuple(_CAPABILITIES.values())
    for capability in active:
        if capability.active and _contains(Path(path), capability):
            raise RuntimeError(
                "Protected output path was accessed without its owning capability"
            )
    return None


def _current_capability() -> _MutationCapability | None:
    token = _MUTATION_TOKEN.get()
    if token is None:
        return None
    with _CAPABILITY_LOCK:
        capability = _CAPABILITIES.get(token)
    if capability is None:
        raise RuntimeError("Protected I/O capability is closed or expired")
    _validate_capability(capability)
    return capability


def mutation_is_anchored(path: Path) -> bool:
    return _capability_for_path(Path(path)) is not None


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _anchored_parent(path: Path, *, create: bool) -> tuple[int, str]:
    root = _capability_for_path(Path(path))
    if root is None:
        raise RuntimeError("No anchored mutation root is active")
    parts = _relative_parts(Path(path), root)
    if not parts:
        raise ValueError("Mutation target must be a file below its output root")
    fd = os.dup(root.fd)
    try:
        for part in parts[:-1]:
            try:
                child = os.open(part, _directory_flags(), dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o755, dir_fd=fd)
                child = os.open(part, _directory_flags(), dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, parts[-1]
    except Exception:
        os.close(fd)
        raise


def _anchored_directory(path: Path) -> int:
    root = _capability_for_path(Path(path))
    if root is None:
        raise RuntimeError("No anchored mutation root is active")
    parts = _relative_parts(Path(path), root)
    fd = os.dup(root.fd)
    try:
        for part in parts:
            child = os.open(part, _directory_flags(), dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except Exception:
        os.close(fd)
        raise


def _validate_regular_entry(parent_fd: int, name: str, *, missing_ok: bool) -> bool:
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"Unsafe symlink mutation target: {name}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Unsafe special non-regular mutation target: {name}")
    if metadata.st_nlink != 1:
        raise ValueError(f"Unsafe hard-linked mutation target: {name}")
    return True


def validate_anchored_tree() -> None:
    """Recursively reject aliases/special files through descriptor-relative traversal."""
    root = _current_capability()
    if root is None:
        return

    def inspect(directory_fd: int) -> None:
        for name in os.listdir(directory_fd):
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Unsafe symlink entry in anchored output: {name}")
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(name, _directory_flags(), dir_fd=directory_fd)
                try:
                    inspect(child)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(
                    f"Unsafe special non-regular entry in anchored output: {name}"
                )
            if metadata.st_nlink != 1:
                raise ValueError(f"Unsafe hard-linked entry in anchored output: {name}")

    inspect(root.fd)


def anchored_read_bytes(path: Path, *, max_bytes: int | None = None) -> bytes:
    capability = _capability_for_path(Path(path))
    if capability is None:
        value = Path(path).read_bytes()
        if max_bytes is not None and len(value) > max_bytes:
            raise ValueError("File exceeds the configured protected read limit")
        return value
    parent_fd, name = _anchored_parent(Path(path), create=False)
    fd = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        fd = os.open(name, flags, dir_fd=parent_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Unsafe aliased read target: {name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError("File exceeds the configured protected read limit")
            chunks.append(chunk)
    finally:
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


def anchored_exists(path: Path) -> bool:
    if _capability_for_path(Path(path)) is None:
        return Path(path).exists()
    try:
        parent_fd, name = _anchored_parent(Path(path), create=False)
    except FileNotFoundError:
        return False
    try:
        return _validate_regular_entry(parent_fd, name, missing_ok=True)
    finally:
        os.close(parent_fd)


def anchored_listdir(path: Path) -> list[str]:
    """List one protected directory through its anchor, rejecting unsafe entries."""
    if _capability_for_path(Path(path)) is None:
        return sorted(entry.name for entry in Path(path).iterdir())
    fd = _anchored_directory(Path(path))
    try:
        names = sorted(os.listdir(fd))
        for name in names:
            metadata = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Unsafe symlink entry in anchored directory: {name}")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"Unsafe special non-regular entry in anchored directory: {name}")
            if metadata.st_nlink != 1:
                raise ValueError(f"Unsafe hard-linked entry in anchored directory: {name}")
        return names
    finally:
        os.close(fd)


def anchored_unlink(path: Path) -> None:
    if _capability_for_path(Path(path)) is None:
        Path(path).unlink(missing_ok=True)
        return
    parent_fd, name = _anchored_parent(Path(path), create=False)
    try:
        if _validate_regular_entry(parent_fd, name, missing_ok=True):
            os.unlink(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_expected_sha256(
    label: str, value: Any, expected_sha256: str | None,
) -> None:
    """Optionally bind portable JSON to an externally supplied trust anchor."""

    if expected_sha256 is None:
        return
    if type(expected_sha256) is not str:
        raise TypeError(f"Expected {label} SHA-256 digest must be a string")
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError(f"Expected {label} SHA-256 digest is malformed")
    if digest(value) != expected_sha256:
        raise ValueError(f"{label} SHA-256 digest does not match the external anchor")


def derive_seed(seed: int, label: str) -> int:
    return int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:16], 16)


def atomic_bytes(path: Path, value: bytes) -> None:
    path = Path(path)
    if mutation_is_anchored(path):
        parent_fd, name = _anchored_parent(path, create=True)
        temporary = f".worldzero-{uuid.uuid4().hex}"
        fd = -1
        try:
            _validate_regular_entry(parent_fd, name, missing_ok=True)
            flags = (
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            )
            fd = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
            view = memoryview(value)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(
                temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
            )
            os.fsync(parent_fd)
            _validate_regular_entry(parent_fd, name, missing_ok=False)
            return
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.close(parent_fd)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".worldzero-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(Path(path), value.encode("utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        Path(path), json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def protected_sqlite_connect(path: Path) -> tuple[sqlite3.Connection, bool]:
    """Open an in-memory working copy when an anchored mutation root is active."""
    path = Path(path)
    if not mutation_is_anchored(path):
        return sqlite3.connect(path, timeout=30), False
    sidecars = [
        path.with_name(path.name + suffix)
        for suffix in ("-journal", "-wal", "-shm")
    ]
    main_exists = anchored_exists(path)
    present_sidecars = [sidecar.name for sidecar in sidecars if anchored_exists(sidecar)]
    if present_sidecars:
        raise ValueError(
            "Protected SQLite database has a forbidden journal/WAL/SHM sidecar: "
            + ", ".join(present_sidecars)
        )
    target = sqlite3.connect(":memory:")
    if not main_exists:
        return target, True
    with tempfile.TemporaryDirectory(prefix="worldzero-sqlite-anchor-") as scratch:
        copied = Path(scratch) / path.name
        copied.write_bytes(anchored_read_bytes(path))
        source = sqlite3.connect(copied)
        try:
            source.backup(target)
        finally:
            source.close()
    return target, True


def protected_sqlite_persist(
    connection: sqlite3.Connection, path: Path, protected: bool
) -> None:
    if not protected:
        return
    connection.commit()
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(path).with_name(Path(path).name + suffix)
        if anchored_exists(sidecar):
            raise ValueError(
                f"Protected SQLite publication refused forbidden sidecar: {sidecar.name}"
            )
    serialize = getattr(connection, "serialize", None)
    if serialize is not None:
        database = serialize()
    else:  # Python 3.10
        with tempfile.TemporaryDirectory(prefix="worldzero-sqlite-image-") as scratch:
            snapshot = Path(scratch) / "database.sqlite"
            target = sqlite3.connect(snapshot)
            try:
                connection.backup(target)
            finally:
                target.close()
            database = snapshot.read_bytes()
    atomic_bytes(Path(path), database)


def require_finite(name: str, value: float, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
