from __future__ import annotations

import asyncio
from contextvars import copy_context
import os
from pathlib import Path
import threading

import pytest

from worldzero.util import anchored_mutations, anchored_read_bytes, atomic_text


def _directory_fd(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )


def test_protected_path_in_worker_thread_fails_closed_instead_of_falling_back(tmp_path):
    target = tmp_path / "worker.txt"
    fd = _directory_fd(tmp_path)
    errors: list[BaseException] = []
    try:
        with anchored_mutations(fd, tmp_path):
            worker = threading.Thread(
                target=lambda: (
                    atomic_text(target, "unsafe")
                    if False
                    else _capture_error(errors, atomic_text, target, "unsafe")
                )
            )
            worker.start()
            worker.join(timeout=2)
            assert not worker.is_alive()
    finally:
        os.close(fd)
    assert errors and isinstance(errors[0], RuntimeError)
    assert not target.exists()


def _capture_error(errors, function, *args):
    try:
        function(*args)
    except BaseException as exc:
        errors.append(exc)


def test_copied_context_cannot_use_expired_capability_after_fd_reuse(tmp_path):
    target = tmp_path / "stale.txt"
    fd = _directory_fd(tmp_path)
    try:
        with anchored_mutations(fd, tmp_path):
            stale = copy_context()
    finally:
        os.close(fd)

    reused = os.open(tmp_path / "unrelated.txt", os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        with pytest.raises(RuntimeError, match="expired|closed|capability"):
            stale.run(atomic_text, target, "unsafe")
    finally:
        os.close(reused)
    assert not target.exists()


def test_async_child_task_cannot_outlive_or_borrow_protected_capability(tmp_path):
    target = tmp_path / "async.txt"
    fd = _directory_fd(tmp_path)

    async def scenario():
        release = asyncio.Event()

        async def child():
            await release.wait()
            atomic_text(target, "unsafe")

        with anchored_mutations(fd, tmp_path):
            task = asyncio.create_task(child())
        release.set()
        with pytest.raises(RuntimeError, match="expired|closed|task|capability"):
            await task

    try:
        asyncio.run(scenario())
    finally:
        os.close(fd)
    assert not target.exists()


def test_legacy_unprotected_atomic_path_remains_available(tmp_path):
    target = tmp_path / "legacy.txt"
    atomic_text(target, "legacy")
    assert target.read_text() == "legacy"


def test_guarded_fifo_read_is_nonblocking_even_after_file_swap(tmp_path, monkeypatch):
    import worldzero.util as util

    target = tmp_path / "value"
    target.write_bytes(b"safe")
    fd = _directory_fd(tmp_path)
    original_open = util.os.open
    swapped = False

    def swap_before_target_open(path, flags, *args, dir_fd=None, **kwargs):
        nonlocal swapped
        if path == "value" and dir_fd is not None and not swapped:
            swapped = True
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(target)
        return original_open(path, flags, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(util.os, "open", swap_before_target_open)
    try:
        with anchored_mutations(fd, tmp_path):
            # O_NONBLOCK must be applied by the descriptor-relative open.  Without it
            # this call hangs forever after the deterministic regular-file -> FIFO swap.
            with pytest.raises(ValueError, match="Unsafe aliased read target"):
                anchored_read_bytes(target)
    finally:
        os.close(fd)
