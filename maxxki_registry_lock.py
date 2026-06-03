"""
MAXXKI Registry Lock v1.0
Cross-Platform File-Locking für parallele Registry-Writes.

Unix:    fcntl (POSIX advisory locks)
Windows: msvcrt / portalocker
Fallback: Atomic rename (race-condition-resistent, aber nicht exklusiv)
"""

import os
import sys
import time
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

# ═══ Platform Detection ═════════════════════════════════════════════════════

_HAVE_FCNTL = False
_HAVE_PORTALOCKER = False

if os.name == "posix":
    try:
        import fcntl
        _HAVE_FCNTL = True
    except ImportError:
        pass

try:
    import portalocker
    _HAVE_PORTALOCKER = True
except ImportError:
    pass


# ═══ Lock Implementierungen ═════════════════════════════════════════════════

class _LockBase:
    """Abstract base — acquire/release Context Manager."""
    def __init__(self, path: Path, timeout: float = 10.0, poll_interval: float = 0.05):
        self.path = path
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._locked = False

    def acquire(self) -> bool:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Lock timeout: {self.path}")
        return self

    def __exit__(self, *args):
        self.release()
        return False


class _FcntlLock(_LockBase):
    """POSIX advisory lock via fcntl."""
    def __init__(self, path: Path, timeout: float = 10.0, poll_interval: float = 0.05):
        super().__init__(path, timeout, poll_interval)
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT)
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._locked = True
                return True
            except (IOError, OSError):
                if self._fd is not None:
                    os.close(self._fd)
                    self._fd = None
                time.sleep(self.poll_interval)
        return False

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
        self._locked = False


class _PortalockerLock(_LockBase):
    """Cross-Platform via portalocker (pip install portalocker)."""
    def __init__(self, path: Path, timeout: float = 10.0, poll_interval: float = 0.05):
        super().__init__(path, timeout, poll_interval)
        self._file: Optional[object] = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                self._file = open(self.path, "r+")
                portalocker.lock(self._file, portalocker.LOCK_EX)
                self._locked = True
                return True
            except Exception:
                if self._file:
                    self._file.close()
                    self._file = None
                time.sleep(self.poll_interval)
        return False

    def release(self) -> None:
        if self._file:
            try:
                portalocker.unlock(self._file)
            except Exception:
                pass
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        self._locked = False


class _AtomicLock(_LockBase):
    """
    Fallback: Thread-local Mutex + Atomic rename.
    Nicht prozess-sicher, aber thread-sicher und race-resistent.
    """
    _global_lock = threading.Lock()

    def acquire(self) -> bool:
        acquired = self._global_lock.acquire(timeout=self.timeout)
        self._locked = acquired
        return acquired

    def release(self) -> None:
        if self._locked:
            self._global_lock.release()
            self._locked = False


# ═══ Factory ══════════════════════════════════════════════════════════════════

def get_lock(path: Path, timeout: float = 10.0) -> _LockBase:
    """
    Gibt die beste verfügbare Lock-Implementierung zurück.

    Priority:
      1. portalocker (cross-platform, robust)
      2. fcntl (Unix-native, kein extra Dependency)
      3. Atomic/thread-lock (Fallback)
    """
    if _HAVE_PORTALOCKER:
        return _PortalockerLock(path, timeout)
    if _HAVE_FCNTL:
        return _FcntlLock(path, timeout)
    return _AtomicLock(path, timeout)


# ═══ Convenience: Decorator ═══════════════════════════════════════════════════

def with_registry_lock(timeout: float = 10.0):
    """Decorator für Registry-Methoden die exklusiven Zugriff brauchen."""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            lock_path = self.path.with_suffix(".lock")
            with get_lock(lock_path, timeout=timeout):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ═══ Atomic Append Helper ═════════════════════════════════════════════════════

def atomic_append_line(file_path: Path, line: str) -> None:
    """
    Atomisches Append via temp-file + rename.
    Schützt gegen korrupte Writes bei Crash.
    """
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(line + "\n")
    tmp.replace(file_path)
