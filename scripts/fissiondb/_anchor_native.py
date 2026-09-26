"""Load the anchor-only product library without initializing legacy engines."""
import ctypes
import os
from pathlib import Path


def load():
    override = os.environ.get('FISSIONDB_ANCHOR_LIBRARY')
    if override:
        return ctypes.CDLL(str(Path(override).resolve()))
    package = Path(__file__).resolve().parent
    candidates = [package/'libfissiondb_anchor.so',
                  package.parent.parent/'libfissiondb_anchor.so',
                  package.parent.parent/'libfissiondb.so',
                  Path('/usr/local/lib/libfissiondb_anchor.so')]
    for path in candidates:
        if path.is_file():
            return ctypes.CDLL(str(path))
    raise OSError('Native anchor library not found. Build with make product or set FISSIONDB_ANCHOR_LIBRARY.')


_lib = load()
