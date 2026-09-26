"""Load the anchor-only product library without initializing legacy engines."""
import ctypes
import os
from pathlib import Path


def load():
    override = os.environ.get('MANGROVE_ANCHOR_LIBRARY')
    if override:
        return ctypes.CDLL(str(Path(override).resolve()))
    package = Path(__file__).resolve().parent
    candidates = [package/'libmangrove_anchor.so',
                  package.parent.parent/'libmangrove_anchor.so',
                  package.parent.parent/'libmangrove.so',
                  Path('/usr/local/lib/libmangrove_anchor.so')]
    for path in candidates:
        if path.is_file():
            return ctypes.CDLL(str(path))
    raise OSError('Native anchor library not found. Build with make product or set MANGROVE_ANCHOR_LIBRARY.')


_lib = load()
