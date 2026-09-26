"""Mangrove: native vector search and an HTTP client for the anchor service."""
from .client import Client, ServiceError

__version__ = '0.3.0.dev0'
__all__ = ['Client', 'ServiceError', 'AnchorIndex', 'AnchorBatchError', 'IdempotencyConflict']

def __getattr__(name):
    if name in ('AnchorIndex', 'AnchorBatchError', 'IdempotencyConflict'):
        from . import anchors
        value = getattr(anchors, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
