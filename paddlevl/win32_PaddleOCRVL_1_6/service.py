"""Compatibility import for the plugin-wide llama.cpp runtime.

New code must use ``runtime.getService()``.  Keeping ServiceController avoids
breaking diagnostics and older local tests that import this module directly.
"""

from .runtime import RuntimeService, tempRoot


class ServiceController(RuntimeService):
    pass
