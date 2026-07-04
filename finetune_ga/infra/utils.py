"""Shared lightweight utilities with no heavy dependencies.

Centralises helpers that were previously copy-pasted across modules.
"""

import os


def env_flag(name: str, default: str = '0') -> bool:
    """Return True if the named environment variable is set to a truthy value.

    Truthy values: '1', 'true', 'yes', 'y', 'on' (case-insensitive).
    """
    return os.environ.get(name, default).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
