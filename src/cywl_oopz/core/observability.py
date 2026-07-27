"""Small helpers for useful logs that do not expose user content or credentials."""

from __future__ import annotations

import hashlib


def opaque_ref(*parts: object) -> str:
    """Return a stable short correlation token for identifiers kept out of logs."""
    material = "\0".join(str(part) for part in parts)
    return hashlib.blake2s(material.encode(), digest_size=6).hexdigest()


def exception_kind(error: BaseException) -> str:
    """Return only the exception type, never an untrusted exception message."""
    return type(error).__name__
