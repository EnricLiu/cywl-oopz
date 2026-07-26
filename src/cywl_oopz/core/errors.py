"""Application errors with safe messages for end users."""

from __future__ import annotations


class CywlError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(CywlError):
    """Raised when required application configuration is invalid or incomplete."""


class DatabaseError(CywlError):
    """Raised when the database cannot safely serve the application."""


class ProviderError(CywlError):
    """Raised when an upstream AI provider cannot fulfil a request."""


class ProviderTimeoutError(ProviderError):
    """Raised when an upstream AI provider exceeds the configured time budget."""


class ProviderResponseError(ProviderError):
    """Raised when an upstream AI provider returns an invalid response."""


class RateLimitExceeded(CywlError):
    """Raised when a user, channel, or process has reached a concurrency limit."""

    def __init__(self, scope: str, retry_after_seconds: float = 0.0) -> None:
        self.scope = scope
        self.retry_after_seconds = max(retry_after_seconds, 0.0)
        super().__init__(f"Rate limit exceeded for {scope}")


class FeatureDisabledError(CywlError):
    """Raised when a command targets a feature disabled by configuration."""


class AuthorizationError(CywlError):
    """Raised when a caller is not allowed to perform an action."""
