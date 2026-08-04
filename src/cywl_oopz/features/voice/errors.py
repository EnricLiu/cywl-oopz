"""Typed failures exposed by the realtime voice use case."""

from __future__ import annotations


class VoiceConversationError(Exception):
    """Base class for expected voice conversation failures."""


class VoiceFeatureDisabledError(VoiceConversationError):
    """Realtime voice is disabled by the application feature gate."""


class VoiceChannelContextRequiredError(VoiceConversationError):
    """A command did not originate from an area text channel."""


class VoiceUserNotInChannelError(VoiceConversationError):
    """The session owner is not currently present in an area voice channel."""


class VoiceBackendBusyError(VoiceConversationError):
    """Music or another conversation owns the single voice backend."""


class VoiceSessionAlreadyActiveError(VoiceConversationError):
    """A voice conversation is already starting or active."""


class VoiceSessionNotActiveError(VoiceConversationError):
    """There is no session to stop."""


class VoiceSessionOwnershipError(VoiceConversationError):
    """Only the session owner may perform this operation."""


class VoiceRuntimeUnavailableError(VoiceConversationError):
    """No configured realtime Provider runtime can create a session."""


class VoiceSessionStartTimeoutError(VoiceConversationError):
    """Session startup exceeded its bounded deadline."""


class VoiceSessionStartCancelledError(VoiceConversationError):
    """The owner stopped a session while it was still starting."""


class VoiceAudioQueueClosedError(VoiceConversationError):
    """A media pump attempted to use a closed transit queue."""


class VoiceOutputBackpressureError(VoiceConversationError):
    """Provider output could not make progress before its hard timeout."""
