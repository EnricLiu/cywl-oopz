"""Capability gate for the realtime voice contract supplied by OOPZ SDK."""

from __future__ import annotations

from dataclasses import dataclass

from oopz_sdk import VoiceCapabilities

from cywl_oopz.core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class OopzVoiceCapabilityGate:
    """Validate the exact SDK primitives required by CYWL voice sessions."""

    minimum_feature_version: int = 1
    required_flags: tuple[str, ...] = (
        "remote_audio_subscription",
        "person_audio_subscription",
        "streaming_pcm_output",
        "playback_cursor",
        "typed_playback_handle",
    )

    def validate(self, capabilities: VoiceCapabilities) -> None:
        """Fail before joining OOPZ when the installed SDK contract is incomplete."""
        missing = tuple(
            name for name in self.required_flags if not bool(getattr(capabilities, name, False))
        )
        if capabilities.feature_version < self.minimum_feature_version or missing:
            details: list[str] = []
            if capabilities.feature_version < self.minimum_feature_version:
                details.append(
                    f"feature_version>={self.minimum_feature_version} "
                    f"(installed={capabilities.feature_version})"
                )
            if missing:
                details.append(f"capabilities={','.join(missing)}")
            raise ConfigurationError(
                "Realtime voice requires a newer OOPZ SDK voice contract: " + "; ".join(details)
            )
