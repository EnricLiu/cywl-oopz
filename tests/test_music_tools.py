from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cywl_oopz.features.agent.models import AgentIdentity, AgentRunLimits
from cywl_oopz.features.agent.tools.models import ToolExecutionContext, ToolExecutionError
from cywl_oopz.features.agent.tools.music import (
    ClearMusicQueueTool,
    EnqueueMusicInput,
    EnqueueMusicTool,
    GetMusicQueueTool,
    MusicPlaybackModeInput,
    MusicSearchInput,
    MusicSourceSelection,
    MusicTrackReferenceInput,
    SearchMusicCatalogTool,
    SetMusicPlaybackModeTool,
)
from cywl_oopz.features.chat.models import ConversationKey
from cywl_oopz.features.music.errors import (
    MusicAuthenticationRequiredError,
    MusicExtractionTimeoutError,
    MusicGeoRestrictedError,
    MusicLiveUnsupportedError,
    MusicNoAudioFormatError,
    MusicSourceDisabledError,
    MusicSourceRateLimitedError,
    MusicTrackTooLongError,
    MusicUnsupportedContentError,
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
)
from cywl_oopz.features.music.models import (
    EnqueueResult,
    MusicPlaybackPolicy,
    MusicQueueClearResult,
    MusicQueueSnapshot,
    MusicTrack,
    MusicTrackReference,
    PlaybackOrder,
    PlaybackPolicyChange,
    PlaybackState,
    QueuedTrack,
    RepeatPolicy,
    VoiceChannelKey,
)


@dataclass
class StubMusic:
    track: MusicTrack

    async def search(self, query: str, *, source=None, limit: int | None = None):
        assert query == self.track.title
        assert source in {None, self.track.source}
        assert limit in {1, 3}
        return (self.track,)

    async def enqueue_input(
        self,
        identity: AgentIdentity,
        value: str,
        *,
        source=None,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        assert identity.person_id == "person"
        assert value == self.track.title
        assert source is None
        assert idempotency_key.endswith(":blue train")
        item = QueuedTrack(self.track, identity.person_id)
        return EnqueueResult(VoiceChannelKey("area", "voice"), item, 1, True)

    async def enqueue_reference(
        self,
        identity: AgentIdentity,
        reference: MusicTrackReference,
        *,
        idempotency_key: str = "",
    ) -> EnqueueResult:
        assert identity.person_id == "person"
        assert reference == self.track.reference
        assert idempotency_key.endswith(f":{reference.source.value}:{reference.source_id}")
        item = QueuedTrack(self.track, identity.person_id)
        return EnqueueResult(VoiceChannelKey("area", "voice"), item, 1, True)

    async def queue(self, identity: AgentIdentity) -> MusicQueueSnapshot:
        item = QueuedTrack(self.track, identity.person_id)
        return MusicQueueSnapshot(
            VoiceChannelKey("area", "voice"),
            PlaybackState.PLAYING,
            MusicPlaybackPolicy(),
            item,
            (),
            0,
            2,
        )

    async def set_policy(
        self,
        identity: AgentIdentity,
        *,
        order: PlaybackOrder | None = None,
        repeat: RepeatPolicy | None = None,
    ) -> PlaybackPolicyChange:
        assert identity.person_id == "person"
        return PlaybackPolicyChange(
            VoiceChannelKey("area", "voice"),
            MusicPlaybackPolicy(
                order=order or PlaybackOrder.SEQUENTIAL,
                repeat=repeat or RepeatPolicy.OFF,
            ),
            True,
        )

    async def clear(self, identity: AgentIdentity) -> MusicQueueClearResult:
        assert identity.person_id == "person"
        return MusicQueueClearResult(VoiceChannelKey("area", "voice"), True, 3)


def context() -> ToolExecutionContext:
    identity = AgentIdentity(
        "person",
        ConversationKey("channel", "area", "text", "person"),
    )
    return ToolExecutionContext(
        uuid4(),
        identity,
        AgentRunLimits(),
        (
            "search_music_catalog",
            "enqueue_music",
            "get_music_queue",
            "set_music_playback_mode",
            "clear_music_queue",
        ),
    )


@pytest.mark.asyncio
async def test_music_tools_expose_bounded_metadata_and_trusted_voice_target() -> None:
    music = StubMusic(MusicTrack("netease", "42", "Blue Train", ("Coltrane",), 1000))
    options = {"timeout_seconds": 1, "max_output_characters": 2000}
    search = SearchMusicCatalogTool(music, **options)
    enqueue = EnqueueMusicTool(music, **options)
    queue = GetMusicQueueTool(music, **options)
    mode = SetMusicPlaybackModeTool(music, **options)
    clear = ClearMusicQueueTool(music, **options)

    search_output = await search.execute(
        context(),
        MusicSearchInput(query="Blue Train", limit=3),
    )
    enqueue_output = await enqueue.execute(
        context(),
        EnqueueMusicInput(query="Blue Train"),
    )
    exact_output = await enqueue.execute(
        context(),
        EnqueueMusicInput(track=MusicTrackReferenceInput(source="netease", source_id="42")),
    )
    queue_output = await queue.execute(context(), queue.descriptor.input_model())
    mode_output = await mode.execute(
        context(),
        MusicPlaybackModeInput(
            order=PlaybackOrder.SHUFFLE,
            repeat=RepeatPolicy.ALL,
        ),
    )
    clear_output = await clear.execute(context(), clear.descriptor.input_model())

    assert search_output.model_dump()["tracks"][0]["source_id"] == "42"
    assert enqueue_output.model_dump()["voice_channel_id"] == "voice"
    assert exact_output.model_dump()["track"]["source_id"] == "42"
    assert queue_output.model_dump()["state"] == "playing"
    assert queue_output.model_dump()["order"] == PlaybackOrder.SEQUENTIAL
    assert queue_output.model_dump()["repeat"] == RepeatPolicy.OFF
    assert mode_output.model_dump() == {
        "order": PlaybackOrder.SHUFFLE,
        "repeat": RepeatPolicy.ALL,
        "changed": True,
    }
    assert clear_output.model_dump() == {
        "voice_channel_id": "voice",
        "stopped_current": True,
        "removed_count": 3,
    }


def test_music_playback_policy_input_requires_at_least_one_dimension() -> None:
    with pytest.raises(ValidationError):
        MusicPlaybackModeInput()


def test_music_enqueue_input_requires_one_query_or_reference() -> None:
    with pytest.raises(ValidationError):
        EnqueueMusicInput()
    with pytest.raises(ValidationError):
        EnqueueMusicInput(
            query="song",
            track=MusicTrackReferenceInput(source="youtube", source_id="dQw4w9WgXcQ"),
        )
    with pytest.raises(ValidationError):
        EnqueueMusicInput(
            source=MusicSourceSelection.YOUTUBE,
            track=MusicTrackReferenceInput(source="youtube", source_id="dQw4w9WgXcQ"),
        )


@pytest.mark.asyncio
async def test_music_tools_map_expected_domain_failures_to_stable_codes() -> None:
    class MissingVoiceMusic(StubMusic):
        async def enqueue_input(
            self,
            identity: AgentIdentity,
            value: str,
            *,
            source=None,
            idempotency_key: str = "",
        ) -> EnqueueResult:
            del identity, value, source, idempotency_key
            raise MusicVoiceChannelRequiredError("join voice first")

    music = MissingVoiceMusic(MusicTrack("netease", "42", "Song", (), 1000))
    tool = EnqueueMusicTool(
        music,
        timeout_seconds=1,
        max_output_characters=2000,
    )

    with pytest.raises(ToolExecutionError) as error:
        await tool.execute(context(), EnqueueMusicInput(query="Song"))

    assert error.value.error_code == "music_voice_channel_required"


@pytest.mark.parametrize(
    ("error_type", "error_code"),
    (
        (MusicSourceDisabledError, "music_source_disabled"),
        (MusicExtractionTimeoutError, "music_extraction_timeout"),
        (MusicSourceRateLimitedError, "music_rate_limited"),
        (MusicAuthenticationRequiredError, "music_authentication_required"),
        (MusicGeoRestrictedError, "music_geo_restricted"),
        (MusicLiveUnsupportedError, "music_live_unsupported"),
        (MusicTrackTooLongError, "music_track_too_long"),
        (MusicNoAudioFormatError, "music_no_audio_format"),
        (MusicUnsupportedContentError, "music_content_unsupported"),
    ),
)
@pytest.mark.asyncio
async def test_music_search_maps_source_failures_to_stable_codes(
    error_type: type[Exception],
    error_code: str,
) -> None:
    class FailingSourceMusic(StubMusic):
        async def search(self, query: str, *, source=None, limit: int | None = None):
            del query, source, limit
            raise error_type("provider failure")

    music = FailingSourceMusic(MusicTrack("youtube", "video-id", "Song", (), 1000))
    tool = SearchMusicCatalogTool(music, timeout_seconds=1, max_output_characters=2000)

    with pytest.raises(ToolExecutionError) as error:
        await tool.execute(
            context(),
            MusicSearchInput(query="Song", source=MusicSourceSelection.YOUTUBE),
        )

    assert error.value.error_code == error_code


@pytest.mark.asyncio
async def test_enqueue_music_tool_reports_shared_voice_conflict() -> None:
    class BusyVoiceMusic(StubMusic):
        async def enqueue_input(
            self,
            identity: AgentIdentity,
            value: str,
            *,
            source=None,
            idempotency_key: str = "",
        ) -> EnqueueResult:
            del identity, value, source, idempotency_key
            raise MusicVoiceBusyError("voice conversation owns the backend")

    music = BusyVoiceMusic(MusicTrack("netease", "42", "Song", (), 1000))
    tool = EnqueueMusicTool(music, timeout_seconds=1, max_output_characters=2000)

    with pytest.raises(ToolExecutionError) as error:
        await tool.execute(context(), EnqueueMusicInput(query="Song"))

    assert error.value.error_code == "music_voice_busy"
