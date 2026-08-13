from __future__ import annotations

import pytest

from cywl_oopz.commands.definitions import CommandUsageError
from cywl_oopz.commands.models import (
    CommandActor,
    CommandLocation,
    CommandRequest,
    CommandScope,
    CommandSource,
    CommandText,
    CommandTrigger,
)
from cywl_oopz.features.music.command_parsing import (
    MusicArgumentsParser,
    MusicModeArguments,
    MusicPlayArguments,
    PlaylistImportArguments,
)
from cywl_oopz.features.music.models import MusicSourceKind, PlaybackOrder, RepeatPolicy


class Responder:
    async def reply(self, message):
        del message

    async def send(self, message):
        del message

    async def react(self, emoji):
        del emoji


def request(*tokens: str) -> CommandRequest:
    tail = " ".join(tokens)
    return CommandRequest(
        CommandTrigger.TEXT,
        CommandActor("person"),
        CommandLocation(CommandScope.CHANNEL, "area", "channel"),
        CommandSource("message"),
        Responder(),
        CommandText(f"!music {tail}".rstrip(), "music", tail, tuple(tokens)),
    )


def test_music_parser_preserves_source_value_and_mode_semantics() -> None:
    parser = MusicArgumentsParser("detailed usage")

    play = parser.parse(request("play", "--source", "youtube", "Tell", "Your", "World"))
    mode = parser.parse(request("mode", "shuffle", "all"))

    assert play == MusicPlayArguments(MusicSourceKind.YOUTUBE, "Tell Your World")
    assert mode == MusicModeArguments(PlaybackOrder.SHUFFLE, RepeatPolicy.ALL)


def test_music_parser_extracts_netease_import_options_once() -> None:
    parser = MusicArgumentsParser("detailed usage")

    parsed = parser.parse(
        request("playlist", "import", "24381616", "--partial", "新的", "云端", "歌单")
    )

    assert parsed == PlaylistImportArguments("24381616", "新的 云端 歌单", True)


@pytest.mark.parametrize(
    "tokens",
    [
        ("search", "初音未来", "--source", "youtube"),
        ("playlist", "show", "not-a-reference"),
        ("mode", "impossible"),
    ],
)
def test_music_parser_rejects_invalid_syntax_with_detailed_usage(tokens) -> None:
    parser = MusicArgumentsParser("🎵 **Music 命令**")

    with pytest.raises(CommandUsageError) as failure:
        parser.parse(request(*tokens))

    assert failure.value.include_usage is False
    assert failure.value.user_message == "🎵 **Music 命令**"
