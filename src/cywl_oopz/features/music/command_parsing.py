"""Typed, I/O-free grammar for the ``music`` command group."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from cywl_oopz.commands.definitions import CommandUsageError
from cywl_oopz.commands.models import CommandRequest

from .models import MusicSourceKind, PlaybackOrder, RepeatPolicy


@dataclass(frozen=True, slots=True)
class MusicHelpArguments:
    pass


@dataclass(frozen=True, slots=True)
class MusicPlayArguments:
    source: MusicSourceKind | None
    value: str


@dataclass(frozen=True, slots=True)
class MusicSearchArguments:
    source: MusicSourceKind | None
    query: str


@dataclass(frozen=True, slots=True)
class MusicSourcesArguments:
    pass


@dataclass(frozen=True, slots=True)
class MusicQueueArguments:
    pass


class MusicControlAction(StrEnum):
    SKIP = "skip"
    PAUSE = "pause"
    RESUME = "resume"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class MusicControlArguments:
    action: MusicControlAction


@dataclass(frozen=True, slots=True)
class MusicModeArguments:
    order: PlaybackOrder | None = None
    repeat: RepeatPolicy | None = None

    @property
    def query_only(self) -> bool:
        return self.order is None and self.repeat is None


@dataclass(frozen=True, slots=True)
class PlaylistListArguments:
    pass


@dataclass(frozen=True, slots=True)
class PlaylistShowArguments:
    playlist: str


@dataclass(frozen=True, slots=True)
class PlaylistCreateArguments:
    name: str


@dataclass(frozen=True, slots=True)
class PlaylistRenameArguments:
    playlist: str
    name: str


class PlaylistMutationAction(StrEnum):
    DELETE = "delete"
    CLEAR = "clear"
    LOAD = "load"


@dataclass(frozen=True, slots=True)
class PlaylistMutationArguments:
    action: PlaylistMutationAction
    playlist: str


@dataclass(frozen=True, slots=True)
class PlaylistAddArguments:
    playlist: str
    source: MusicSourceKind | None
    value: str


@dataclass(frozen=True, slots=True)
class PlaylistRemoveArguments:
    playlist: str
    entry: str


@dataclass(frozen=True, slots=True)
class PlaylistPreviewArguments:
    reference: str


@dataclass(frozen=True, slots=True)
class PlaylistImportArguments:
    reference: str
    name: str | None
    allow_partial: bool


type MusicArguments = (
    MusicHelpArguments
    | MusicPlayArguments
    | MusicSearchArguments
    | MusicSourcesArguments
    | MusicQueueArguments
    | MusicControlArguments
    | MusicModeArguments
    | PlaylistListArguments
    | PlaylistShowArguments
    | PlaylistCreateArguments
    | PlaylistRenameArguments
    | PlaylistMutationArguments
    | PlaylistAddArguments
    | PlaylistRemoveArguments
    | PlaylistPreviewArguments
    | PlaylistImportArguments
)


class MusicArgumentsParser:
    """Parse aliases and options once before any music service or RBAC work."""

    _ORDER_ALIASES = {
        "sequential": PlaybackOrder.SEQUENTIAL,
        "顺序": PlaybackOrder.SEQUENTIAL,
        "shuffle": PlaybackOrder.SHUFFLE,
        "random": PlaybackOrder.SHUFFLE,
        "随机": PlaybackOrder.SHUFFLE,
    }
    _REPEAT_ALIASES = {
        "off": RepeatPolicy.OFF,
        "none": RepeatPolicy.OFF,
        "不循环": RepeatPolicy.OFF,
        "one": RepeatPolicy.ONE,
        "single": RepeatPolicy.ONE,
        "单曲": RepeatPolicy.ONE,
        "all": RepeatPolicy.ALL,
        "list": RepeatPolicy.ALL,
        "列表": RepeatPolicy.ALL,
    }
    _SOURCE_ALIASES = {
        "auto": None,
        "自动": None,
        "netease": MusicSourceKind.NETEASE,
        "163": MusicSourceKind.NETEASE,
        "网易云": MusicSourceKind.NETEASE,
        "youtube": MusicSourceKind.YOUTUBE,
        "yt": MusicSourceKind.YOUTUBE,
        "bilibili": MusicSourceKind.BILIBILI,
        "bili": MusicSourceKind.BILIBILI,
        "b站": MusicSourceKind.BILIBILI,
    }

    def __init__(self, usage_text: str) -> None:
        self._usage_text = usage_text

    def parse(self, request: CommandRequest) -> MusicArguments:
        assert request.text is not None
        tokens = request.text.tokens
        if not tokens:
            return MusicHelpArguments()
        action = tokens[0].casefold()
        values = tokens[1:]
        if action in {"help", "帮助"}:
            self._require_count(values, 0)
            return MusicHelpArguments()
        if action in {"play", "add", "点歌"}:
            source, value = self._source_and_value(values)
            return MusicPlayArguments(source, value)
        if action in {"search", "find", "搜索"}:
            source, value = self._source_and_value(values)
            return MusicSearchArguments(source, value)
        if action in {"sources", "source", "来源"}:
            self._require_count(values, 0)
            return MusicSourcesArguments()
        if action in {"queue", "status", "队列"}:
            self._require_count(values, 0)
            return MusicQueueArguments()
        controls = {
            "skip": MusicControlAction.SKIP,
            "next": MusicControlAction.SKIP,
            "下一首": MusicControlAction.SKIP,
            "pause": MusicControlAction.PAUSE,
            "暂停": MusicControlAction.PAUSE,
            "resume": MusicControlAction.RESUME,
            "继续": MusicControlAction.RESUME,
            "clear": MusicControlAction.CLEAR,
            "清空": MusicControlAction.CLEAR,
        }
        if action in controls:
            self._require_count(values, 0)
            return MusicControlArguments(controls[action])
        if action in {"mode", "policy", "模式"}:
            return self._mode(values)
        if action in {"playlist", "playlists", "pl", "歌单"}:
            return self._playlist(values)
        self._invalid()

    def _mode(self, values: tuple[str, ...]) -> MusicModeArguments:
        order: PlaybackOrder | None = None
        repeat: RepeatPolicy | None = None
        for value in values:
            token = value.casefold()
            if token in {"order", "repeat", "顺序方式", "循环方式"}:
                continue
            if token in self._ORDER_ALIASES and order is None:
                order = self._ORDER_ALIASES[token]
            elif token in self._REPEAT_ALIASES and repeat is None:
                repeat = self._REPEAT_ALIASES[token]
            else:
                self._invalid()
        if values and order is None and repeat is None:
            self._invalid()
        return MusicModeArguments(order, repeat)

    def _playlist(self, values: tuple[str, ...]) -> MusicArguments:
        if not values:
            return PlaylistListArguments()
        action = values[0].casefold()
        arguments = values[1:]
        if action in {"list", "ls", "列表"}:
            self._require_count(arguments, 0)
            return PlaylistListArguments()
        if action in {"show", "get", "查看"}:
            self._require_count(arguments, 1)
            return PlaylistShowArguments(self._reference(arguments[0]))
        if action in {"create", "new", "新建"}:
            return PlaylistCreateArguments(self._joined(arguments))
        if action in {"rename", "重命名"}:
            self._require_minimum(arguments, 2)
            return PlaylistRenameArguments(
                self._reference(arguments[0]),
                self._joined(arguments[1:]),
            )
        mutations = {
            "delete": PlaylistMutationAction.DELETE,
            "rm": PlaylistMutationAction.DELETE,
            "删除": PlaylistMutationAction.DELETE,
            "clear": PlaylistMutationAction.CLEAR,
            "清空": PlaylistMutationAction.CLEAR,
            "load": PlaylistMutationAction.LOAD,
            "play": PlaylistMutationAction.LOAD,
            "播放": PlaylistMutationAction.LOAD,
        }
        if action in mutations:
            self._require_count(arguments, 1)
            return PlaylistMutationArguments(
                mutations[action],
                self._reference(arguments[0]),
            )
        if action in {"add", "添加"}:
            self._require_minimum(arguments, 2)
            source, value = self._source_and_value(arguments[1:])
            return PlaylistAddArguments(self._reference(arguments[0]), source, value)
        if action in {"remove", "移除"}:
            self._require_count(arguments, 2)
            return PlaylistRemoveArguments(
                self._reference(arguments[0]),
                self._reference(arguments[1]),
            )
        if action in {"preview", "预览"}:
            self._require_count(arguments, 1)
            return PlaylistPreviewArguments(arguments[0])
        if action in {"import", "导入"}:
            self._require_minimum(arguments, 1)
            options = list(arguments[1:])
            allow_partial = "--partial" in options
            options = [value for value in options if value != "--partial"]
            if any(value.startswith("--") for value in options):
                self._invalid()
            return PlaylistImportArguments(
                arguments[0],
                " ".join(options).strip() or None,
                allow_partial,
            )
        self._invalid()

    def _source_and_value(
        self,
        arguments: tuple[str, ...],
    ) -> tuple[MusicSourceKind | None, str]:
        if not arguments:
            self._invalid()
        values = arguments
        source: MusicSourceKind | None = None
        first = values[0].casefold()
        if first == "--source":
            if len(values) < 3:
                self._invalid()
            source = self._source(values[1])
            values = values[2:]
        elif first.startswith("--source="):
            source = self._source(first.partition("=")[2])
            values = values[1:]
        if any(
            value.casefold() == "--source" or value.casefold().startswith("--source=")
            for value in values
        ):
            self._invalid()
        return source, self._joined(values)

    def _source(self, value: str) -> MusicSourceKind | None:
        try:
            return self._SOURCE_ALIASES[value.casefold()]
        except KeyError:
            self._invalid()

    def _reference(self, value: str) -> str:
        try:
            UUID(value)
        except ValueError:
            normalized = value.removeprefix("#")
            if not normalized.isdecimal() or int(normalized) < 1:
                self._invalid()
        return value

    def _joined(self, values: tuple[str, ...]) -> str:
        result = " ".join(values).strip()
        if not result:
            self._invalid()
        return result

    def _require_count(self, values: tuple[str, ...], count: int) -> None:
        if len(values) != count:
            self._invalid()

    def _require_minimum(self, values: tuple[str, ...], count: int) -> None:
        if len(values) < count:
            self._invalid()

    def _invalid(self) -> None:
        raise CommandUsageError(self._usage_text, include_usage=False)
