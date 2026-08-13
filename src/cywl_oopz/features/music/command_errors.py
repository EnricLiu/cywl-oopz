"""User-safe presentation for expected music command failures."""

from __future__ import annotations

from cywl_oopz.core.errors import DatabaseError

from .errors import (
    MusicAreaRequiredError,
    MusicAuthenticationRequiredError,
    MusicCatalogError,
    MusicExtractionTimeoutError,
    MusicGeoRestrictedError,
    MusicLiveUnsupportedError,
    MusicNoAudioFormatError,
    MusicNotFoundError,
    MusicPlaybackError,
    MusicPlaylistConflictError,
    MusicPlaylistEmptyError,
    MusicPlaylistFullError,
    MusicPlaylistNameError,
    MusicPlaylistNotFoundError,
    MusicQueryError,
    MusicQueueFullError,
    MusicReferenceError,
    MusicSourceDisabledError,
    MusicSourceRateLimitedError,
    MusicSourceUnavailableError,
    MusicTrackTooLongError,
    MusicUnsupportedContentError,
    MusicVoiceBusyError,
    MusicVoiceChannelRequiredError,
    NeteasePlaylistIncompleteError,
    NeteasePlaylistNotFoundError,
    NeteasePlaylistReferenceError,
    NeteasePlaylistTooLargeError,
)


class MusicCommandErrorPresenter:
    """Translate documented domain failures without hiding programming errors."""

    def message(self, error: Exception) -> str:
        if isinstance(error, MusicVoiceChannelRequiredError):
            return "请先加入当前 area 的语音频道。"
        if isinstance(error, MusicVoiceBusyError):
            return "语音频道正被实时对话或其他功能使用，请稍后再试。"
        if isinstance(error, MusicQueueFullError):
            return "播放队列已满，请先播放、跳过或清空一些歌曲。"
        if isinstance(error, MusicNotFoundError):
            return "没有找到可用的歌曲。"
        if isinstance(error, MusicSourceDisabledError):
            return "这个音乐来源当前没有启用，可用来源请查看 music sources。"
        if isinstance(error, MusicAuthenticationRequiredError):
            return "这首内容需要来源账号权限；当前没有可用登录配置。"
        if isinstance(error, MusicGeoRestrictedError):
            return "这首内容在 Bot 当前地区不可用。"
        if isinstance(error, MusicSourceRateLimitedError):
            return "音乐来源暂时限制了请求，请稍后重试或直接粘贴单曲链接。"
        if isinstance(error, MusicLiveUnsupportedError):
            return "暂不支持直播、预约直播或首映中的内容。"
        if isinstance(error, MusicTrackTooLongError):
            return "这段内容超过了允许的最长播放时间。"
        if isinstance(error, MusicNoAudioFormatError):
            return "没有找到可播放的音频格式。"
        if isinstance(error, MusicUnsupportedContentError):
            return "暂不支持这种内容形态（例如合集、互动视频或 DRM 内容）。"
        if isinstance(error, MusicReferenceError):
            return "无法识别这个音乐链接或来源 ID。"
        if isinstance(error, MusicExtractionTimeoutError):
            return "读取音乐页面超时，请稍后重试。"
        if isinstance(error, MusicSourceUnavailableError):
            return "这个音乐来源暂时不可用，请稍后重试。"
        if isinstance(error, MusicCatalogError):
            return "音乐目录暂时不可用，请稍后重试。"
        if isinstance(error, MusicPlaylistConflictError):
            return "当前 area 已有同名共享歌单。"
        if isinstance(error, MusicPlaylistNotFoundError):
            return "当前 area 没有这个共享歌单或条目，请重新查看歌单列表。"
        if isinstance(error, MusicPlaylistFullError):
            return "这个共享歌单已满。"
        if isinstance(error, MusicPlaylistEmptyError):
            return "这个共享歌单还是空的，暂时不能载入播放。"
        if isinstance(error, MusicPlaylistNameError):
            return "歌单名称不能为空，且最多 80 个字符。"
        if isinstance(error, MusicAreaRequiredError):
            return "共享歌单命令只能在 OOPZ area 频道中使用。"
        if isinstance(error, NeteasePlaylistReferenceError):
            return "无法识别网易云歌单 ID 或链接。"
        if isinstance(error, NeteasePlaylistNotFoundError):
            return "没有找到这个网易云歌单，或它当前不可见。"
        if isinstance(error, (NeteasePlaylistIncompleteError, NeteasePlaylistTooLargeError)):
            return (
                "这个网易云歌单只能部分导入。请先 preview，确认后在 import 命令中加入 --partial。"
            )
        if isinstance(error, MusicQueryError):
            return "音乐命令参数不正确，请缩短关键词或检查播放模式。"
        if isinstance(error, MusicPlaybackError):
            return "音乐播放操作失败，请稍后重试。"
        if isinstance(error, DatabaseError):
            return "共享歌单服务暂时不可用，请稍后重试。"
        return "音乐操作失败，请稍后重试。"
