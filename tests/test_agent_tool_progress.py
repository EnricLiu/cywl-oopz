from cywl_oopz.features.agent.tool_progress import ToolProgressCatalog


def test_request_summary_keeps_useful_fields_and_redacts_sensitive_values() -> None:
    catalog = ToolProgressCatalog()

    presentation = catalog.request(
        "search_web",
        {
            "query": "  初音未来 最新演出  ",
            "time_range": "w",
            "api_key": "must-not-leak",
        },
    )

    assert presentation.subject == "「初音未来 最新演出」"
    assert presentation.summary == ""
    assert "must-not-leak" not in repr(presentation)


def test_result_summary_explains_success_and_known_errors_without_raw_payloads() -> None:
    catalog = ToolProgressCatalog()

    succeeded = catalog.result(
        "search_web",
        {
            "ok": True,
            "data": {
                "results": [
                    {
                        "title": "one",
                        "url": "https://example.com/one",
                        "snippet": "private body",
                    },
                    {
                        "title": "two",
                        "url": "https://example.com/two",
                        "snippet": "private body",
                    },
                ]
            },
        },
        succeeded=True,
    )
    failed = catalog.result(
        "search_web",
        {"ok": False, "error": "web_search_unavailable"},
        succeeded=False,
    )
    unknown = catalog.result(
        "lookup",
        {"ok": False, "error": "private_internal_error"},
        succeeded=False,
    )

    assert succeeded.summary == "找到 2 条结果"
    assert succeeded.items == (
        "https://example.com/one",
        "https://example.com/two",
    )
    assert failed.summary == "网页搜索服务暂不可用"
    assert unknown.summary == "工具执行失败"
    assert "private" not in repr((succeeded, failed, unknown))


def test_browser_and_music_results_have_compact_human_summaries() -> None:
    catalog = ToolProgressCatalog()

    page = catalog.result(
        "browser_open",
        {
            "ok": True,
            "data": {
                "title": "Example Domain",
                "url": "https://example.com",
                "snapshot": "must-not-be-shown",
            },
        },
        succeeded=True,
    )
    music = catalog.result(
        "enqueue_music",
        {
            "ok": True,
            "data": {
                "position": 2,
                "track": {"title": "Tell Your World", "source_id": "private"},
            },
        },
        succeeded=True,
    )
    queue = catalog.result(
        "get_music_queue",
        {
            "ok": True,
            "data": {
                "current": {"track": {"title": "Tell Your World"}},
                "upcoming": [{"track": {"title": "39"}}],
                "mode": "shuffle",
            },
        },
        succeeded=True,
    )
    mode = catalog.result(
        "set_music_playback_mode",
        {
            "ok": True,
            "data": {"mode": "repeat_all", "changed": True},
        },
        succeeded=True,
    )
    playlists = catalog.result(
        "list_music_playlists",
        {
            "ok": True,
            "data": {
                "playlists": [
                    {"id": "one", "name": "夜间电台", "track_count": 3},
                    {"id": "two", "name": "V家", "track_count": 8},
                ]
            },
        },
        succeeded=True,
    )
    loaded_playlist = catalog.result(
        "load_music_playlist",
        {
            "ok": True,
            "data": {
                "playlist_name": "夜间电台",
                "loaded_count": 3,
                "voice_channel_id": "private",
            },
        },
        succeeded=True,
    )
    preview_import = catalog.result(
        "preview_netease_playlist",
        {
            "ok": True,
            "data": {
                "name": "Miku Favorites",
                "declared_track_count": 80,
                "visible_track_count": 50,
                "complete": False,
            },
        },
        succeeded=True,
    )
    imported = catalog.result(
        "import_netease_playlist",
        {
            "ok": True,
            "data": {
                "playlist": {"name": "Miku Favorites"},
                "imported_track_count": 50,
                "skipped_track_count": 30,
            },
        },
        succeeded=True,
    )

    assert page.summary == "Example Domain"
    assert music.summary == "歌曲「Tell Your World」 · 队列第 2 位"
    assert queue.summary == "正在播放 · 后续 1 首 · 随机播放"
    assert mode.summary == "列表循环已设置"
    assert playlists.summary == "找到 2 个共享歌单"
    assert loaded_playlist.summary == "歌单「夜间电台」· 已载入 3 首"
    assert preview_import.summary == ("歌单「Miku Favorites」· 可导入 50/80 首 · 需要确认部分导入")
    assert imported.summary == "歌单「Miku Favorites」· 已导入 50 首 · 跳过 30 首"
    assert "voice_channel_id" not in repr(loaded_playlist)
    assert "snapshot" not in repr(page)
    assert "source_id" not in repr(music)


def test_web_page_request_uses_host_and_result_exposes_bounded_preview() -> None:
    catalog = ToolProgressCatalog()

    request = catalog.request(
        "read_web_page",
        {"url": "https://www.baidu.com/search?q=miku"},
    )
    result = catalog.result(
        "read_web_page",
        {
            "ok": True,
            "data": {
                "title": "百度一下，你就知道",
                "url": "https://www.baidu.com/",
                "content": "第一行\n\n第二行\n第二行\n第三行\n第四行",
                "truncated": False,
            },
        },
        succeeded=True,
    )

    assert request.subject == "www.baidu.com"
    assert result.summary == "百度一下，你就知道"
    assert result.preview_lines == ("第一行", "第二行", "第三行")


def test_skill_progress_shows_identity_and_size_without_loaded_content() -> None:
    catalog = ToolProgressCatalog()

    request = catalog.request("load_agent_skill", {"name": "web-research"})
    loaded = catalog.result(
        "load_agent_skill",
        {
            "ok": True,
            "data": {
                "skill": {
                    "name": "web-research",
                    "display_name": "网页研究",
                    "version": "2",
                },
                "instructions": "must-not-be-shown",
                "character_count": 3210,
                "already_loaded": False,
            },
        },
        succeeded=True,
    )
    resource = catalog.result(
        "read_agent_skill_resource",
        {
            "ok": True,
            "data": {
                "resource": {
                    "key": "source-guide",
                    "display_name": "来源指南",
                },
                "content": "must-not-be-shown",
                "character_count": 840,
                "already_loaded": False,
            },
        },
        succeeded=True,
    )
    failed = catalog.result(
        "load_agent_skill",
        {"ok": False, "error": "skill_context_limit"},
        succeeded=False,
    )

    assert request.subject == "web-research"
    assert loaded.subject == "网页研究"
    assert loaded.summary == "v2 · 3.2k 字"
    assert resource.subject == "来源指南"
    assert resource.summary == "840 字"
    assert failed.summary == "本轮技能内容已达上限"
    assert "must-not-be-shown" not in repr((loaded, resource, failed))
