from cywl_oopz.features.agent.tool_progress import ToolProgressFormatter


def test_request_summary_keeps_useful_fields_and_redacts_sensitive_values() -> None:
    formatter = ToolProgressFormatter()

    detail = formatter.request(
        "search_web",
        {
            "query": "  初音未来 最新演出  ",
            "time_range": "w",
            "api_key": "must-not-leak",
        },
    )

    assert detail == "查询：「初音未来 最新演出」 · 时段：w"
    assert "must-not-leak" not in detail


def test_result_summary_explains_success_and_known_errors_without_raw_payloads() -> None:
    formatter = ToolProgressFormatter()

    succeeded = formatter.result(
        "search_web",
        {
            "ok": True,
            "data": {
                "results": [
                    {"title": "one", "snippet": "private body"},
                    {"title": "two", "snippet": "private body"},
                ]
            },
        },
        succeeded=True,
    )
    failed = formatter.result(
        "search_web",
        {"ok": False, "error": "web_search_unavailable"},
        succeeded=False,
    )
    unknown = formatter.result(
        "lookup",
        {"ok": False, "error": "private_internal_error"},
        succeeded=False,
    )

    assert succeeded == "找到 2 条结果"
    assert failed == "错误：网页搜索服务暂不可用"
    assert unknown == "错误：工具执行失败"
    assert "private" not in succeeded + failed + unknown


def test_browser_and_music_results_have_compact_human_summaries() -> None:
    formatter = ToolProgressFormatter()

    page = formatter.result(
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
    music = formatter.result(
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

    assert page == "当前页面「Example Domain」"
    assert music == "歌曲「Tell Your World」 · 队列第 2 位"
    assert "snapshot" not in page
    assert "source_id" not in music
