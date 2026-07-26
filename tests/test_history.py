import pytest

from cywl_oopz.features.chat.history import ChatInputTooLongError, HistoryTrimmer
from cywl_oopz.features.chat.models import ChatMessage, ChatRole


def message(content: str) -> ChatMessage:
    return ChatMessage(ChatRole.USER, content)


def test_history_trimmer_keeps_newest_messages_within_count_limit() -> None:
    trimmer = HistoryTrimmer(max_messages=3, max_characters=100)

    result = trimmer.trim(tuple(message(str(index)) for index in range(5)))

    assert [item.content for item in result] == ["2", "3", "4"]


def test_history_trimmer_keeps_only_complete_messages_within_character_limit() -> None:
    trimmer = HistoryTrimmer(max_messages=10, max_characters=5)

    result = trimmer.trim((message("old"), message("new")))

    assert [item.content for item in result] == ["new"]


def test_history_trimmer_rejects_oversized_current_input() -> None:
    trimmer = HistoryTrimmer(max_messages=10, max_characters=5)

    with pytest.raises(ChatInputTooLongError):
        trimmer.validate_input("too long")


def test_history_trimmer_drops_orphaned_leading_assistant_message() -> None:
    trimmer = HistoryTrimmer(max_messages=2, max_characters=100)
    messages = (
        ChatMessage(ChatRole.USER, "previous question"),
        ChatMessage(ChatRole.ASSISTANT, "previous answer"),
        ChatMessage(ChatRole.USER, "current question"),
    )

    result = trimmer.trim(messages)

    assert result == (ChatMessage(ChatRole.USER, "current question"),)
