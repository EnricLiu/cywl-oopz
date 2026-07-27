"""Opt-in live checks for OOPZ message editing.

Run with ``CYWL_RUN_LIVE_OOPZ_TESTS=1 uv run pytest
tests/test_oopz_editable_messages_live.py -q``. The test recalls every message it
creates and never logs message text or credentials.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from dotenv import find_dotenv, load_dotenv
from oopz_sdk import OopzBot, OopzConfig

from cywl_oopz.integrations.oopz.editable_messages import (
    MessageAddress,
    OopzEditableMessageGateway,
)


def _live_enabled() -> bool:
    return os.environ.get("CYWL_RUN_LIVE_OOPZ_TESTS", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.mark.asyncio
async def test_live_channel_message_can_be_created_edited_and_recalled() -> None:
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_OOPZ_TESTS=1 to run the OOPZ live test")
    load_dotenv(find_dotenv(usecwd=True), override=False)
    bot = OopzBot(await OopzConfig.from_env_async())
    gateway = OopzEditableMessageGateway(bot)
    created = None
    attempted_types: list[str] = []
    last_error: Exception | None = None
    try:
        areas = await bot.areas.get_joined_areas()
        if not areas:
            pytest.skip("the OOPZ account has no joined area")
        for area in areas:
            groups = await bot.areas.get_area_channels(area.area_id)
            for channel in (
                channel
                for group in groups
                for channel in group.channels
                if channel.channel_type.casefold() not in {"voice", "category"}
            ):
                attempted_types.append(channel.channel_type)
                address = MessageAddress(
                    scope="channel",
                    area_id=area.area_id,
                    channel_id=channel.channel_id,
                    target_person_id="",
                    reference_message_id="",
                )
                try:
                    created = await gateway.create_reply(
                        address,
                        "🧪 CYWL 消息编辑协议验证中…",
                    )
                except Exception as exc:
                    last_error = exc
                    continue
                break
            if created is not None:
                break
        if created is None:
            pytest.skip(
                "no joined text channel accepted a temporary bot message; "
                f"types={sorted(set(attempted_types))!r}; "
                f"last_error={type(last_error).__name__}: {last_error}"
            )

        for index in range(10):
            await gateway.replace(
                created,
                f"**协议验证** ♪\n第 {index + 1}/10 次节流编辑",
            )
            if index < 9:
                await asyncio.sleep(0.8)

        await gateway.replace(created, "a" * 1999)
        await gateway.replace(created, "b" * 2000)
        await gateway.replace(created, "c" * 2001)
        messages = await bot.messages.get_channel_messages(
            created.area_id,
            created.channel_id,
            size=50,
        )
        over_limit = next(
            message for message in messages if message.message_id == created.message_id
        )
        assert (over_limit.text or over_limit.content) == "c" * 2001

        await gateway.replace(created, "😀" * 1000)
        await gateway.replace(created, "😀" * 1001)

        await gateway.replace(created, "**协议验证完成** ♪")
        messages = await bot.messages.get_channel_messages(
            created.area_id,
            created.channel_id,
            size=50,
        )
        edited = next(
            (message for message in messages if message.message_id == created.message_id),
            None,
        )
        assert edited is not None
        assert (edited.text or edited.content) == "**协议验证完成** ♪"
        assert edited.edit_time > 0
    finally:
        if created is not None:
            await bot.messages.recall_message(
                created.message_id,
                area=created.area_id,
                channel=created.channel_id,
                timestamp=created.timestamp,
            )
        await bot.rest.close()


@pytest.mark.asyncio
async def test_live_private_message_can_be_created_edited_and_recalled() -> None:
    if not _live_enabled():
        pytest.skip("set CYWL_RUN_LIVE_OOPZ_TESTS=1 to run the OOPZ live test")
    load_dotenv(find_dotenv(usecwd=True), override=False)
    bot = OopzBot(await OopzConfig.from_env_async())
    gateway = OopzEditableMessageGateway(bot)
    created = None
    try:
        target = os.environ.get(
            "CYWL_OOPZ_LIVE_PRIVATE_TARGET",
            str(bot.config.person_uid),
        ).strip()
        try:
            session = await bot.messages.open_private_session(target)
        except Exception as exc:
            pytest.skip(f"no private test session is available: {type(exc).__name__}: {exc}")
        address = MessageAddress(
            scope="private",
            area_id="",
            channel_id=session.session_id,
            target_person_id=target,
            reference_message_id="",
        )
        created = await gateway.create_reply(address, "🧪 CYWL 私信编辑协议验证中…")
        await gateway.replace(created, "**私信协议验证完成** ♪")
    finally:
        if created is not None:
            await bot.messages.recall_private_message(
                created.message_id,
                channel=created.channel_id,
                target=created.target_person_id,
                timestamp=created.timestamp,
            )
        await bot.rest.close()
