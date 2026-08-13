"""Test helpers that exercise the real OOPZ command projection boundary."""

from __future__ import annotations

from typing import Any

from cywl_oopz.commands.parsing import CommandTextParser
from cywl_oopz.commands.router import CommandRouter
from cywl_oopz.integrations.oopz.command_requests import OopzCommandRequestFactory


async def dispatch_command(
    router: CommandRouter,
    message: Any,
    context: Any,
) -> bool:
    """Project and dispatch one fake OOPZ event using production adapters."""
    factory = OopzCommandRequestFactory(CommandTextParser(router.prefix))
    request = factory.from_message(message, context)
    if request is None:
        return False
    return (await router.dispatch_request(request)).consumed
