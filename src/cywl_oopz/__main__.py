"""Command-line entry point."""

import asyncio
import logging
import os

from .application import BotApplication
from .core.errors import ConfigurationError
from .settings import AppSettings

logger = logging.getLogger(__name__)


def main() -> None:
    """Load configuration and run the bot until it disconnects."""
    logging.basicConfig(
        level=os.getenv("CYWL_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = AppSettings.from_environment()
        logger.info(
            "Starting CYWL OOPZ: agent=%s live_display=%s music=%s web_search=%s browser=%s",
            settings.agent.enabled,
            settings.agent.live_display,
            settings.music.enabled,
            settings.web.search_enabled,
            settings.web.browser_enabled,
        )
        application = BotApplication(settings)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", type(exc).__name__)
        raise SystemExit(2) from exc
    asyncio.run(application.run())


if __name__ == "__main__":
    main()
