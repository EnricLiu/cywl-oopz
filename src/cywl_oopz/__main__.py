"""Command-line entry point."""

import asyncio
import logging

from .application import BotApplication
from .core.errors import ConfigurationError
from .settings import AppSettings


def main() -> None:
    """Load configuration and run the bot until it disconnects."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = AppSettings.from_environment()
        logging.info(settings)
        application = BotApplication(settings)
    except ConfigurationError as exc:
        logging.getLogger(__name__).error("Configuration error: %s", exc)
        raise SystemExit(2) from exc
    asyncio.run(application.run())


if __name__ == "__main__":
    main()
