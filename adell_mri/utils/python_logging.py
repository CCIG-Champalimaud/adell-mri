import logging
import os

logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given name. Repeated calls with
    the same name return the same logger without adding duplicate handlers.

    Args:
        name (str): logger name (usually ``__name__``).

    Returns:
        logging.Logger: configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
