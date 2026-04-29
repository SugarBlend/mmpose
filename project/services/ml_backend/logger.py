import os
import logging
import colorlog


class CustomLogger(logging.Logger):

    log_colors = {
        "DEBUG": "green",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red"
    }

    def __init__(self, name: str, level: int = logging.DEBUG) -> None:
        super().__init__(name, level)

        if not self.handlers:
            handler = colorlog.StreamHandler()

            formatter = colorlog.ColoredFormatter(
                fmt="%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - "
                    "%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%H:%M:%S",
                log_colors=self.log_colors
            )

            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG) # handler passes everything, the logger filters it itself

            self.addHandler(handler)


def get_logger(name: str = __name__) -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger = CustomLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    return logger
