import sys
from loguru import logger
from typing import Optional
import json
import logging

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging(
    json_logs: bool = False,
    log_level: str = "INFO",
):
    """Setup logging configuration"""
    # Clear existing handlers
    logging.root.handlers = [InterceptHandler()]
    
    # Set log level for all loggers
    logging.root.setLevel(log_level)
    
    # Remove all other loggers
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    
    # Configure loguru
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "level": log_level,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
                if not json_logs
                else lambda msg: json.dumps({
                    "time": msg.record["time"].isoformat(),
                    "level": msg.record["level"].name,
                    "name": msg.record["name"],
                    "function": msg.record["function"],
                    "line": msg.record["line"],
                    "message": msg.record["message"],
                    **msg.record["extra"]
                })
            }
        ]
    )
    
    # Uvicorn logger configuration
    for _log in ["uvicorn", "uvicorn.error", "fastapi"]:
        _logger = logging.getLogger(_log)
        _logger.handlers = [InterceptHandler()]

    return logger

# Initialize logger
logger = setup_logging()
