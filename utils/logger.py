import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def log_info(message: str, data: str = ""):
    """Log info level message"""
    if data:
        logger.info(f"{message} {data}")
    else:
        logger.info(message)

def log_warn(message: str, data: str = ""):
    """Log warning level message"""
    if data:
        logger.warning(f"{message} {data}")
    else:
        logger.warning(message)

def log_error(message: str, error=None):
    """Log error level message"""
    if error:
        logger.error(f"{message} {str(error)}")
    else:
        logger.error(message)

def log_debug(message: str, data: str = ""):
    """Log debug level message"""
    if data:
        logger.debug(f"{message} {data}")
    else:
        logger.debug(message)
