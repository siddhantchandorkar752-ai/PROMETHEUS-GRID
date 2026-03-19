from loguru import logger
import sys

# Pehle purane handlers hatao
logger.remove()

# Naya professional format set karo
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>", level="INFO")

def get_logger():
    return logger
