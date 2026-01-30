# src/logger.py
import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger("OlistPricing")
logger.setLevel(logging.INFO)

# Avoid duplicate logs if this module is imported multiple times
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

"""
from src.logger import logger

def ingest_data():
    try:
        logger.info("Starting data ingestion from Olist dataset...")
        # Your code here
    except Exception as e:
        logger.error(f"Error in data ingestion: {str(e)}")

        """