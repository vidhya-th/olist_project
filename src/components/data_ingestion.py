import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import OlistException
from src.logger import logger

@dataclass
class DataIngestionConfig:
    """Defines paths for saving the ingested data outputs."""
    train_data_path: str = os.path.join('data', 'processed', 'train.csv')
    test_data_path: str = os.path.join('data', 'processed', 'test.csv')
    raw_data_path: str = os.path.join('data', 'processed', 'master_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the Data Ingestion component")
        try:
            # 1. Load Raw CSVs from data/raw/
            # Update filenames based on your actual raw folder content
            items = pd.read_csv('C:/olist_project/data/raw/olist_order_items_dataset.csv')
            orders = pd.read_csv('C:/olist_project/data/raw/olist_orders_dataset.csv')
            products = pd.read_csv('C:/olist_project/data/raw/olist_products_dataset.csv')
            translation = pd.read_csv('C:/olist_project/data/raw/product_category_name_translation.csv')

            logger.info("Raw datasets loaded. Starting merge for Master Dataset.")

            # 2. Merge Logic (Creating the 112K x 22 structure)
            df = items.merge(orders, on='order_id')
            df = df.merge(products, on='product_id')
            df = df.merge(translation, on='product_category_name')

            logger.info(f"Master dataset merged. Shape: {df.shape}")

            # 3. Save the full master dataset
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # 4. Train-Test Split
            logger.info("Initiating Train-Test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info("Ingestion of data is completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise OlistException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()