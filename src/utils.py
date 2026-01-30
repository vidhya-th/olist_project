
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.logger import logger 
from src.exception import OlistException

# MASTER DATASET CONSTANTS (112k x 22 from exploration.ipynb)
REQUIRED_COLS = [
    'order_id', 'order_item_id', 'product_id', 'seller_id', 'shipping_limit_date',
    'price', 'freight_value', 'customer_id', 'order_status', 'order_purchase_timestamp',
    'order_approved_at', 'order_delivered_customer_date', 'order_estimated_delivery_date',
    'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
    'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
    'product_category'
]

class OlistDataUtils:
    """Data loading & validation"""
    
    @staticmethod
    def load_master_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
        """Load/create master dataset with your logger"""
        processed_path = "C:/olist_project/data/processed/olist_master_dataset.csv"
        
        try:
            df = pd.read_csv(processed_path)
            logger.info(f" Loaded master dataset: {df.shape}")
        except FileNotFoundError:
            logger.info(" Creating master dataset from raw files...")
            df = OlistDataUtils._create_master_dataset(raw_dir)
            df.to_csv(processed_path, index=False)
            logger.info(f"Saved master dataset: {df.shape}")
        
        OlistDataUtils.validate_master_df(df)
        return df
    
    @staticmethod
    def _create_master_dataset(raw_dir: str) -> pd.DataFrame:
        """Exact replica of exploration.ipynb merge logic"""
        files = {
            'order_items': f"{raw_dir}/olist_order_items_dataset.csv",
            'orders': f"{raw_dir}/olist_orders_dataset.csv", 
            'products': f"{raw_dir}/olist_products_dataset.csv",
            'category_translation': f"{raw_dir}/product_category_name_translation.csv"
        }
        
        logger.info(" Loading raw Olist datasets...")
        order_items = pd.read_csv(files['order_items'])
        orders = pd.read_csv(files['orders'])
        products = pd.read_csv(files['products'])
        category_translation = pd.read_csv(files['category_translation'])
        
        logger.info(" Merging datasets...")
        master = (order_items.merge(orders, on='order_id', how='left')
                         .merge(products, on='product_id', how='left')
                         .merge(category_translation, on='product_category_name', how='left'))
        
        # Clean exactly like notebook
        master.drop(columns=['product_category_name'], inplace=True)
        master.rename(columns={'product_category_name_english': 'product_category'}, inplace=True)
        master['product_category'] = master['product_category'].fillna('unknown')
        
        date_cols = ['order_purchase_timestamp', 'order_approved_at',
                    'order_delivered_customer_date', 'order_estimated_delivery_date']
        for col in date_cols:
            master[col] = pd.to_datetime(master[col])
            
        logger.info(f"Master dataset created: {master.shape}")
        return master
    
    @staticmethod
    def validate_master_df(df: pd.DataFrame) -> None:
        """Validate expected shape & columns"""
        if df.shape != (112650, 22):
            logger.warning(f"Dataset shape {df.shape} != expected (112650, 22)")
        
        missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing_cols:
            raise OlistException(ValueError(f"Missing columns: {missing_cols}"), 
                               "Master dataset validation")


class PricingUtils:
    """Flash sale pricing calculations"""
    
    @staticmethod
    def calculate_revenue(price: float, freight_value: float) -> float:
        """Olist revenue = price + freight_value"""
        return price + freight_value
    
    @staticmethod
    def generate_discount_candidates(base_price: float, max_discount: float = 0.45) -> List[float]:
        """5-45% discount candidates for policy optimization"""
        discounts = np.arange(0.05, max_discount + 0.05, 0.05)
        return [base_price * (1 - d) for d in discounts]


class TimeUtils:
    """Flash sale timing features (11-16h peak from EDA)"""
    
    @staticmethod
    def extract_flash_sale_features(df: pd.DataFrame) -> pd.DataFrame:
        """Peak hours 11-16, weekdays for flash sales"""
        ts_col = 'order_purchase_timestamp'
    
       
        logger.info(f" Converting {ts_col} to datetime...")
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')

        logger.info(" Extracting time features...")
        df['hour'] = df[ts_col].dt.hour
        df['day_of_week'] = df[ts_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df[ts_col].dt.month
        df['is_peak_hour'] = ((df['hour'] >= 11) & (df['hour'] <= 16)).astype(int)
        df['flash_sale_window'] = (
            (df['hour'].isin([11,12,13,14,15,16])) & (df['day_of_week'] < 5)
        ).astype(int)
        
        logger.info(" Time features added")
        return df


class FeatureUtils:
    """ML features for discount optimization"""
    
    @staticmethod
    def engineer_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
        """Core features for LightGBM/XGBoost"""
        logger.info(" Engineering pricing features...")
        
        # Revenue features
        df['revenue'] = PricingUtils.calculate_revenue(df['price'], df['freight_value'])
        df['revenue_per_kg'] = df['revenue'] / df['product_weight_g'].replace(0, 1)
        
        # Product complexity score
        df['product_complexity'] = (
            df['product_name_lenght'].fillna(0) + 
            df['product_description_lenght'].fillna(0) + 
            df[['product_length_cm', 'product_height_cm', 'product_width_cm']].sum(axis=1)
        )
        
        # Top categories (from EDA)
        top_cats = ['bed_bath_table', 'health_beauty', 'computers', 'phones']
        df['top_category'] = df['product_category'].isin(top_cats).astype(int)
        
        logger.info(" Pricing features engineered")
        return df


# Convenience functions (your preferred pattern)
def load_master_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Quick master dataset loader"""
    return OlistDataUtils.load_master_dataset(raw_dir)

def extract_flash_sale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick time feature extraction"""
    return TimeUtils.extract_flash_sale_features(df)

def engineer_pricing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick pricing feature engineering"""
    return FeatureUtils.engineer_pricing_features(df)
