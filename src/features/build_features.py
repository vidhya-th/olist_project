import pandas as pd
import numpy as np

processed_data_path = r'C:\Users\HP\olist_project\data\processed\pricing_engine_master_data_cleaned.csv'
output_features_path = r'src\features\pricing_model_features_v2.csv'

# 1. Load the cleaned master dataset
master_df = pd.read_csv(processed_data_path)


# 2. Initialize Features DataFrame 
#Existing feautures
features_df = master_df[['order_id', 'customer_id', 'product_id', 'price', 'freight_value','payment_installments']].copy()
#Calculated feautures
#discount
discount = (master_df['price'] - master_df['payment_value']) / master_df['price']
features_df['discount_ratio'] = discount
#purchase time
features_df['purchase_dayofweek'] = master_df['order_purchase_timestamp'].dt.dayofweek
features_df['purchase_month'] = master_df['order_purchase_timestamp'].dt.month
features_df['purchase_year'] = master_df['order_purchase_timestamp'].dt.month


## most bpurchased product
"""
top_most_sold_categories = features_df['most_sold_category'].value_counts().nlargest(20).index
features_df['most_sold_category_group'] = np.where(features_df['most_sold_category'].isin(top_most_sold_categories), features_df['most_sold_category'], 'category_other')
most_sold_category_ohe = pd.get_dummies(features_df['most_sold_category_group'], prefix='most_sold_cat')
features_df = pd.concat([features_df, most_sold_category_ohe], axis=1)
"""
## time since last order
"""
temp_df = master_df[['customer_id', 'order_purchase_timestamp']].sort_values(
    by=['customer_id', 'order_purchase_timestamp']).copy()
previous_purchase_date = temp_df.groupby('customer_id')['order_purchase_timestamp'].shift(1)
time_since_last_order_days = (temp_df['order_purchase_timestamp'] - previous_purchase_date).dt.days
time_since_last_order_days.fillna(365, inplace=True) 
features_df['time_since_last_order_days'] = time_since_last_order_days.set_axis(master_df.index)
"""

# Save Features
final_features_df.to_csv(output_features_path)