import pandas as pd

# --- Base Directory Path ---
BASE_PATH = r'C:\Users\HP\olist_project\data\raw\\'

# 1. Load All Datasets
df_orders = pd.read_csv(BASE_PATH + 'olist_orders_dataset.csv')
df_items = pd.read_csv(BASE_PATH + 'olist_order_items_dataset.csv')
df_products = pd.read_csv(BASE_PATH + 'olist_products_dataset.csv')
df_payments = pd.read_csv(BASE_PATH + 'olist_order_payments_dataset.csv')
df_category = pd.read_csv(BASE_PATH + 'product_category_name_translation.csv')

# 2. Arranding datasets 

# a. Merge datasets 
#  Orders and Items, then add details (Payments, Reviews, Customers, Products, Sellers, Translation)
merge_df = pd.merge(df_orders, df_items, on='order_id', how='left')
merge_df = pd.merge(merge_df, df_products, on='product_id', how='left')
merge_df = pd.merge(merge_df, df_payments, on='order_id', how='left')
merge_df = pd.merge(merge_df, df_category, on='product_category_name', how='left')

print("--- Merged Dataset ---")
print(f"Merged DataFrame Shape: {merge_df.shape}")
print(merge_df.columns.tolist())
print("------------------------------------")

# b. Dropping irrelevant columns
TARGET_COLUMNS = [
    'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
    'product_id', 'price', 'freight_value',  'payment_sequential', 'payment_type', 
    'payment_installments', 'payment_value', 'product_category_name_english'
]

# Select only the specified columns and overwrite the master_df
master_df = merge_df[TARGET_COLUMNS]

print("--- Master Dataset ---")
print(f"Master DataFrame Shape after column selection: {master_df.shape}")
print(master_df.columns.tolist())
print("------------------------------------")

# 3. Data Cleaning 
# A. Missing Values Check (Before Cleaning)
missing_counts = master_df.isnull().sum()
print(missing_counts)