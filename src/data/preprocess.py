import pandas as pd

# --- Base Directory Path ---
base_path = r'C:\Users\HP\olist_project\data\raw\\'

# 1. Load All Datasets
df_orders = pd.read_csv(base_path + 'olist_orders_dataset.csv')
df_items = pd.read_csv(base_path + 'olist_order_items_dataset.csv')
df_products = pd.read_csv(base_path + 'olist_products_dataset.csv')
df_payments = pd.read_csv(base_path + 'olist_order_payments_dataset.csv')
df_category = pd.read_csv(base_path + 'product_category_name_translation.csv')

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
target_columns = [
    'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 
    'product_id', 'price', 'freight_value',  'payment_sequential', 'payment_type', 
    'payment_installments', 'payment_value', 'product_category_name_english'
]

# Select only the specified columns and overwrite the master_df
master_df = merge_df[target_columns]

print("--- Master Dataset ---")
print(f"Master DataFrame Shape after column selection: {master_df.shape}")
print(master_df.columns.tolist())
print("------------------------------------")

# 3. Data Cleaning 
# A. Missing Values Check (Before Cleaning)
missing_counts = master_df.isnull().sum()
print(missing_counts)



# B. Addressing missing values 
# Drop 3 rows where payment info is missing 
master_df.dropna(subset=['payment_value'], inplace=True)

# Impute monetary values with 0 where the product ID is missing(830 rows)
missing_product_mask = master_df['product_id'].isnull()
master_df.loc[missing_product_mask, ['price', 'freight_value']] = 0


# Handle Product Category Missing Values (2553 rows)
master_df['product_category_name_english'].fillna('category_unknown', inplace=True)

# Handle Payment Type Missing Values 
master_df['payment_type'].fillna('not_applicable', inplace=True)
master_df['payment_installments'].fillna(0, inplace=True) # 0 installments for zero-value orders
master_df['payment_sequential'].fillna(1, inplace=True) # Assume single payment attempt

# Final verification
print("\nMissing values after imputation:")
print(master_df.isnull().sum())

#C.Outlier Check (Price)
Q1 = master_df['price'].quantile(0.25)
Q3 = master_df['price'].quantile(0.75)
IQR = Q3 - Q1
upper_bound_iqr = Q3 + 1.5 * IQR # Standard IQR Outlier definition
outliers_count = master_df[master_df['price'] > upper_bound_iqr].shape[0]
print(f"\n[C] Price Outliers: {outliers_count} rows above standard 1.5*IQR threshold.")


# 4. Save the resulting DataFrame to a new file in your project path
output_path = r'C:\Users\HP\olist_project\data\processed\pricing_engine_master_data_cleaned.csv'
master_df.to_csv(output_path, index=False)
print(f"\nData saved to: {output_path}")