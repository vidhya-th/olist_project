# Processed Data Description: pricing_engine_master_data_cleaned.csv

This file contains the merged, cleaned, and filtered historical transaction data used as the base for the Dynamic Pricing Engine. It focuses on successful ('delivered') orders and the features essential for modeling demand elasticity and conversion.


## Key Columns (Features & Target)

| Column Name | Source Dataset | Description / Relevance to Pricing |
| :--- | :--- | :--- |
| **order_purchase_timestamp** | Orders | **CRITICAL:** Time-series feature for dynamic pricing analysis (when the sale occurred). |
| **price** | Order Items | **TARGET:** The historical price accepted by the customer. Used as the target variable for regression models. |
| **freight_value** | Order Items | The cost of shipping. Used as a cost feature. |
| **product_category_name_english** | Translation | High-level feature defining demand elasticity. Missing values are filled with 'unknown'. |
| **payment_installments** | Payments | Customer behavior feature. Used to model willingness to pay a higher price over time. |
| **order_status** | Orders | Kept to confirm all rows are 'delivered' (cleaned data). |
| **product_id** | Order Items | Unique identifier for specific products. |
| **order_id** | Orders | Unique identifier for the transaction. |