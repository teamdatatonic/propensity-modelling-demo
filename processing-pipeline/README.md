# Brand Propensity Model for Major Retailer

> The purpose of these scripts are to carry out the data processing and
> feature engineering in **BigQuery**.


### Raw files required:
------
These were loaded into BigQuery from Cloud Storage from CSV format.
Big Query dataset: `propensity_dataset`
- history (promotional information)
- transactions (transaction table)


### Files required in package:
------

Ad-hoc SQL queries:
- `supporting-queries.sql`

SQL queries for data preprocessing:
- `a010_impute_missing_values.sql`
- `a020_remove_zero_quantity_rows.sql`
- `a030_create_returned_flag.sql`
- `a040_absolute_values.sql`
- `a050_create_transaction_id.sql`
- `a060_create_product_id.sql`
- `a070_create_product_price.sql`

SQL queries for feature engineering:
- `f010_top_brands.sql`
- `x010_cross_join.sql`
- `f020_label_creation.sql`
- `f030_promo_sensitive_feature.sql`
- `f040_brand_seasonality.sql`
- `f050_create_brand_features.sql`
- `f060_create_overall_features.sql`
- `f070_compute_aov.sql`
- `f080_impute_nulls.sql`
- `f090_type_cast.sql`

SQL queries for splitting into train/dev/test, generating baseline and downsampling:
- `t010_train_test_field.sql`
- `t020_train_data.sql`
- `t030_test_data.sql`
- `t040_test_dev_split.sql`
- `t050_dev_data.sql`
- `t060_test_data_final.sql`
- `b010_create_baseline.sql`
- `b020_join_baseline.sql`
- `b030_baseline_metrics.sql`
- `s010_downsampling.sql`
- `s020_downsampled_features.sql`

Bash script which references SQL queries and runs them sequentially:
- `bq-processing.sh`


### To excute:
------
Please enter the GCP project and dataset name as command-line arguments.

1) Create a dataset in BigQuery
2) Load the "transactions" and "history" tables from Cloud Storage into this dataset
3) Execute the script with the GCP project and dataset as arguments

```
bash bq-processing.sh <GCP PROJECT> <BQ DATASET>
```

### Running time:
------
Prefix the above command with `/usr/bin/time` to compute the running time.

Total running time: **~25 minutes**
