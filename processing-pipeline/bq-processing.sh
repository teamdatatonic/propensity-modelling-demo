#/bin/bash
# /usr/bin/time prefix to get the time it took to run script

# script to run the data processing and feature engineering on the dataset

# enter GCP project and BigQuery dataset name as command line args
# ensure that the "transactions" and "history" table are in this dataset
project=$1
dataset_name=$2

dataset="${project}:${dataset_name}"

# table names
processed_table="cleaned"
top_brands="top_brands"
crossed_table="customerxbrand"
feature_table="features"
train="train"
test="test"
dev="dev"
baseline="baseline"
baseline_metrics="baseline_metrics"
downsampled="downsampled"

# target month for prediction
target_month="2013-03-01"

: '
Data processing and creation of reference tables.
'
# imputing missing values for products
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a010_impute_missing_values.sql > a010_impute_missing_values_temp.sql
cat a010_impute_missing_values_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results
rm a010_impute_missing_values_temp.sql

# removing rows where product quantity is zero
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a020_remove_zero_quantity_rows.sql > a020_remove_zero_quantity_rows_temp.sql
cat a020_remove_zero_quantity_rows_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results --replace=true
rm a020_remove_zero_quantity_rows_temp.sql

# creating variable for whether transaction is to return items or not
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a030_create_returned_flag.sql > a030_create_returned_flag_temp.sql
cat a030_create_returned_flag_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results --replace=true
rm a030_create_returned_flag_temp.sql

# convert product quantity and purchase amount to absolute when not marked as return
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a040_absolute_values.sql > a040_absolute_values_temp.sql
cat a040_absolute_values_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results --replace=true
rm a040_absolute_values_temp.sql

# creating pseudo transaction ID based on concatenation of columns
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a050_create_transaction_id.sql > a050_create_transaction_id_temp.sql
cat a050_create_transaction_id_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results --replace=true
rm a050_create_transaction_id_temp.sql

# creating pseudo product ID based on concatenation of columns
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a060_create_product_id.sql > a060_create_product_id_temp.sql
cat a060_create_product_id_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results --replace=true
rm a060_create_product_id_temp.sql

# creating product price for each unit as purchaseamount is multiple of quantity
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" a070_create_product_price.sql > a070_create_product_price_temp.sql
cat a070_create_product_price_temp.sql | bq query --destination_table=$dataset.$processed_table --use_legacy_sql=false --allow_large_results --replace=true
rm a070_create_product_price_temp.sql

: '
Feature engineering for demographic data.
'
# extract top 1000 brands by volume of transactions
sed "s/TRGT_MONTH/$target_month/g;s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f010_top_brands.sql > f010_top_brands_temp.sql
cat f010_top_brands_temp.sql | bq query --destination_table=$dataset.$top_brands --use_legacy_sql=false --allow_large_results
rm f010_top_brands_temp.sql

# cross join between top 1000 brands and customers who have transacted at least once with these brands
sed "s/TRGT_MONTH/$target_month/g;s/PROJECT/$project/g;s/DATASET/$dataset_name/g" x010_cross_join.sql > x010_cross_join_temp.sql
cat x010_cross_join_temp.sql | bq query --destination_table=$dataset.$crossed_table --use_legacy_sql=false --allow_large_results
rm x010_cross_join_temp.sql

# generate binary target variable
sed "s/TRGT_MONTH/$target_month/g;s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f020_label_creation.sql > f020_label_creation_temp.sql
cat f020_label_creation_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results
rm f020_label_creation_temp.sql

# generate promo sensitive feature
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f030_promo_sensitive_feature.sql > f030_promo_sensitive_feature_temp.sql
cat f030_promo_sensitive_feature_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f030_promo_sensitive_feature_temp.sql

# generate brand seasonality feature
sed "s/TRGT_MONTH/$target_month/g;s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f040_brand_seasonality.sql > f040_brand_seasonality_temp.sql
cat f040_brand_seasonality_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f040_brand_seasonality_temp.sql

# generation of brand-related behavioural features
sed "s/TRGT_MONTH/$target_month/g;s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f050_create_brand_features.sql > f050_create_brand_features_temp.sql
cat f050_create_brand_features_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f050_create_brand_features_temp.sql

# generation of behavioural features across all brands
sed "s/TRGT_MONTH/$target_month/g;s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f060_create_overall_features.sql > f060_create_overall_features_temp.sql
cat f060_create_overall_features_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f060_create_overall_features_temp.sql

# compute AOV features
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f070_compute_aov.sql > f070_compute_aov_temp.sql
cat f070_compute_aov_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f070_compute_aov_temp.sql

# impute NULLs with zero
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f080_impute_nulls.sql > f080_impute_nulls_temp.sql
cat f080_impute_nulls_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f080_impute_nulls_temp.sql

# type cast
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" f090_type_cast.sql > f090_type_cast_temp.sql
cat f090_type_cast_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm f090_type_cast_temp.sql

: '
Creation of train / test set
'
# create train / test field
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" t010_train_test_field.sql > t010_train_test_field_temp.sql
cat t010_train_test_field_temp.sql | bq query --destination_table=$dataset.$feature_table --use_legacy_sql=false --allow_large_results --replace=true
rm t010_train_test_field_temp.sql

# create training data
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" t020_train_data.sql > t020_train_data_temp.sql
cat t020_train_data_temp.sql | bq query --destination_table=$dataset.$train --use_legacy_sql=false --allow_large_results
rm t020_train_data_temp.sql

# create testing data
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" t030_test_data.sql > t030_test_data_temp.sql
cat t030_test_data_temp.sql | bq query --destination_table=$dataset.$test --use_legacy_sql=false --allow_large_results
rm t030_test_data_temp.sql

# create test / dev field
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" t040_test_dev_split.sql > t040_test_dev_split_temp.sql
cat t040_test_dev_split_temp.sql | bq query --destination_table=$dataset.$test --use_legacy_sql=true --allow_large_results --replace=true
rm t040_test_dev_split_temp.sql

# create dev data
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" t050_dev_data.sql > t050_dev_data_temp.sql
cat t050_dev_data_temp.sql | bq query --destination_table=$dataset.$dev --use_legacy_sql=false --allow_large_results
rm t050_dev_data_temp.sql

# create final test data
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" t060_test_data_final.sql > t060_test_data_final_temp.sql
cat t060_test_data_final_temp.sql | bq query --destination_table=$dataset.$test --use_legacy_sql=false --allow_large_results --replace=true
rm t060_test_data_final_temp.sql

# create baseline
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" b010_create_baseline.sql > b010_create_baseline_temp.sql
cat b010_create_baseline_temp.sql | bq query --destination_table=$dataset.$baseline --use_legacy_sql=false --allow_large_results
rm b010_create_baseline_temp.sql

# join true labels onto baseline
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" b020_join_baseline.sql > b020_join_baseline_temp.sql
cat b020_join_baseline_temp.sql | bq query --destination_table=$dataset.$baseline --use_legacy_sql=false --allow_large_results --replace=true
rm b020_join_baseline_temp.sql

# compute baseline metrics (accuracy, precision, recall, auc)
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" b030_baseline_metrics.sql > b030_baseline_metrics_temp.sql
cat b030_baseline_metrics_temp.sql | bq query --destination_table=$dataset.$baseline_metrics --use_legacy_sql=false --allow_large_results
rm b030_baseline_metrics_temp.sql

# downsampling majority class in training set to 1:1 ratio
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" s010_downsampling.sql > s010_downsampling_temp.sql
cat s010_downsampling_temp.sql | bq query --destination_table=$dataset.$downsampled --use_legacy_sql=true --allow_large_results
rm s010_downsampling_temp.sql

# joining corresponding features to downsampled customer/brand combinations
sed "s/PROJECT/$project/g;s/DATASET/$dataset_name/g" s020_downsampled_features.sql > s020_downsampled_features_temp.sql
cat s020_downsampled_features_temp.sql | bq query --destination_table=$dataset.$downsampled --use_legacy_sql=false --allow_large_results --replace=true
rm s020_downsampled_features_temp.sql