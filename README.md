## Brand Propensity Model for Major Retailer

Python version: 3.5

### Folders required in package:

---

- `exploration/` - SQL queries for exploratory analysis and Tableau workbook
- `processing-pipeline/` - SQL queries for data preprocessing pipeline in BigQuery
- `ml-engine/` - python package to train estimators locally and/or using ML Engine with hyperparameter tuning


### Files and execution:

---

`exploration/`:

- `visualisation-queries/`:
	- `custom_query_1.sql`
	- `custom_query_2.sql`
	- `custom_query_3.sql`
	- `custom_query_4.sql`

`processing-pipeline/`:

- `README.md`
- `bq-processing.sh`

1) Create a dataset in BigQuery
2) Load "transactions" and "history" datasets from GCS into this BigQuery dataset
3) Execute bash script with GCP project and dataset as command-line arguments:

```
/usr/bin/time bash bq-processing.sh {PROJECT} {BQ DATASET}
```

- `supporting-queries.sql`
- `a010_impute_missing_values.sql`
- `a020_remove_zero_quantity_rows.sql`
- `a030_create_returned_flag.sql`
- `a040_absolute_values.sql`
- `a050_create_transaction_id.sql`
- `a060_create_product_id.sql`
- `a070_create_product_price.sql`
- `b010_create_baseline.sql`
- `b020_join_baseline.sql`
- `b030_baseline_metrics.sql`
- `f010_top_brands.sql`
- `f020_label_creation.sql`
- `f030_promo_sensitive_feature.sql`
- `f040_brand_seasonality.sql`
- `f050_create_brand_features.sql`
- `f060_create_overall_features.sql`
- `f070_compute_aov.sql`
- `f080_impute_nulls.sql`
- `f090_type_cast.sql`
- `s010_downsampling.sql`
- `s020_downsampled_features.sql`
- `t010_train_test_field.sql`
- `t020_train_data.sql`
- `t030_test_data.sql`
- `t040_test_validation_split.sql`
- `t050_validation_data.sql`
- `t060_test_data_final.sql`
- `x010_cross_join.sql`


`ml-engine/`:

- `requirements.txt` - downloading dependencies locally
- `package/` 
	- `__init__.py` 
	- `setup.py` - package dependencies 
	- `trainer/` 
		- `__init__.py` 
		- `task.py` - model train / predict / evaluate using TensorFlow Estimator API
- `config.yaml` - configuration file for ML Engine training job
- `hptuning_config.yaml` - configuration file for ML Engine training job with hyperparameter tuning for DNN and WD
- `local_train.sh` - bash script to train locally in virtual environment (ensure paths are the **full** paths)

  ```
  conda create --name propensity_modelling python=3.5
  source activate propensity_modelling
  pip install -r requirements.txt
  bash local_train.sh {MODEL TYPE} {MODEL DIR} {TRAIN DATA/*.csv} {TEST DATA/*.csv} {SCHEMA PATH}
  ```

- `train.sh` - bash script to train on ML Engine

```
bash train.sh {MODEL TYPE} {CONFIG} {PROJECT NAME} {BUCKET NAME}
```

- `evaluate.sh` - bash script to run just the evaluation on ML Engine

```
bash evaluate.sh {MODEL TYPE} {JOB NAME} {PROJECT NAME} {BUCKET NAME}
```

- `deploy.sh` - bash script to deploy model on ML Engine

```
bash deploy.sh {MODEL TYPE} {VERSION} {BUCKET NAME} {JOBNAME} {SERVING ID}
```

- `batch_prediction.sh` - bash script to run batch predictions using deployed model

```
bash batch_prediction.sh {MODEL NAME} {VERSION} {BUCKET NAME}
```

- `online_prediction.sh` - bash script to run online predictions using deployed model

```
bash online_prediction.sh {MODEL NAME} {VERSION}
```


To run **Tensorboard**:

```
tensorboard --logdir=gs://${BUCKET}/models/${MODEL_TYPE}/${JOBNAME}/model
```

The **signature** (inputs/outputs) of the saved model can be observed with bash command:

```
saved_model_cli show --dir gs://{BUCKET NAME}/models/{MODEL TYPE}/{JOBNAME}/serving/{ID} --tag serve --signature_def predict
```
