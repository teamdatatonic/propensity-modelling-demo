## Brand Propensity Model for Major Retailer

Author: Laxmi Prajapat (laxmi.prajapat@datatonic.com)

Python version: 3.5

TensorFlow version: 1.13.1

Data: [Acquired Value Shoppers](https://www.kaggle.com/c/acquire-valued-shoppers-challenge)

### Folders required in package:

---

- `exploration/` - SQL queries for exploratory analysis
- `processing-pipeline/` - SQL queries for BigQuery data preprocessing pipeline
- `ai-platform/` - python package to train estimators locally and/or using AI Platform with hyperparameter tuning


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

1) Download the "transactions" and "history" data from [here](https://www.kaggle.com/c/acquire-valued-shoppers-challenge)
2) Create a Cloud Storage bucket and upload the CSV files
3) Create a dataset in BigQuery
4) Load "transactions" and "history" datasets from Cloud Storage bucket into this BigQuery dataset
5) Execute bash script with GCP project and BigQuery dataset as command-line arguments:

```
/usr/bin/time bash bq-processing.sh <GCP PROJECT> <BQ DATASET>
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
- `t040_test_dev_split.sql`
- `t050_dev_data.sql`
- `t060_test_data_final.sql`
- `x010_cross_join.sql`


`ai-platform/`:

- `requirements.txt` - python dependencies
- `brand_vocab.csv` - brand vocabulary list
- `test-predictions/`
    - `batch_test.json` - sample JSON for running batch predictions (15 lines)
    - `online_test.json` - sample JSON for running online prediction (1 line)
- `package/` 
	- `__init__.py` 
	- `setup.py` - package dependencies 
	- `trainer/` 
		- `__init__.py` 
		- `task.py` - model train / predict / evaluate using TensorFlow Estimator API
        - `utils.py` - helpful functions for `task.py`
- `config.yaml` - configuration file for AI Platform training job
- `hptuning_config.yaml` - configuration file for AI Platform training job with hyperparameter tuning (DNN and WD)
- `local.sh` - bash script to run train / predict / evaluate locally in virtual environment

  ```
  conda create --name propensity_modelling python=3.5
  source activate propensity_modelling
  pip install -r requirements.txt
  ```
  
  Ensure the data and any supporting files are downloaded locally (or on virtual machine) using `gsutil cp` tool.
  
  Local training:
  ```
  bash local.sh <GCS BUCKET> train <MODEL DIR> <TRAIN DATA/*.csv> <DEV DATA/*.csv> <TEST DATA/*.csv> <SCHEMA PATH> <VOCAB PATH>
  ```
  
  Local predicting:
  ```
  bash local.sh <GCS BUCKET> predict <MODEL CHECKPOINTS DIR> <TRAIN DATA/*.csv> <DEV DATA/*.csv> <TEST DATA/*.csv> <SCHEMA PATH> <VOCAB PATH>
  ```
  
  Local evaluating:
  ```
  bash local.sh <GCS BUCKET> evaluate <MODEL CHECKPOINTS DIR> <TRAIN DATA/*.csv> <DEV DATA/*.csv> <TEST DATA/*.csv> <SCHEMA PATH> <VOCAB PATH>
  ```

- `train.sh` - bash script to run training on AI Platform

```
bash train.sh <GCP PROJECT> <GCS BUCKET>
```

- `evaluate.sh` - bash script to run evaluation on AI Platform

```
bash evaluate.sh <JOB NAME> <GCP PROJECT> <GCS BUCKET>
```

- `deploy.sh` - bash script to deploy selected model on AI Platform

```
bash deploy.sh <VERSION> <GCS BUCKET> <JOB NAME> <SERVING ID>
```

- `batch_prediction.sh` - bash script to run batch predictions on AI Platform using deployed model

```
bash batch_prediction.sh <MODEL NAME> <VERSION> <GCS BUCKET>
```

- `online_prediction.sh` - bash script to run online predictions on AI Platform using deployed model

```
bash online_prediction.sh <MODEL NAME> <VERSION>
```


To run **Tensorboard**:

```
tensorboard --logdir=gs://<GCS BUCKET>/models/<MODEL TYPE>/<JOB NAME>/model
```

The **signature** (inputs/outputs) of the saved model can be observed with bash command:

```
saved_model_cli show --dir gs://<GCS BUCKET>/models/<MODEL TYPE>/<JOB NAME>/serving/<SERVING ID> --tag serve --signature_def predict
```
