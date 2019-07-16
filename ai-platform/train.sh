#!/usr/bin/env bash
# model training on AI Platform
# --scale-tier=CUSTOM and --config=config.yaml for custom machine types
# --config=hptuning_config.yaml for hyperparameter tuning job

PROJECT=$1
BUCKET=$2
MODEL_TYPE=WD # DNN, WD or BT
JOBNAME=${MODEL_TYPE}_training_$(date +"%Y%m%d_%H%M%S")

gcloud ai-platform jobs submit training ${JOBNAME}  \
	--scale-tier=BASIC \
	--job-dir=gs://${BUCKET}/models/WD/${JOBNAME} \
	--region=europe-west1 \
	--package-path=package/trainer \
	--module-name=trainer.task \
	--runtime-version=1.13 \
	--python-version=3.5 \
	-- \
	--project=${PROJECT} \
	--bucket=${BUCKET} \
	--train_data=gs://${BUCKET}/data/trainData/*.csv \
	--dev_data=gs://${BUCKET}/data/devData/*.csv \
	--test_data=gs://${BUCKET}/data/testData/*.csv \
	--schema_path=gs://${BUCKET}/schema.json \
	--brand_vocab=gs://${BUCKET}/brand_vocab.csv \
	--cloud \
	--early_stopping \
	--model_type=${MODEL_TYPE} \
    --train_epochs=5 \
    --batch_size=256 \
    --learning_rate=0.0004843447060766648 \
    --optimizer=ProximalAdagrad \
    --hidden_units='128,64,32,16' \
    --dropout=0.488892936706543 \
    --feature_selec='[1,2,3,4,5,6]'