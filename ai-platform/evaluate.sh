#!/usr/bin/env bash
# model evaluation on AI Platform
# --scale-tier=CUSTOM and --config=config.yaml for custom machine types
# use same training parameters for evaluation
# --job-dir is the path to model checkpoints

JOBNAME=$1
PROJECT=$2
BUCKET=$3
MODEL_TYPE=WD

JOBID=${JOBNAME}_eval_$(date +"%Y%m%d_%H%M%S")

gcloud ai-platform jobs submit training ${JOBID}  \
	--scale-tier=BASIC \
	--job-dir=gs://${BUCKET}/models/${MODEL_TYPE}/${JOBNAME}/model/1 \
	--region=europe-west1 \
	--package-path=package/trainer \
	--module-name=trainer.task \
	--runtime-version=1.13 \
	--python-version=3.5 \
	-- \
	--mode=evaluate \
    --project=${PROJECT} \
	--bucket=${BUCKET} \
	--test_data=gs://${BUCKET}/data/testData/*.csv \
	--schema_path=gs://${BUCKET}/schema.json \
	--brand_vocab=gs://${BUCKET}/brand_vocab.csv \
	--cloud \
	--model_type=${MODEL_TYPE} \
    --train_epochs=5 \
    --batch_size=256 \
    --learning_rate=0.0004843447060766648 \
    --optimizer=ProximalAdagrad \
    --hidden_units='128,64,32,16' \
    --dropout=0.488892936706543 \
    --feature_selec='[1,2,3,4,5,6]'