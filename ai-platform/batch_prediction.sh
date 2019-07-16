#!/usr/bin/env bash
# use deployed batch serving model on AI Platform to run batch predictions

MODEL_NAME=$1
VERSION_NAME=$2
BUCKET=$3

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOBNAME=${MODEL_NAME}_predictions_${TIMESTAMP}
MAX_WORKER_COUNT=1

gcloud ai-platform jobs submit prediction ${JOBNAME} \
	--data-format=text \
	--input-paths=gs://${BUCKET}/testPredictions/batch_test.json \
	--output-path=gs://${BUCKET}/testPredictions/${MODEL_NAME}/${VERSION_NAME}/${JOBNAME} \
	--region=europe-west1 \
	--model=${MODEL_NAME} \
	--version=${VERSION_NAME} \
	--max-worker-count=${MAX_WORKER_COUNT}