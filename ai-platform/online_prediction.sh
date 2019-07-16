#!/usr/bin/env bash
# online prediction using gcloud tool
# example: bash online_prediction.sh WD v1

MODEL_TYPE=$1
VERSION_NAME=$2

gcloud ai-platform predict \
	--model=${MODEL_TYPE} \
	--version=${VERSION_NAME} \
	--json-instances=test-predictions/online_test.json