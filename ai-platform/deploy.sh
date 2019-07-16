#!/usr/bin/env bash
# deploy model on AI Platform for serving
# model location in Cloud Storage

MODEL_TYPE=WD
VERSION_NAME=$1
BUCKET=$2
JOBNAME=$3
MODELID=$4

MODEL_PATH=gs://${BUCKET}/models/${MODEL_TYPE}/${JOBNAME}/serving/1/${MODELID}

gcloud ai-platform models create ${MODEL_TYPE} \
    --regions=europe-west1

gcloud ai-platform versions create ${VERSION_NAME} \
    --model=${MODEL_TYPE} \
    --origin=${MODEL_PATH} \
    --python-version=3.5 \
    --runtime-version=1.13 \
    --framework=tensorflow