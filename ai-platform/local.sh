#!/usr/bin/env bash
# run training / evaluation / predictions locally
# evaluate / predict modes: job-dir should be the path to where the model checkpoints are stored

BUCKET=$1
MODE=$2 # train / evaluate / predict
MODEL_DIR=$3
TRAIN_DATA=$4
DEV_DATA=$5
TEST_DATA=$6
SCHEMA=$7
VOCAB=$8
MODEL_TYPE=WD

gcloud ai-platform local train \
	--package-path=package/trainer \
    --module-name=trainer.task \
    --job-dir=${MODEL_DIR} \
    -- \
    --bucket=${BUCKET} \
    --mode=${MODE} \
    --train_data=${TRAIN_DATA} \
    --dev_data=${DEV_DATA} \
    --test_data=${TEST_DATA} \
    --model_dir=${MODEL_DIR} \
    --schema_path=${SCHEMA} \
    --brand_vocab=${VOCAB} \
	--early_stopping \
	--model_type=${MODEL_TYPE} \
    --train_epochs=1 \
    --batch_size=256 \
    --learning_rate=0.0004843447060766648 \
    --optimizer=ProximalAdagrad \
    --hidden_units='128,64,32,16' \
    --dropout=0.488892936706543 \
    --feature_selec='[1,2,3,4,5,6]'