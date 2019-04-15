# command line args for model directory and train / test data

# BT, DNN or WD
MODEL_TYPE=$1
MODEL_DIR=$2
TRAIN_DATA=$3
TEST_DATA=$4
SCHEMA=$5

gcloud ml-engine local train \
	--package-path=package/trainer \
    --module-name=trainer.task \
    --job-dir=${MODEL_DIR} \
    -- \
    --train_data=${TRAIN_DATA} \
    --test_data=${TEST_DATA} \
    --model_dir=${MODEL_DIR} \
    --model_type=${MODEL_TYPE} \
    --schema_path=${SCHEMA}
