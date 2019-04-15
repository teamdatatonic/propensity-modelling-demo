# submitting the python package to ML Engine for training
# change scale-tier to BASIC if HPT job

# DNN or WD
MODEL_TYPE=$1
# config.yaml or hptuning_config.yaml
CONFIG=$2
PROJECT=$3
BUCKET=$4
# trainData or trainDataSampled
DATA=$5

JOBNAME=${MODEL_TYPE}_training_$(date +"%Y%m%d_%H%M%S")

gcloud ml-engine jobs submit training ${JOBNAME}  \
	--scale-tier=CUSTOM \
	--job-dir=gs://${BUCKET}/models/${MODEL_TYPE}/${JOBNAME} \
	--region=europe-west1 \
	--package-path=package/trainer \
	--module-name=trainer.task \
	--runtime-version=1.12 \
	--python-version=3.5 \
	--config=${CONFIG} \
	-- \
	--train_data=gs://${BUCKET}/${DATA}/*.csv \
	--test_data=gs://${BUCKET}/testData/*.csv \
	--schema_path=gs://${BUCKET}/schema.json \
	--model_type=${MODEL_TYPE} \
	--project=${PROJECT} \
	--bucket=${BUCKET} \
	--mle \
	--early_stopping \
    --train_epochs=5 \
    --batch_size=256 \
    --learning_rate=0.0004843447060766648 \
    --optimizer=ProximalAdagrad \
    --hidden_units='[128, 64, 32, 16]' \
    --dropout=0.488892936706543 \
    --feature_selection='[1,2,3,4,5,6]'