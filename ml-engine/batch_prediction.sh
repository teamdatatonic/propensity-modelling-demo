# use a deployed batch serving model on ML Engine to get predictions

MODEL_TYPE=$1
VERSION_NAME=$2
BUCKET=$3

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME=${MODEL_TYPE}_predictions_$TIMESTAMP
MAX_WORKER_COUNT="1"

gcloud ml-engine jobs submit prediction $JOB_NAME \
	--data-format=text \
	--input-paths=gs://${BUCKET}/testPredictions/batch_test.json \
	--output-path=gs://${BUCKET}/testPredictions/${MODEL_TYPE}/${VERSION_NAME}/${JOB_NAME} \
	--region=europe-west1 \
	--model=${MODEL_TYPE} \
	--version=${VERSION_NAME} \
	--max-worker-count=${MAX_WORKER_COUNT}