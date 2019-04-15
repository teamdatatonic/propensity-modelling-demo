# online prediction using gcloud tool
# e.g. bash online_prediction.sh WD v2

MODEL_TYPE=$1
VERSION_NAME=$2

gcloud ml-engine predict \
	--model=${MODEL_TYPE} \
	--version=${VERSION_NAME} \
	--json-instances=test-predictions/online_test.json