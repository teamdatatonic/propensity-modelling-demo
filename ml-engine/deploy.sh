# find model in GCS based on the job name / model id and deploy it on ML Engine for serving

MODEL_TYPE=$1
VERSION_NAME=$2
BUCKET=$3
JOBNAME=$4
MODELID=$5

MODEL_PATH=gs://${BUCKET}/models/${MODEL_TYPE}/${JOBNAME}/serving/${MODELID}

gcloud ml-engine models create ${MODEL_TYPE} \
    --regions=europe-west1

gcloud ml-engine versions create ${VERSION_NAME}  \
    --model ${MODEL_TYPE} \
    --origin ${MODEL_PATH} \
    --python-version=3.5 \
    --runtime-version 1.12