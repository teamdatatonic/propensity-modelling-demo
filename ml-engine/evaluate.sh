# submitting the python package to ML Engine for evaluation
# note: the same training parameters should be listed here as command-line args

# DNN or WD
MODEL_TYPE=$1
JOBNAME=$2
PROJECT=$3
BUCKET=$4

JOBID=${JOBNAME}_eval_$(date +"%Y%m%d_%H%M%S")


# note: remember to add the trial number to the job-dir path
# if the model is from a hyper parameter tuning job
gcloud ml-engine jobs submit training ${JOBID}  \
	--scale-tier=BASIC \
	--job-dir=gs://${BUCKET}/models/${MODEL_TYPE}/${JOBNAME} \
	--region=europe-west1 \
	--package-path=package/trainer \
	--module-name=trainer.task \
	--runtime-version=1.12 \
	--python-version=3.5 \
	-- \
	--mode=evaluate \
	--test_data=gs://${BUCKET}/validationData/*.csv \
	--schema_path=gs://${BUCKET}/schema.json \
	--model_type=${MODEL_TYPE} \
	--project=${PROJECT} \
	--bucket=${BUCKET} \
	--mle \
    --train_epochs=5 \
    --batch_size=256 \
    --learning_rate=0.0004843447060766648 \
    --optimizer=ProximalAdagrad \
    --hidden_units='[128, 64, 32, 16]' \
    --dropout=0.488892936706543 \
    --feature_selection='[1,2,3,4,5,6]'
