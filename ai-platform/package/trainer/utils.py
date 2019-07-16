from google.cloud import storage
import datetime
import yaml
import json
import os


def make_csv_cols(schema):
    """
    Create list containing column defaults
    :param schema: schema dict containing column names, default types
    :return: list of column names, list of column defaults
    """
    columns_default_dict = {
        'STRING': ' ',
        'INTEGER': 0,
        'FLOAT': 0.0,
        'NUMERIC': 0.0,
        'BOOLEAN': False,
        'TIMESTAMP': None,
        'RECORD': None
    }

    csv_columns = [item['name'] for item in schema]
    csv_column_defaults = [columns_default_dict[item['type']] for item in schema]

    return csv_columns, csv_column_defaults


def get_file_from_gcs(project, bucket, path):
    """
    Retrieves file from Google Cloud Storage
    :param project: GCP project name
    :param bucket: Cloud Storage bucket name
    :param path: path to file
    :return: dict for JSON or YAML object
    """
    client = storage.Client(project)
    bucket_obj = client.get_bucket(bucket)

    blob = bucket_obj.get_blob(path.split(bucket + '/')[1])
    file_string = blob.download_as_string()

    ext = path.split('.')[-1]
    if ext == 'json':
        return json.loads(file_string.decode('utf-8'))
    elif ext == 'yaml':
        return yaml.load(file_string.decode('utf-8'))


def get_schema(flags):
    """
    Return schema based on local or AI Platform mode
    :param flags: parsed args
    :return: dict containing schema
    """
    if flags.cloud:
        return get_file_from_gcs(
            project=flags.project,
            bucket=flags.bucket,
            path=flags.schema_path)
    else:
        with open(flags.schema_path, "r") as read_file:
            return json.load(read_file)


def upload_to_gcs(flags, filename, data):
    """
    Upload files to Cloud Storage
    :param flags: parsed args
    :param filename: name of file to upload
    :param data: dict to dump to file
    """
    client = storage.Client(flags.project)
    bucket = client.get_bucket(flags.bucket)
    path = flags.job_dir.replace('gs://{}/'.format(flags.bucket), '')
    path = os.path.join(path, filename)
    json_data = json.dumps(data)
    blob = bucket.blob(path)
    blob.upload_from_string(json_data)


def export_train_results(flags, trial_no, results=None):
    """
    Generate JSON with model settings and evaluation metrics
    :param flags: parsed args
    :param trial_no: model trial number
    :param results: dict containing training details
    """
    timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    filename = 'train_settings_trial{}_{}.json'.format(str(trial_no), timestamp)

    results_dict = {
        # job details
        'job_name': flags.job_dir.split('/')[-1],
        'train_data': flags.train_data,
        'test_data': flags.test_data,
        'dev_data': flags.dev_data,
        'schema_path': flags.schema_path,
        'brand_vocab': flags.brand_vocab,
        'mode': flags.mode,
        'trial': trial_no,
        'ai_platform': flags.cloud,
        'model_type': flags.model_type,
        'learning_rate': str(flags.learning_rate),
        'train_epochs': str(flags.train_epochs),
        'batch_size': str(flags.batch_size),
        'feature_selec': flags.feature_selec
    }

    if flags.model_type in ['DNN', 'WD']:
        results_dict['optimizer'] = flags.optimizer,
        results_dict['hidden_units'] = flags.hidden_units,
        results_dict['dropout'] = str(flags.dropout)

    elif flags.model_type == 'BT':
        results_dict['n_trees'] = flags.n_trees
        results_dict['max_depth'] = flags.max_depth

    # add training duration and training metrics
    if results:
        for k in results.keys():
            results[k] = str(results[k])
        results_dict.update(results)

    if flags.cloud:
        upload_to_gcs(flags, filename, results_dict)
    else:
        with open(os.path.join(flags.job_dir, filename), 'w') as write_file:
            write_file.write(
                json.dumps(results_dict, sort_keys=True, indent=2))


def export_eval_results(flags, trial_no, results):
    """
    Generate JSON with evaluation metrics
    :param flags: parsed args
    :param trial_no: model trial number
    :param results: dict containing eval details
    """
    timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    filename = 'eval_results_trial{}_{}.json'.format(trial_no, timestamp)

    # eval metrics
    results_dict = {
        'job_dir': str(flags.job_dir),
        'trial': str(trial_no),
        'accuracy': str(results['accuracy']),
        'precision': str(results['precision']),
        'recall': str(results['recall']),
        'f1': str(results['f1']),
        'auc': str(results['auc'])
    }

    if flags.cloud:
        upload_to_gcs(flags, filename, results_dict)
    else:
        with open(os.path.join(flags.job_dir, filename), 'w') as write_file:
            write_file.write(
                json.dumps(results_dict, sort_keys=True, indent=2))
