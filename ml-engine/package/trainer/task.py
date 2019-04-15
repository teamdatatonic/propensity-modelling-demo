import argparse
import ast
import json
import os
import sys

import tensorflow as tf
import dask.dataframe as dd
import pandas as pd
import numpy as np

from google.cloud import storage
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

RANDOM_SEED = 42
LABEL = 'label'
CHECKPOINT_STEPS = 100
SUMMARY_STEPS = 1000
COLUMNS_DEFAULT_DICT = {'INTEGER': [0], 'STRING': [' ']}
COLUMNS_TYPE_DICT = {'INTEGER': tf.int64, 'STRING': tf.string}

tf.logging.set_verbosity(tf.logging.FATAL)
np.random.seed(RANDOM_SEED)


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--project', type=str, default='example-project', help='GCP project')
    parser.add_argument(
        '--bucket', type=str, default='example-bucket', help='GCS bucket')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Mode: {train, predict, evaluate}')
    parser.add_argument(
        '--mle',
        dest='mle',
        action='store_true',
        help='Running on ML Engine or locally.')
    parser.add_argument(
        '--schema_path',
        type=str,
        default='schema.json',
        help="Path to data schema.")
    parser.add_argument(
        '--train_data',
        type=str,
        default='data/trainData/*.csv',
        help='Path to the training data.')
    parser.add_argument(
        '--test_data',
        type=str,
        default='data/testData/*.csv',
        help='Path to the testing data.')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='model',
        help='Base directory for the trained model.')
    parser.add_argument(
        '--model_type',
        type=str,
        default='BT',
        help='Valid model types: {DNN, WD, BT}.')
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=5,
        help='Number of training epochs.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Number of examples per batch.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Model learning rate.')
    parser.add_argument(
        '--hidden_units',
        type=str,
        default='[128, 64]',
        help='n hidden units.')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help='Model optimizer: {ProximalAdagrad, Adagrad, Adam}.')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Probability to drop out a given coordinate.')
    parser.add_argument(
        '--n_trees',
        type=int,
        default=100,
        help='Number of trees for BT model.')
    parser.add_argument(
        '--max_depth',
        type=int,
        default=6,
        help='Maximum depth for BT model.')
    parser.add_argument(
        '--feature_selection',
        type=str,
        default=None,
        help='List of options (1 to 7) to eliminate certain features.')
    parser.add_argument(
        '--early_stopping',
        dest='early_stopping',
        action='store_true',
        help='Early stopping during training.')
    return parser.parse_known_args()


def get_schema_from_gcs():
    """Reads schema from file in Google Cloud Storage."""
    client = storage.Client(FLAGS.project)
    bucket = client.get_bucket(FLAGS.bucket)
    blob = bucket.get_blob(FLAGS.schema_path.split(FLAGS.bucket + '/')[1])
    schema_string = blob.download_as_string()
    return json.loads(schema_string.decode('utf-8'))


def get_schema():
    """Return schema based on local or ML Engine mode."""
    if FLAGS.mle:
        return get_schema_from_gcs()
    else:
        with open(FLAGS.schema_path, "r") as read_file:
            return json.load(read_file)


FLAGS, unparsed = get_args()
SCHEMA = get_schema()
CSV_COLUMNS = [item['name'] for item in SCHEMA]
CSV_COLUMN_DEFAULTS = [COLUMNS_DEFAULT_DICT[item['type']] for item in SCHEMA]


def upload_to_gcs(filename, data):
    client = storage.Client(FLAGS.project)
    bucket = client.get_bucket(FLAGS.bucket)
    path = FLAGS.job_dir.replace('gs://{}/'.format(FLAGS.bucket), '')
    path = os.path.join(path, filename)
    json_data = json.dumps(data)
    blob = bucket.blob(path)
    blob.upload_from_string(json_data)


def download_from_gcs(filename):
    client = storage.Client(FLAGS.project)
    bucket = client.get_bucket(FLAGS.bucket)
    path = FLAGS.job_dir.replace('gs://{}/'.format(FLAGS.bucket), '')
    path = os.path.join(path, filename)
    blob = bucket.get_blob(path)
    results_string = blob.download_as_string()
    return json.loads(results_string.decode('utf-8'))


def export_results(trial_no, results=None):
    """Training locally - generate JSON with model information and metrics"""
    results_file = "eval_results.json"
    results_dict = {
        'job_name': FLAGS.job_dir.split('/')[-1],
        'trial': trial_no,
        'model_type': FLAGS.model_type,
        'feature_selection': FLAGS.feature_selection,
        'train_epochs': FLAGS.train_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': str(FLAGS.learning_rate)
    }

    if FLAGS.model_type in ['DNN', 'WD']:
        results_dict['dropout'] = FLAGS.dropout
        results_dict['activation_fn'] = 'relu'
        results_dict['optimizer'] = FLAGS.optimizer
        results_dict['hidden_units'] = FLAGS.hidden_units
    elif FLAGS.model_type == 'BT':
        results_dict['n_trees'] = FLAGS.n_trees
        results_dict['max_depth'] = FLAGS.max_depth

    if results:
        if FLAGS.mle:
            results_dict = download_from_gcs(results_file)
        else:
            with open(results_file, 'r') as read_file:
                results_dict = json.load(read_file)

    try:
        results_dict['accuracy'] = str(results['accuracy'])
        results_dict['precision'] = str(results['precision'])
        results_dict['recall'] = str(results['recall'])
        results_dict['f1'] = str(results['f1'])
        results_dict['auc'] = str(results['auc'])
    except:
        pass

    if FLAGS.mle:
        upload_to_gcs(results_file, results_dict)
    else:
        with open(results_file, 'w') as write_file:
            write_file.write(
                json.dumps(results_dict, sort_keys=True, indent=2))


def input_fn(data_dir,
             num_epochs=None,
             shuffle=True,
             batch_size=128,
             skip_header_lines=1):
    """
    Generation of features and labels for Estimator
    :param data_dir: path to directory containing data
    :param num_epochs: number of times to repeat
    :param shuffle: True / False
    :param batch_size: stacks n consecutive elements of dataset into single element
    :param skip_header_lines: lines to skip if header present
    :return:
    """

    def parse_csv(line):
        """
        records: A Tensor of type string. Each string is a record/row in the csv.
        record_defaults: A list of Tensor objects with specific types.
                         (float32, float64, int32, int64, string)
        """
        columns = tf.decode_csv(
            records=line, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL)

        return features, labels

    data_list = tf.gfile.Glob(data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(data_list)

    # to process many input files concurrently
    dataset = dataset.interleave(
        lambda filename: (tf.data.TextLineDataset(filename).skip(skip_header_lines)),
        cycle_length=2,
        block_length=4)

    if shuffle:
        dataset = dataset.shuffle(seed=RANDOM_SEED, buffer_size=1000000)

    dataset = dataset.batch(batch_size) \
        .map(parse_csv, num_parallel_calls=8) \
        .repeat(num_epochs) \
        .prefetch(1)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def serving_input_receiver_fn():
    """
    Build the serving inputs during online prediction
    :return: ServingInputReceiver
    """
    json_features = {}
    INPUT = [field for field in SCHEMA if field['name'] not in [LABEL]]
    for field in INPUT:
        dtype = COLUMNS_TYPE_DICT[field['type']]
        json_features[field['name']] = tf.placeholder(
            shape=[None], dtype=dtype)
    return tf.estimator.export.ServingInputReceiver(json_features,
                                                    json_features)


def build_feature_columns():
    """
    Build feature columns as input to the model
    :return: feature column tensors
    """

    # most of the columns are numeric columns
    exclude = ['customer_id', 'brand', 'promo_sensitive', 'label']

    if FLAGS.feature_selection:
        feature_dict = {
            1: 'returned',
            2: 'chains',
            3: 'max_sale_quantity',
            4: 'overall',
            5: '12m',
            6: '6m',
            7: '3m'
        }
        terms = [
            feature_dict[key]
            for key in ast.literal_eval(FLAGS.feature_selection)
        ]
        feature_list = [
            col for col in CSV_COLUMNS if any(word in col for word in terms)
        ]
        exclude += feature_list

    numeric_column_names = [col for col in CSV_COLUMNS if col not in exclude]
    numeric_columns = [
        tf.feature_column.numeric_column(col) for col in numeric_column_names
    ]

    # promo sensitive
    promo_sensitive = tf.feature_column.categorical_column_with_identity(
        key='promo_sensitive', num_buckets=2)

    # customer id and brand hashing
    customer_id = tf.feature_column.categorical_column_with_hash_bucket(
        key='customer_id', hash_bucket_size=100000)
    brand = tf.feature_column.categorical_column_with_hash_bucket(
        key='brand', hash_bucket_size=1000)

    # bucketizing columns
    seasonality_names = [
        col for col in numeric_columns if 'seasonality' in col.key
    ]
    brand_seasonality = [
        tf.feature_column.bucketized_column(col, boundaries=[3, 6, 9, 12])
        for col in seasonality_names
    ]

    aov_column_names = [col for col in numeric_columns if 'aov' in col.key]
    aov_columns = [
        tf.feature_column.bucketized_column(
            col, boundaries=[0, 3, 6, 9, 12, 15, 30, 50, 100])
        for col in aov_column_names
    ]

    days_1m_names = [
        col for col in numeric_columns if 'days_shopped_1m' in col.key
    ]
    days_3m_names = [
        col for col in numeric_columns if 'days_shopped_3m' in col.key
    ]
    days_6m_names = [
        col for col in numeric_columns if 'days_shopped_6m' in col.key
    ]
    days_12m_names = [
        col for col in numeric_columns if 'days_shopped_12m' in col.key
    ]
    days_1m = [
        tf.feature_column.bucketized_column(
            col, boundaries=[0, 2, 5, 10, 20, 30]) for col in days_1m_names
    ]
    days_3m = [
        tf.feature_column.bucketized_column(col, boundaries=[0, 30, 60, 90])
        for col in days_3m_names
    ]
    days_6m = [
        tf.feature_column.bucketized_column(col, boundaries=[0, 60, 120, 180])
        for col in days_6m_names
    ]
    days_12m = [
        tf.feature_column.bucketized_column(
            col, boundaries=[0, 90, 180, 270, 360]) for col in days_12m_names
    ]

    quantity_column_names = [
        col for col in numeric_columns
        if any(word in col.key for word in ['quantity', 'distinct_brands'])
    ]
    quantity_columns = [
        tf.feature_column.bucketized_column(
            col, boundaries=[0, 5, 10, 20, 50, 100, 500, 1000])
        for col in quantity_column_names
    ]

    product_column_names = [
        col for col in numeric_columns if 'products' in col.key
    ]
    product_columns = [
        tf.feature_column.bucketized_column(
            col, boundaries=[0, 5, 10, 20, 50, 100])
        for col in product_column_names
    ]

    cat_column_names = [
        col for col in numeric_columns if 'category' in col.key
    ]
    cat_columns = [
        tf.feature_column.bucketized_column(
            col, boundaries=[0, 2, 5, 10, 20, 30, 50, 100])
        for col in cat_column_names
    ]

    customer_embeddings = tf.feature_column.embedding_column(
        customer_id, dimension=18)
    brand_embeddings = tf.feature_column.embedding_column(brand, dimension=6)

    deep_columns = numeric_columns + [customer_embeddings, brand_embeddings]
    bucketized_columns = (
        brand_seasonality + aov_columns + days_1m + days_3m + days_6m +
        days_12m + quantity_columns + product_columns + cat_columns)
    wide_columns = [promo_sensitive] + bucketized_columns

    return wide_columns, deep_columns, bucketized_columns


def metrics(labels, predictions):
    """Define evaluation metrics."""
    return {
        'accuracy': tf.metrics.accuracy(labels, predictions['class_ids']),
        'precision': tf.metrics.precision(labels, predictions['class_ids']),
        'recall': tf.metrics.recall(labels, predictions['class_ids']),
        'f1': tf.contrib.metrics.f1_score(labels, predictions['class_ids']),
        'auc': tf.metrics.auc(labels, predictions['logistic'])
    }


def initialize_optimizer():
    optimizers = {
        'Adagrad':
        tf.train.AdagradOptimizer(FLAGS.learning_rate),
        'ProximalAdagrad':
        tf.train.ProximalAdagradOptimizer(
            FLAGS.learning_rate,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.001),
        'Adam':
        tf.train.AdamOptimizer(FLAGS.learning_rate)
    }

    if optimizers.get(FLAGS.optimizer):
        return optimizers[FLAGS.optimizer]

    raise Exception('Optimizer {} not recognised'.format(FLAGS.optimizer))


def initialize_estimator(bucketized_features, deep_features, wide_features,
                         model_checkpoints):
    optimizer = initialize_optimizer()
    run_config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_checkpoints_steps=CHECKPOINT_STEPS,
        save_summary_steps=SUMMARY_STEPS)

    if FLAGS.model_type == 'DNN':
        return tf.estimator.DNNClassifier(
            n_classes=2,
            feature_columns=deep_features,
            activation_fn=tf.nn.relu,
            optimizer=optimizer,
            hidden_units=ast.literal_eval(FLAGS.hidden_units),
            dropout=FLAGS.dropout,
            batch_norm=True,
            model_dir=model_checkpoints,
            config=run_config)
    elif FLAGS.model_type == 'WD':
        return tf.estimator.DNNLinearCombinedClassifier(
            n_classes=2,
            linear_feature_columns=wide_features,
            linear_optimizer='Ftrl',
            dnn_feature_columns=deep_features,
            dnn_optimizer=optimizer,
            dnn_hidden_units=ast.literal_eval(FLAGS.hidden_units),
            dnn_activation_fn=tf.nn.relu,
            dnn_dropout=FLAGS.dropout,
            batch_norm=True,
            model_dir=model_checkpoints,
            config=run_config)
    elif FLAGS.model_type == 'BT':
        n_batches = 500
        return tf.estimator.BoostedTreesClassifier(
            n_classes=2,
            n_batches_per_layer=n_batches,
            feature_columns=bucketized_features,
            learning_rate=FLAGS.learning_rate,
            n_trees=FLAGS.n_trees,
            max_depth=FLAGS.max_depth,
            model_dir=model_checkpoints,
            config=run_config)

    raise Exception(
        'Model type {} not recognised - choose from DNN, WD or BT.'.format(
            FLAGS.model_type))


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    model_checkpoints = FLAGS.job_dir + '/model'
    model_serving = FLAGS.job_dir + '/serving'
    trial = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get(
        'trial', '')

    wide_features, deep_features, bucketized_features = build_feature_columns()

    estimator = initialize_estimator(bucketized_features, deep_features,
                                     wide_features, model_checkpoints)

    estimator = tf.contrib.estimator.add_metrics(estimator, metrics)
    estimator = tf.contrib.estimator.forward_features(
        estimator, keys=['customer_id', 'brand'])

    if FLAGS.mode == "train":
        export_results(trial)

        # stop if metric does not decrease within given max steps
        if FLAGS.early_stopping:
            early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
                estimator, metric_name='loss', max_steps_without_decrease=5000)
            hooks = [early_stopping]
        else:
            hooks = None

        tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(
                input_fn=lambda: input_fn(
                    FLAGS.train_data,
                    FLAGS.train_epochs,
                    shuffle=True,
                    batch_size=FLAGS.batch_size),
                hooks=hooks
            ),
            tf.estimator.EvalSpec(
                input_fn=lambda: input_fn(
                    FLAGS.test_data,
                    num_epochs=1,
                    shuffle=False,
                    batch_size=FLAGS.batch_size
                ),
                steps=1000,
                throttle_secs=60
            )
        )

        estimator.export_savedmodel(
            export_dir_base=model_serving,
            serving_input_receiver_fn=serving_input_receiver_fn)
        print('Model has been saved.')

    elif FLAGS.mode == 'predict':

        predictions = estimator.predict(
            input_fn=lambda: input_fn(
                FLAGS.test_data,
                num_epochs=1,
                shuffle=False,
                batch_size=FLAGS.batch_size
            ))

        eval_df = dd.read_csv(
            FLAGS.test_data,
            header=0,
            usecols=['customer_id', 'brand', 'label']).compute()
        eval_df['customer_id'] = eval_df['customer_id'].astype(str)
        eval_df['brand'] = eval_df['brand'].astype(str)

        predictions_df = pd.DataFrame()

        for p in predictions:
            predictions_df = predictions_df.append(
                {
                    'customer_id': p['customer_id'],
                    'brand': p['brand'],
                    'y_pred': p['class_ids'][0],
                    'y_pred_prob': p['logistic'][0]
                },
                ignore_index=True)
        predictions_df['y_pred'] = predictions_df['y_pred'].astype(int)

        eval_df = eval_df.merge(
            predictions_df, how='left', on=['customer_id', 'brand'])

        results = {}
        results['accuracy'] = accuracy_score(eval_df['label'],
                                             eval_df['y_pred'])
        results['f1'] = f1_score(eval_df['label'], eval_df['y_pred'])
        results['precision'] = precision_score(eval_df['label'],
                                               eval_df['y_pred'])
        results['recall'] = recall_score(eval_df['label'], eval_df['y_pred'])
        results['auc'] = roc_auc_score(eval_df['label'],
                                       eval_df['y_pred_prob'])

        export_results(trial, results)

    elif FLAGS.mode == 'evaluate':
        results = estimator.evaluate(
            input_fn=lambda: input_fn(
                FLAGS.test_data,
                num_epochs=1,
                shuffle=False,
                batch_size=FLAGS.batch_size
            ),
            steps=1000)

        export_results(trial, results)

    else:
        print('Unrecognised mode {}'.format(FLAGS.mode))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
