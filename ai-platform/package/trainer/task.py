"""
Usage:
    task.py [options]

Options:
    --mode=<mode>           train/evaluate/predict [default: train]
    --project=<name>        name of the GCP project
    --bucket=<name>         GCS bucket

    --schema_path=<path>    path to schema json in GCS [default: None]
    --brand_vocab=<path>    path to brand vocabulary file [default: 'brand_vocab.csv']
    --train_data=<path>     path to train data
    --dev_data=<path>       path to dev (validation) data
    --test_data=<path>      path to test data (for evaluation, or prediction)

    --model_type=<type>     model to train [default: WD]
    --feature_selec=<str>   features to exclude (from 1 to 7) [default: None]
    --train_epochs=<n>      number of epochs to train for [default: 1]
    --batch_size=<n>        batch size while training [default: 1024]
    --learning_rate=<lr>    learning rate while training [default: 0.01]
    --optimizer=<opt>       GD optimizer [default: Adam]
    --dropout=<frac>        dropout fraction [default: 0.2]
    --hidden_units=<str>    hidden units of DNN [default: '128,64']

    --n_trees=<n>           number of trees [default: 100]
    --max_depth=<n>         maximum depth of BT model [default: 6]

    --early_stopping        stop training early if no decrease in loss over specified number of steps
    --cloud                 running jobs on AI Platform
    --job-dir=<dir>         working directory for models and checkpoints [default: 'model']

"""
import argparse
import ast
import datetime
import json
import os
import subprocess
import sys
import time

import tensorflow as tf

# local or AI Platform training
try:
    from utils import (
        export_train_results, export_eval_results, get_schema, make_csv_cols)
except:
    from trainer.utils import (
        export_train_results, export_eval_results, get_schema, make_csv_cols)

RANDOM_SEED = 42


def get_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='{train, predict, evaluate}')
    parser.add_argument('--project', type=str, default='example-project')
    parser.add_argument('--bucket', type=str, default='example-bucket')
    parser.add_argument('--schema_path', type=str, default='schema.json')
    parser.add_argument('--brand_vocab', type=str, default='brand_vocab.csv')

    parser.add_argument('--train_data', type=str, default='trainData/*.csv')
    parser.add_argument('--dev_data', type=str, default='devData/*.csv')
    parser.add_argument('--test_data', type=str, default='testData/*.csv')

    parser.add_argument('--model_type', type=str, default='WD', help='{DNN, WD, BT}')
    parser.add_argument('--feature_selec', type=str, default=None)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='Adam', help='{ProximalAdagrad, Adagrad, Adam}')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hidden_units', type=str, default='128,64')

    parser.add_argument('--n_trees', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)

    parser.add_argument('--cloud', dest='cloud', action='store_true')
    parser.add_argument('--early_stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--job-dir', type=str, default='model')

    return parser.parse_known_args()


COLUMNS_TYPE_DICT = {
    'STRING': tf.string,
    'INTEGER': tf.int64,
    'FLOAT': tf.float32,
    'NUMERIC': tf.float32,
    'BOOLEAN': tf.bool,
    'TIMESTAMP': None,
    'RECORD': None
}

FLAGS, unparsed = get_args()
SCHEMA = get_schema(FLAGS)
LABEL = 'label'
CSV_COLUMNS, CSV_COLUMN_DEFAULTS = make_csv_cols(SCHEMA)


def input_fn(path_dir, epochs, batch_size=1024, shuffle=True, skip_header_lines=1):
    """
    Generation of features and labels for Estimator
    :param path_dir: path to directory containing data
    :param epochs: number of times to repeat
    :param batch_size: stacks n consecutive elements of dataset into single element
    :param skip_header_lines: lines to skip if header present
    :return: features, labels
    """

    def parse_csv(records):
        """
        :param records: A Tensor of type string - each string is a record/row in the CSV
        :return: features, labels
        """
        columns = tf.decode_csv(
            records=records, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))

        # forwarding features for customer ID and brand
        features['customer_identity'] = tf.identity(features['customer_id'])
        features['brand_identity'] = tf.identity(features['brand'])

        try:
            labels = features.pop(LABEL)
            return features, labels
        except KeyError:
            return features, []

    file_list = tf.gfile.Glob(path_dir)
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    # shuffle file list
    if shuffle:
        dataset = dataset.shuffle(50, seed=RANDOM_SEED)

    # read lines of files as row strings, then shuffle and batch
    f = lambda filepath: tf.data.TextLineDataset(filepath).skip(skip_header_lines)
    dataset = dataset.interleave(f, cycle_length=8, block_length=8)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100000, seed=RANDOM_SEED)

    dataset = dataset.batch(batch_size) \
        .map(parse_csv, num_parallel_calls=8) \
        .repeat(epochs)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def serving_input_receiver_fn():
    """
    Build the serving inputs during online prediction
    :return: ServingInputReceiver
    """
    raw_features = dict()
    INPUT = [field for field in SCHEMA if field['name'] not in [LABEL]]

    for field in INPUT:
        dtype = COLUMNS_TYPE_DICT[field['type']]
        raw_features[field['name']] = tf.placeholder(
            shape=[None], dtype=dtype)

    features = raw_features.copy()
    features['customer_identity'] = tf.identity(features['customer_id'])
    features['brand_identity'] = tf.identity(features['brand'])

    return tf.estimator.export.ServingInputReceiver(features,
                                                    raw_features)


def metrics(labels, predictions):
    """
    Define evaluation metrics
    :return: dict of metrics
    """
    return {
        'accuracy': tf.metrics.accuracy(labels, predictions['class_ids']),
        'precision': tf.metrics.precision(labels, predictions['class_ids']),
        'recall': tf.metrics.recall(labels, predictions['class_ids']),
        'f1': tf.contrib.metrics.f1_score(labels, predictions['class_ids']),
        'auc': tf.metrics.auc(labels, predictions['logistic'])
    }


def build_feature_columns():
    """
    Build feature columns as input to the model
    :return: feature column tensors
    """

    # most of the columns are numeric columns
    exclude = ['customer_id', 'brand', 'promo_sensitive', 'label']

    if FLAGS.feature_selec:
        feature_dict = {
            1: 'returned',
            2: 'chains',
            3: 'max_sale_quantity',
            4: 'overall',
            5: '12m',
            6: '6m',
            7: '3m'
        }
        terms = [feature_dict[key] for key in ast.literal_eval(FLAGS.feature_selec)]
        feature_list = [col for col in CSV_COLUMNS if any(word in col for word in terms)]
        exclude += feature_list

    numeric_column_names = [col for col in CSV_COLUMNS if col not in exclude]
    numeric_columns = [tf.feature_column.numeric_column(col) for col in numeric_column_names]

    # promo sensitive
    promo_sensitive = tf.feature_column.categorical_column_with_identity(
        key='promo_sensitive', num_buckets=2)

    # customer id and brand hash buckets
    customer_id = tf.feature_column.categorical_column_with_hash_bucket(
        key='customer_id', hash_bucket_size=100000)
    brand = tf.feature_column.categorical_column_with_vocabulary_file(
        key='brand', vocabulary_file=FLAGS.brand_vocab)

    # bucketizing columns
    seasonality_names = [col for col in numeric_columns if 'seasonality' in col.key]
    brand_seasonality = [tf.feature_column.bucketized_column(
        col, boundaries=[3, 6, 9, 12]) for col in seasonality_names]

    aov_column_names = [col for col in numeric_columns if 'aov' in col.key]
    aov_columns = [tf.feature_column.bucketized_column(
            col, boundaries=[0, 3, 6, 9, 12, 15, 30, 50, 100]) for col in aov_column_names]

    days_1m_names = [col for col in numeric_columns if 'days_shopped_1m' in col.key]
    days_3m_names = [col for col in numeric_columns if 'days_shopped_3m' in col.key]
    days_6m_names = [col for col in numeric_columns if 'days_shopped_6m' in col.key]
    days_12m_names = [col for col in numeric_columns if 'days_shopped_12m' in col.key]

    days_1m = [tf.feature_column.bucketized_column(
            col, boundaries=[0, 2, 5, 10, 20, 30]) for col in days_1m_names]
    days_3m = [tf.feature_column.bucketized_column(
        col, boundaries=[0, 30, 60, 90]) for col in days_3m_names]
    days_6m = [tf.feature_column.bucketized_column(
        col, boundaries=[0, 60, 120, 180]) for col in days_6m_names]
    days_12m = [tf.feature_column.bucketized_column(
            col, boundaries=[0, 90, 180, 270, 360]) for col in days_12m_names]

    quantity_column_names = [col for col in numeric_columns if any(
        word in col.key for word in ['quantity', 'distinct_brands'])]
    quantity_columns = [tf.feature_column.bucketized_column(
            col, boundaries=[0, 5, 10, 20, 50, 100, 500, 1000]) for col in quantity_column_names]

    product_column_names = [col for col in numeric_columns if 'products' in col.key]
    product_columns = [tf.feature_column.bucketized_column(
        col, boundaries=[0, 5, 10, 20, 50, 100]) for col in product_column_names]

    cat_column_names = [col for col in numeric_columns if 'category' in col.key]
    cat_columns = [tf.feature_column.bucketized_column(
            col, boundaries=[0, 2, 5, 10, 20, 30, 50, 100]) for col in cat_column_names]

    customer_embeddings = tf.feature_column.embedding_column(customer_id, dimension=18)
    brand_embeddings = tf.feature_column.embedding_column(brand, dimension=6)

    deep_columns = numeric_columns + [customer_embeddings, brand_embeddings]
    tree_columns = (
        brand_seasonality + aov_columns + days_1m + days_3m + days_6m +
        days_12m + quantity_columns + product_columns + cat_columns)
    wide_columns = [promo_sensitive] + tree_columns

    return wide_columns, deep_columns, tree_columns


def initialize_optimizer():
    """
        Define GD optimizer
        :return: optimizer
    """
    optimizers = {
        'Adagrad': tf.train.AdagradOptimizer(FLAGS.learning_rate),
        'ProximalAdagrad': tf.train.ProximalAdagradOptimizer(FLAGS.learning_rate),
        'Adam': tf.train.AdamOptimizer(FLAGS.learning_rate)
    }

    if optimizers.get(FLAGS.optimizer):
        return optimizers[FLAGS.optimizer]

    raise Exception('Optimizer {} not recognised'.format(FLAGS.optimizer))


def initialize_estimator(model_checkpoints):
    """
        Define estimator
        :return: estimator
    """
    optimizer = initialize_optimizer()
    run_config = tf.estimator.RunConfig(
        tf_random_seed=RANDOM_SEED,
        save_checkpoints_steps=100,
        save_summary_steps=100)

    wide_columns, deep_columns, tree_columns = build_feature_columns()

    if FLAGS.model_type == 'DNN':
        return tf.estimator.DNNClassifier(
            n_classes=2,
            feature_columns=deep_columns,
            activation_fn=tf.nn.relu,
            optimizer=optimizer,
            hidden_units=FLAGS.hidden_units.split(','),
            dropout=FLAGS.dropout,
            batch_norm=True,
            model_dir=model_checkpoints,
            config=run_config)
    elif FLAGS.model_type == 'WD':
        return tf.estimator.DNNLinearCombinedClassifier(
            n_classes=2,
            linear_feature_columns=wide_columns,
            linear_optimizer='Ftrl',
            dnn_feature_columns=deep_columns,
            dnn_optimizer=optimizer,
            dnn_hidden_units=FLAGS.hidden_units.split(','),
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
            feature_columns=tree_columns,
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

    model_checkpoints = os.path.join(FLAGS.job_dir, 'model')
    model_serving = os.path.join(FLAGS.job_dir, 'serving')

    if not FLAGS.cloud:
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
        model_checkpoints = model_checkpoints + '_{}'.format(timestamp)
        model_serving = model_serving + '_{}'.format(timestamp)

    trial = json.loads(os.environ.get('TF_CONFIG', '{}')).get('task', {}).get(
        'trial', '')

    if not trial:
        trial = 1

    model_checkpoints = os.path.join(model_checkpoints, str(trial))
    model_serving = os.path.join(model_serving, str(trial))

    estimator = initialize_estimator(model_checkpoints=model_checkpoints)
    estimator = tf.contrib.estimator.add_metrics(estimator, metrics)
    estimator = tf.contrib.estimator.forward_features(
        estimator, keys=['customer_identity', 'brand_identity'])

    # train / evaluate / predict

    if FLAGS.mode == "train":
        start_time = time.time()

        # stop if loss does not decrease within given max steps
        if FLAGS.early_stopping:
            early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
                estimator, metric_name='loss', max_steps_without_decrease=1000)
            hooks = [early_stopping]
        else:
            hooks = None

        results = tf.estimator.train_and_evaluate(
            estimator,
            tf.estimator.TrainSpec(
                input_fn=lambda: input_fn(
                    path_dir=FLAGS.train_data,
                    epochs=FLAGS.train_epochs,
                    shuffle=True,
                    batch_size=FLAGS.batch_size),
                hooks=hooks
            ),
            tf.estimator.EvalSpec(
                input_fn=lambda: input_fn(
                    path_dir=FLAGS.dev_data,
                    epochs=1,
                    shuffle=False,
                    batch_size=FLAGS.batch_size
                ),
                steps=1000,
                throttle_secs=60
            )
        )

        duration = time.time() - start_time

        print("Training time: {} seconds / {} minutes".format(
            round(duration, 2), round((duration/60.0), 2)))

        # export model for serving
        estimator.export_savedmodel(export_dir_base=model_serving,
                                    serving_input_receiver_fn=serving_input_receiver_fn)

        # export model settings (add results from train_and_evaluate)
        if results:
            results = results[0]
        else:
            results = {}
        results['duration'] = duration
        results['checkpoints_dir'] = model_checkpoints

        export_train_results(FLAGS, trial, results)

    # use for local predictions on a test set - for batch scoring use AI Platform predict
    elif FLAGS.mode == 'predict':
        predictions = estimator.predict(
            input_fn=lambda: input_fn(
                path_dir=FLAGS.test_data,
                shuffle=False,
                epochs=1,
                batch_size=FLAGS.batch_size
            ),
            checkpoint_path=tf.train.latest_checkpoint(FLAGS.job_dir))

        timestamp = datetime.datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
        file_name = 'predictions_{}.json'.format(timestamp)
        output_path = os.path.join(FLAGS.job_dir, file_name)

        with open(output_path, 'w') as json_output:
            for p in predictions:
                results = {
                    'customer_id': p['customer_identity'].decode('utf-8'),
                    'brand': p['brand_identity'].decode('utf-8'),
                    'predicted_label': int(p['class_ids'][0]),
                    'logistic': float(p['logistic'][0])
                }

                json_output.write(json.dumps(results, ensure_ascii=False) + '\n')

        gcs_command = 'gsutil -m cp -r ' + output_path + ' gs://{}/evaluation/{}'.format(
            FLAGS.bucket, file_name)
        subprocess.check_output(gcs_command.split())

        bq_schema = 'logistic:FLOAT,predicted_label:INTEGER,customer_id:STRING,brand:STRING'
        bq_command = ('bq --location=EU load --source_format=NEWLINE_DELIMITED_JSON propensity_dataset.{} '
                      'gs://{}/evaluation/{} {}').format(
            file_name.replace('.json', ''), FLAGS.bucket, file_name, bq_schema
        )
        subprocess.check_output(bq_command.split())

    # use for evaluation on a test set
    elif FLAGS.mode == 'evaluate':

        results = estimator.evaluate(
            input_fn=lambda: input_fn(
                path_dir=FLAGS.test_data,
                epochs=1,
                shuffle=False,
                batch_size=FLAGS.batch_size
            ),
            checkpoint_path=tf.train.latest_checkpoint(FLAGS.job_dir)
        )

        export_eval_results(FLAGS, trial, results)

    else:
        print('Unrecognised mode {}'.format(FLAGS.mode))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
