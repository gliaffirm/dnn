import boto3
import pickle
import numpy as np

import torch.nn as nn
from collections import OrderedDict

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import IntegerType, DoubleType, ArrayType
from pyspark.sql import functions as fn


class Configs(object):
    TRAINING_BASE_PATH = 's3://affirm-risk-sherlock/ml/adhoc/xz/dnn_v0/training/'
    TRAINING_DATA_PATH = TRAINING_BASE_PATH + 'epoch/{}/'
    VALIDATION_DATA_PATH = 's3://affirm-risk-sherlock/ml/adhoc/xz/dnn_v0/validation/'
    TEST_DATA_PATH = 's3://affirm-risk-sherlock/ml/adhoc/xz/dnn_v0/test/'

    JOINED_PATH = 's3://affirm-risk-sherlock/ml/adhoc/xz/dnn_v0/top10000/joined/'
    MERCHANT_IDX_PATH = 's3://affirm-risk-sherlock/ml/adhoc/xz/dnn_v0/merchant_idx/'

    INDEX_COL = [
        'idx',
    ]

    NUMERIC_FEATURES = [
        'start_count_vector_minmax',
        'capture_count_vector_minmax',
        'capture_rate_vector_minmax',
        'mean_amount_requested_vector_minmax',
        'max_amount_requested_vector_minmax',
        'mean_downpayment_amount_vector_minmax',
        'mean_loan_amount_vector_minmax',
        'mean_term_vector_minmax',
        'mean_annual_income_vector_minmax',
        'mean_fico_vector_minmax',
        'user_age_days_vector_minmax',
        'account_age_days_vector_minmax',
        'time_since_last_app_days_vector_minmax',
    ]

    CATEGORICAL_FEATURES = [
        'merchant_industry_last_idx',
        'product_type_last_idx',
        'email_domain_last_idx',
        'email_domain_suffix_last_idx',
        'billing_address_city_last_idx',
        'billing_address_region_last_idx',
    ]

    MERCHANT_SET_FEATURES = [
        'merchant_set',
    ]

    LABEL_COL = [
        'label',
    ]

    NUMERIC_VECTOR = ['numeric_features']  # contains all numeric features concatenated together

    JOINED_COLUMNS = (
            INDEX_COL
            + NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
            + MERCHANT_SET_FEATURES
            + LABEL_COL
    )

    TRAIN_VALIDATE_TEST_SPLIT = [0.7, 0.15, 0.15]
    MERCHANT_SET_TRUNC = 5  # take only up to first 7 elements of each merchant set for padding
    MERCHANT_SET_SIZE = 10  # pads up to MERCHANT_SET_SIZE elements for the merchant set

    BUCKET_NAME = 'affirm-risk-sherlock'
    CONFIG_KEY = 'ml/adhoc/xz/dnn_v0/configs.pkl'

    TRAINING_PREFIX = 'ml/adhoc/xz/dnn_v0/training/epoch/{}/'
    VALIDATION_PREFIX = 'ml/adhoc/xz/dnn_v0/validation/'
    TEST_PREFIX = 'ml/adhoc/xz/dnn_v0/test/'
    TRAINING_PARTITIONS = 6
    VALIDATION_PARTITIONS = 2
    TEST_PARTITIONS = 2

    EMBEDDING_SIZES = {
        'merchant_set': 256,
        'merchant_industry_last_idx': 50,
        'product_type_last_idx': 10,
        'email_domain_last_idx': 50,
        'email_domain_suffix_last_idx': 10,
        'billing_address_city_last_idx': 128,
        'billing_address_region_last_idx': 128,
    }
    EMBEDDINGBAG_MODE = 'sum'

    NET_PARAMS = {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "num_workers": 4,
        "num_epochs": 3,
        "save_summary_steps": 100,
    }

    def _init__(self):
        pass


class FeatureConfig(object):
    def __init__(self,
                 merchant_count,
                 training_df,
                 numeric_features,
                 categorical_features,
                 merchant_set_features,
                 embedding_sizes,
                 ):
        self.merchant_count = merchant_count
        self.training_df = training_df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.merchant_set_features = merchant_set_features
        self.embedding_sizes = embedding_sizes

    def get_config(self):  # returns a dictionary of tuples specifying size and embedding size if applicable
        numeric_features_size = len(self.numeric_features)

        categorical_token_counts = (
            self.training_df
                .agg(*[fn.countDistinct(c).alias(c) for c in self.categorical_features])
                .collect()[0]
        )
        categorical_feature_vocabulary_sizes = OrderedDict(
            (c, (categorical_token_counts[c] + 1, self.embedding_sizes[c]))  # bucket at the end is for unknown
            for c in self.categorical_features
        )

        merchant_vocabulary_size = (
            self.merchant_count + 1, self.embedding_sizes['merchant_set'])  # bucket at the end is for unknown

        config = {
            'numeric_features_size': (numeric_features_size, None),
            'categorical_feature_vocabulary_sizes': categorical_feature_vocabulary_sizes,
            'merchant_vocabulary_sizes': OrderedDict([('merchant_set', merchant_vocabulary_size)]),
        }

        return config


class LayerConfig(object):
    def __init__(self, input_size, n_output_classes):
        self.input_size = input_size
        self.n_output_classes = n_output_classes

    def get_config(self):
        _out_sizes = [128, 64, ]
        config = OrderedDict([
            ('fc0', {'layer_type': nn.Linear,
                     'args': {'in_features': self.input_size, 'out_features': _out_sizes[0], 'bias': True}}),
            ('activation0', {'layer_type': nn.ReLU, 'args': {}}),
            ('dropout0', {'layer_type': nn.Dropout, 'args': {'p': 0.5}}),
            ('fc1', {'layer_type': nn.Linear,
                     'args': {'in_features': _out_sizes[0], 'out_features': self.n_output_classes, 'bias': True}}),
            ('logsoftmax', {'layer_type': nn.LogSoftmax
                , 'args': {'dim': 1}}),
        ])

        return config, self.input_size, self.n_output_classes


configs = Configs()


def load_dfs(spark,
             joined_path,
             merchant_idx_path,
             joined_columns,
             index_column,
             label_column,
             numeric_features,
             categorical_features,
             merchant_set_features,
             train_validate_test_split):
    joined_df = spark.read.parquet(joined_path).select(joined_columns)
    feature_assembler = VectorAssembler(
        inputCols=numeric_features,
        outputCol='numeric_features_vector'
    )
    joined_df = feature_assembler.transform(joined_df)
    to_array_double = fn.udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))
    joined_df = joined_df.withColumn('numeric_features', to_array_double('numeric_features_vector'))

    joined_df = joined_df.drop('numeric_features_vector')
    training_df, validation_df, test_df = joined_df.select(
        *(index_column
          + categorical_features
          + merchant_set_features
          + label_column
          + ['numeric_features'])
    ).randomSplit(train_validate_test_split, 23)

    merchant_count = spark.read.parquet(merchant_idx_path).count()

    return {
        'training_df': training_df,
        'validation_df': validation_df,
        'test_df': test_df,
        'merchant_count': merchant_count
    }


# helper function for creating minibatches
def _repartition_and_shuffle(df, rand_multiplier, num_partitions, rand_seed):
    df = df.withColumn('temp_col', fn.round(
        rand_multiplier * fn.rand(seed=rand_seed)
    ).cast(IntegerType()))

    cols = df.columns

    df = (
        df
            .rdd
            .map(lambda r: (r.temp_col, r))
            .repartitionAndSortWithinPartitions(
            numPartitions=num_partitions,
            partitionFunc=lambda x: round(x) % num_partitions,
            ascending=True,
            keyfunc=lambda x: x,
        )
            .map(lambda x: x[1])
    ).toDF(cols)

    df = df.drop('temp_col')

    return df


def create_batch_files(df, filepath, num_partitions, epoch, shuffle=True):
    # deterministically shuffle using epoch number * 100
    if shuffle:
        df = _repartition_and_shuffle(df, 500000, num_partitions, epoch)
    df.write.parquet(filepath, mode='overwrite')

    return True


# calculate merchant weights for loss function. Using the same logic as sklearn.utils.class_weight for
# 'balance' type.
def merchant_weight(df, col_name, _type='balance'):
    df = df.select(fn.explode(df[col_name]).alias(col_name))
    merchant_counts = df.groupby(col_name).count()
    total_num_merchants = df.count()
    num_merchants = merchant_counts.count()

    if _type == 'balance':
        weights = {i[col_name]: float(total_num_merchants) / (num_merchants * i['count']) for i in
                   merchant_counts.collect()}

    # normalize weights to prevent large weight values on long tail merchants. The large weights
    # may cause the gradient exploding.
    min_count = np.min(weights.values())
    max_min_range = np.ptp(weights.values())
    weights = {i: (j - min_count) / max_min_range for i, j in weights.items()}

    return weights


# Load data
data = load_dfs(spark,
                configs.JOINED_PATH,
                configs.MERCHANT_IDX_PATH,
                configs.JOINED_COLUMNS,
                configs.INDEX_COL,
                configs.LABEL_COL,
                configs.NUMERIC_FEATURES,
                configs.CATEGORICAL_FEATURES,
                configs.MERCHANT_SET_FEATURES,
                configs.TRAIN_VALIDATE_TEST_SPLIT)

# Add to configs
training_df = data['training_df']
validation_df = data['validation_df']
test_df = data['test_df']
merchant_count = data['merchant_count']

params = configs.NET_PARAMS

# calculate weights for merchants
weights = merchant_weight(training_df, configs.MERCHANT_SET_FEATURES[0])
params['weights'] = weights

# specify the train and val dataset sizes
params['training_size'] = training_df.count()
params['validation_size'] = validation_df.count()
params['test_size'] = test_df.count()
params['merchant_count'] = merchant_count

# Define configs
feature_conf = FeatureConfig(merchant_count=merchant_count,
                             training_df=training_df,
                             numeric_features=configs.NUMERIC_FEATURES,
                             categorical_features=configs.CATEGORICAL_FEATURES,
                             merchant_set_features=configs.MERCHANT_SET_FEATURES,
                             embedding_sizes=configs.EMBEDDING_SIZES).get_config()
input_size = (
        feature_conf['numeric_features_size'][0]
        + sum([v[1] for _, v in feature_conf['categorical_feature_vocabulary_sizes'].items()])
        + sum([v[1] for _, v in feature_conf['merchant_vocabulary_sizes'].items()])
)

params['input_size'] = input_size

layer_conf, _, _ = LayerConfig(input_size=input_size, n_output_classes=merchant_count).get_config()

# Shuffle and create files
bucket = boto3.resource('s3').Bucket(configs.BUCKET_NAME)
training_file_list = []
for epoch in range(configs.NET_PARAMS['num_epochs']):
    create_batch_files(training_df,
                       configs.TRAINING_DATA_PATH.format(epoch),
                       configs.TRAINING_PARTITIONS,
                       epoch,
                       shuffle=True)
    keys = [obj.key for obj in bucket.objects.filter(Prefix=configs.TRAINING_PREFIX.format(epoch)) if
            'parquet' in obj.key]
    training_file_list.append(['{bucket}/{key}'.format(bucket=configs.BUCKET_NAME, key=k) for k in keys])

create_batch_files(validation_df,
                   configs.VALIDATION_DATA_PATH,
                   configs.VALIDATION_PARTITIONS,
                   0,
                   shuffle=False)
keys = [obj.key for obj in bucket.objects.filter(Prefix=configs.VALIDATION_PREFIX) if 'parquet' in obj.key]
validation_file_list = ['{bucket}/{key}'.format(bucket=configs.BUCKET_NAME, key=k) for k in keys]

create_batch_files(test_df,
                   configs.TEST_DATA_PATH,
                   configs.TEST_PARTITIONS,
                   0,
                   shuffle=False)
keys = [obj.key for obj in bucket.objects.filter(Prefix=configs.TEST_PREFIX) if 'parquet' in obj.key]
test_file_list = ['{bucket}/{key}'.format(bucket=configs.BUCKET_NAME, key=k) for k in keys]

params['training_file_list'] = training_file_list
params['validation_file_list'] = validation_file_list
params['test_file_list'] = test_file_list

pickle_byte_obj = pickle.dumps([feature_conf, layer_conf, params])
s3_resource = boto3.resource('s3')
s3_resource.Object(configs.BUCKET_NAME, configs.CONFIG_KEY).put(Body=pickle_byte_obj)

# Load it up this way for now: make this something that is better configurable later
s3 = boto3.resource('s3')
feature_conf, layer_conf, params = pickle.loads(s3.Bucket(configs.BUCKET_NAME)
                                                .Object(configs.CONFIG_KEY)
                                                .get()['Body'].read())
