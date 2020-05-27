from collections import OrderedDict

import torch.nn as nn

TRAINING_BASE_PATH =  's3://affirm-risk-sherlock/ml/adhoc/xz/dnn_v0/training/'
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
            [(c, (categorical_token_counts[c] + 1, self.embedding_sizes[c]))  # bucket at the end is for unknown
            for c in self.categorical_features]
        )

        merchant_vocabulary_size = (self.merchant_count + 1, self.embedding_sizes['merchant_set'])  # bucket at the end is for unknown

        config = {
            'numeric_features_size': (numeric_features_size, None),
            'categorical_feature_vocabulary_sizes': categorical_feature_vocabulary_sizes,
            'merchant_vocabulary_sizes': OrderedDict([('merchant_set', merchant_vocabulary_size)]),
        }

        return config


# Layer config
class LayerConfig(object):
    def __init__(self, input_size, n_output_classes):
        self.input_size = input_size
        self.n_output_classes = n_output_classes

    def get_config(self):
        _out_sizes = [128, 64,]
        config = OrderedDict([
            ('fc0', {'layer_type': nn.Linear, 'args': {'in_features': self.input_size, 'out_features': _out_sizes[0], 'bias': True}}),
            ('activation0', {'layer_type': nn.ReLU, 'args': {}}),
            ('dropout0', {'layer_type': nn.Dropout, 'args': {'p': 0.5}}),
            ('fc1', {'layer_type': nn.Linear,
                    'args': {'in_features': _out_sizes[0], 'out_features': self.n_output_classes, 'bias': True}}),
            ('logsoftmax', {'layer_type': nn.LogSoftmax
            , 'args': {'dim': 1}}),
        ])

        return config, self.input_size, self.n_output_classes
