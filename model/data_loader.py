import math
import numpy as np
import pandas as pd

from itertools import chain

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from . import configs

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    worker_id = worker_info.id
    overall_start = 0
    overall_end = len(dataset.file_list)

    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


class ContextDataset(IterableDataset):
    def __init__(self, start, end, bucket_name, file_list, feature_config,
                 label_col, index_col, merchant_count):
        super(ContextDataset, self).__init__()
        assert end > start, 'start and end are fubar'
        self.start = start
        self.end = end
        self.bucket_name = bucket_name
        self.file_list = file_list[start:end]
        self.feature_config = feature_config
        self.label_col=label_col
        self.index_col=index_col
        self.merchant_count = merchant_count

    def pad_merchant_set(self, merchant_set, trunc_length, output_length):
        truncated_set = merchant_set[:trunc_length]
        return np.pad(truncated_set, (0, output_length - len(truncated_set)),
                      mode='constant', constant_values=self.merchant_count)

    def process_data(self, data):  # data is a filepath including bucket
        #dataset = pq.ParquetDataset(data, filesystem=self.s3)
        #pd_df = dataset.read_pandas().to_pandas()
        # no idea why the above doesn't work on an EMR (but we prob need this because perf)
        pd_df = pd.read_parquet('s3://{}'.format(data))  # because FML
        # iterate over numpy arrays instead of pandas series because pandas sucks go brr
        categorical_features = self.feature_config['categorical_feature_vocabulary_sizes'].keys()
        merchant_features = self.feature_config['merchant_vocabulary_sizes'].keys()
        numeric_features = ['numeric_features']
        categorical_feature_list = [torch.LongTensor(pd_df[feature].values) for feature in categorical_features]

        merchant_feature_list = []
        for feature in merchant_features:
            merchant_set_list = pd_df[feature].tolist()
            padded = [self.pad_merchant_set(e, configs.MERCHANT_SET_TRUNC, configs.MERCHANT_SET_SIZE) for e in merchant_set_list]
            merchant_feature_list.append(torch.LongTensor(np.vstack(padded)))

        numeric_feature_list = [torch.FloatTensor(np.vstack(pd_df[feature].values)) for feature in numeric_features]

        labels = [torch.LongTensor(pd_df[self.label_col].values)]

        #categorical_fcounts = len(categorical_features)
        #merchant_fcounts = len(merchant_features)
        #numeric_fcounts = len(numeric_features)

        for i in zip(*(categorical_feature_list + merchant_feature_list + numeric_feature_list + labels)):
            #categorical_sample = i[:categorical_fcounts]
            #merchant_sample = i[categorical_fcounts:categorical_fcounts + merchant_fcounts]
            #numeric_features = i[categorical_fcounts + merchant_fcounts: \
            #                     categorical_fcounts + merchant_fcounts + numeric_fcounts]
            #labels = i[-1]
            yield i

    def __iter__(self):
        return chain.from_iterable(map(self.process_data, self.file_list[self.start:self.end]))
