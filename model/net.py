import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn


# build model
class ContextModel(nn.Module):
    def __init__(self,
                 numeric_features,
                 categorical_features,
                 merchant_set_features,
                 feature_config,
                 input_size,
                 n_output_classes,
                 layer_config,
                 embeddingbag_mode):
        super(ContextModel, self).__init__()

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.merchant_set_features = merchant_set_features
        self.feature_config = feature_config
        self.layer_config = layer_config
        self.input_size = input_size
        self.n_output_classes = n_output_classes

        self.categorical_confs = self.feature_config['categorical_feature_vocabulary_sizes']
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=c[0], embedding_dim=c[1], padding_idx=c[0] - 1)
            for _, c in self.categorical_confs.items()
        ])

        self.merchant_conf = self.feature_config['merchant_vocabulary_sizes']
        self.merchant_set_embeddings = nn.ModuleList([nn.EmbeddingBag(num_embeddings=c[0], embedding_dim=c[1],
                                                                      mode=embeddingbag_mode) for _, c in
                                                      self.merchant_conf.items()])
        self.categorical_fcounts = len(self.categorical_confs)
        self.merchant_fcounts = len(self.merchant_conf)
        self.numeric_fcounts = 1

        # build NN
        self.model = nn.Sequential(OrderedDict([
            (name, conf['layer_type'](**conf['args'])) for name, conf in layer_config.items()
        ]))

    def forward(self, x):
        categorical_sample = x[:self.categorical_fcounts]
        merchant_sample = x[self.categorical_fcounts:self.categorical_fcounts + self.merchant_fcounts]
        numeric_sample = x[self.categorical_fcounts + self.merchant_fcounts: \
                           self.categorical_fcounts + self.merchant_fcounts + self.numeric_fcounts]

        categorical_features = [emb(val) for val, emb in zip(categorical_sample,
                                                             self.categorical_embeddings)]

        categorical_features = torch.cat(categorical_features, 1)

        merchant_set_features = [emb(val) for val, emb in zip(merchant_sample,
                                                              self.merchant_set_embeddings)]
        merchant_set_features = torch.cat(merchant_set_features, 1)

        numeric_features = torch.cat(numeric_sample, 1)

        inputs = torch.cat((categorical_features, merchant_set_features, numeric_features), 1)

        output = self.model(inputs)

        return output


def loss_fn(outputs, labels, unknown_label, weight=None):
    """
    Args:
        outputs: (Tensor) dimension batch_size x num_merchants - log softmax output of the model
        labels: (Tensor) dimension batch_size num_merchants where each element is either a label in [0, 1, ... merchant_count-1]
                or merchant_count in case it is a padding token.
        unknown_label: (Int) merchant_count
        weight: (Dict) merchant index with corresponding weights {0:0, 1:0.11, 2:0.23, ...}.
    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch
    """

    # reshape labels to give a flat vector of length batch_size
    labels = labels.view(-1)

    # since unknown tokens have label merchant_count, we can generate a mask to exclude the loss from those terms
    mask = (labels < unknown_label).float()

    num_tokens = int(torch.sum(mask))

    if weight is not None:
        mask = [weight[labels[i]] if labels[i] in weight else mask[i] for i in range(len(mask))]

    # compute cross entropy loss for all tokens (except unknown tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


def accuracy(outputs, labels, unknown_label):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.
    Args:
        outputs: (np.ndarray) dimension batch_size x num_merchants - log softmax output of the model
        labels: (np.ndarray) dimension batch_size where each element is either a label in
                [0, 1, ... merchant_count-1], or merchant_count in case it is an unknown token.
    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()  # equivalent of pytorch's view(-1)

    # since unknown tokens have label unknown_label, we can generate a mask to exclude the loss from those terms
    mask = (labels < unknown_label)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels
    return np.sum(outputs == labels) / float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': (accuracy, [('merchant_count')])
}
