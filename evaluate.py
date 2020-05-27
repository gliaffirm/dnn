import argparse
import logging
import os
from s3fs import S3FileSystem

from torch.utils.data import IterableDataset, DataLoader, get_worker_info

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

import model.net as net
from model import data_loader, configs

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='data/',
                    help="Path containing the dataset")
parser.add_argument('--model_dir',
                    default='experiments/base_model/',
                    help="Path containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

def evaluate(model, loss_fn, data_iterator, metrics, params, merchant_count, epoch, num_steps, writer):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        merchant_count: (int) need this to calculate loss
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summaries = []

    # compute metrics over the dataset
    with torch.no_grad(): 
        for _ in range(num_steps):
            batch = next(data_iterator)
            batch = [val.cuda(non_blocking=True) if params.cuda else val for val in batch]
            inputs = batch[:-1]
            labels = batch[-1]

            # compute model output and loss
            outputs = model(inputs)
            loss = loss_fn(outputs, labels, unknown_label=merchant_count)

            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            # compute all metrics on this batch
            summary = {
                metric: metrics[metric][0](outputs, labels, *[params.dict[a] for a in metrics[metric][1]])
                for metric in metrics  # we can pass in arguments to metric from params
            }

            summary['loss'] = loss.item()
            summaries.append(summary)

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([s[metric] for s in summaries])
                for metric in summaries[0]
    }
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    writer.add_scalar('Loss/validation', metrics_mean['loss'], epoch)
    writer.add_scalar('Accuracy/validation', metrics_mean['accuracy'], epoch)

    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)


    # Create the input data pipeline
    logging.info("Loading params...")
    s3 = boto3.resource('s3')
    feature_conf, layer_conf, params = pickle.loads(s3.Bucket(configs.BUCKET_NAME)
                                                    .Object(configs.CONFIG_KEY)
                                                    .get()['Body'].read())

    logging.info("- done.")

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(23)
    if params.cuda:
        torch.cuda.manual_seed(23)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")
    file_list = params.test_file_list
    training_dataset = data_loader.ContextDataset(start=0, end=len(file_list),
                                                  bucket_name=bucket_name,
                                                  file_list=file_list,
						  feature_config=feature_conf,
                                                  label_col=label_col,
						  index_col=index_col,
                                                  merchant_count=params.merchant_count)

    loader = DataLoader(training_dataset,
                        batch_size=params.batch_size,
                        num_workers=params.num_workers,
                        worker_init_fn=data_loader.worker_init_fn,
                        pin_memory=True)

    data_iterator = iter(loader)
    logging.info("- done.")

    # Define the model
    model = net.ContextModel(numeric_features=configs.NUMERIC_FEATURES,
                             categorical_features=configs.CATEGORICAL_FEATURES,
                             merchant_set_features=configs.MERCHANT_SET_FEATURES,
                             feature_config=feature_conf,
                             input_size=params.input_size,
                             n_output_classes=params.merchant_count,
                             layer_config=layer_conf,
                             embeddingbag_mode=configs.EMBEDDINGBAG_MODE)
    if params.cuda:
        model = model.cuda()

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size) // params.batch_size
    test_metrics = evaluate(model, loss_fn, data_iterator, metrics, params, merchant_count, num_steps)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
