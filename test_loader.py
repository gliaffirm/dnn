import argparse
import logging
import os
import utils
import boto3
import pickle
from itertools import chain

from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from tqdm import trange

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

def train(model, optimizer, loss_fn, data_iterator, metrics, params, merchant_count, num_steps):
    """Run num_steps batches
    Args:
        model: (torch.nn.Module) model object
        optimizer: (torch.optim) optimizer
        loss_fn: loss function (cross-entropy)
        data_iterator: (generator) batch generator, usually of the form iter(dataset_loader)
        metrics: (dict) dict of performance metrics, calculated at the batch level
        params: (Params) configs and hyperparameters
        merchant_count: (int) need this to evaluate loss
        num_steps: (int) number of batches to train on
    """
    # train mode
    model.train()

    # summary for current training loop and a running average object for loss
    summaries = []
    loss_avg = utils.RunningAverage()

    # pretty tqdm progress bar because yolo
    t = trange(num_steps)
    for i in t:
        batch = next(data_iterator)
        inputs = batch[:-1].cuda(non_blocking=True) if params.cuda else batch[:-1]
        labels = batch[-1].cuda(non_blocking=True) if params.cuda else batch[-1]

        optimizer.zero_grad()

        # compute model output and loss
        outputs = model(inputs)
        loss = loss_fn(outputs, labels, unknown_label=merchant_count)

        loss.backward()
        optimizer.step()

        # Evaluation
        if i % params.save_summary_steps == 0:
            outputs = outputs.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            # compute all metrics on this batch
            summary = {
                metric: metrics[metric](outputs, labels)
                for metric in metrics
            }
            summary['loss'] = loss.item()
            summaries.append(summary)

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    metrics_mean = {
        metric: np.mean([s[metric] for s in summaries])
                for metric in summaries[0]
    }
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())

    logging.info("- Train metrics: " + metrics_string)

def train_and_evaluate(model,
                       optimizer,
                       loss_fn,
                       training_data,
                       validation_data,
                       metrics,
                       params,
                       feature_config,
                       model_dir,
                       restore_file=None):

    """Main training method; evaluate every epoch.
    Args:
        model: (torch.nn.Module) model object
        optimizer: (torch.optim) optimizer
        loss_fn: loss function, (cross-entropy)
        training_data: (dict) training data
        validation_data: (dict) validation data
        metrics: (dict) dict of performance metrics, calculated at the batch level
        params: (Params) configs and hyperparameters
        model_dir: (string) directory containing config, weights, and logs
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    # best value achieved
    best_accuracy = 0.0

    bucket_name = configs.BUCKET_NAME
    numeric_features = configs.NUMERIC_VECTOR
    categorical_features = configs.CATEGORICAL_FEATURES
    merchant_set_features = configs.MERCHANT_SET_FEATURES
    label_col = configs.LABEL_COL
    index_col = configs.INDEX_COL
    merchant_count = params.merchant_count

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch
        num_steps = (params.training_size) // params.batch_size

        # Get parquet file names for streaming from S3
        file_list = params.training_file_list[epoch]
        training_dataset = data_loader.ContextDataset(start=0, end=len(file_list),
                                                      bucket_name=bucket_name,
                                                      file_list=file_list,
                                                      feature_config=feature_config,
                                                      label_col=label_col,
                                                      index_col=index_col,
                                                      merchant_count=merchant_count)

        loader = DataLoader(training_dataset,
                            batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            worker_init_fn=data_loader.worker_init_fn,
                            pin_memory=True)

        data_iterator = iter(loader)
        train(model, optimizer, loss_fn, data_iterator, metrics, params, merchant_count, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = (params.validation_size) // params.batch_size
        validation_file_list = params.validation_file_list
        validation_dataset = data_loader.ContextDataset(start=0,
                                                        end=len(validation_file_list),
                                                        bucket_name=bucket_name,
                                                        file_list=validation_file_list,
                                                        feature_config=feature_config,
                                                        label_col=label_col, index_col=index_col)

        loader = DataLoader(validation_dataset,
                            batch_size=params.batch_size,
                            num_workers=params.num_workers,
                            worker_init_fn=data_loader,
                            pin_memory=True)

        # switch model to evaluation mode
        data_iterator = iter(loader)
        num_steps = (params.validation_size) // params.batch_size
        val_metrics = evaluate(
            model, loss_fn, data_iterator, metrics, params, merchant_count, num_steps
        )

        validation_accuracy = val_metrics['accuracy']
        is_best = validation_accuracy >= best_accuracy
        logging.info("- Validation accuracy: " + '{:05.3f}'.format(validation_accuracy))

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_accuracy = validation_accuracy

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json({'accuracy': best_accuracy}, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json({'accuracy': validation_accuracy}, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params_loader.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Create the input data pipeline
    logging.info("Loading params...")
    s3 = boto3.resource('s3')
    feature_conf, layer_conf, saved_params = pickle.loads(s3.Bucket(configs.BUCKET_NAME)
                                                          .Object(configs.CONFIG_KEY)
                                                          .get()['Body'].read())

    logging.info("- done.")

    print(params.dict)
    params.dict.update(saved_params)
    print(params.dict)
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(23)
    if params.cuda:
        torch.cuda.manual_seed(23)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Define model and optimiser
    model = net.ContextModel(numeric_features=configs.NUMERIC_FEATURES,
                             categorical_features=configs.CATEGORICAL_FEATURES,
                             merchant_set_features=configs.MERCHANT_SET_FEATURES,
                             feature_config=feature_conf,
                             input_size=params.input_size,
                             n_output_classes=params.merchant_count,
                             layer_config=layer_conf,
                             embeddingbag_mode=configs.EMBEDDINGBAG_MODE)

    if params.cuda: model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))

    bucket_name = configs.BUCKET_NAME
    numeric_features = configs.NUMERIC_VECTOR
    categorical_features = configs.CATEGORICAL_FEATURES
    merchant_set_features = configs.MERCHANT_SET_FEATURES
    label_col = configs.LABEL_COL
    index_col = configs.INDEX_COL
    merchant_count = params.merchant_count

    # Run one epoch
    num_steps = (params.training_size) // params.batch_size

    # Get parquet file names for streaming from S3
    epoch = 0
    file_list = params.training_file_list[epoch]
    training_dataset = data_loader.ContextDataset(start=0, end=len(file_list),
                                                  bucket_name=bucket_name,
                                                  file_list=file_list,
                                                  feature_config=feature_conf,
                                                  label_col=label_col,
                                                  index_col=index_col,
                                                  merchant_count=merchant_count)

    loader = DataLoader(training_dataset,
                        batch_size=params.batch_size,
                        num_workers=params.num_workers,
                        worker_init_fn=data_loader.worker_init_fn,
                        pin_memory=True)

    data_iterator = iter(loader)

    print(file_list[:10])
    print('input size')
    print(params.input_size)
    print('training_size')
    print(params.training_size)
    print('num_steps')
    print(num_steps)
    for i in range(20):
        it = next(data_iterator)
        print('workers {}'.format(it[1]))
        print('indices {}'.format(it[0][-1]))
        print(it[0])
