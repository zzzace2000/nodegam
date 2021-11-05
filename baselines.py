import os, sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lib
import torch, torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from os.path import join as pjoin, exists as pexists
import shutil
import platform
from filelock import FileLock
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle
from lib.gams import model_utils
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


# Use it to create figure instead of interactive
matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser(description='Training a classifier serving as reward for RL')

    ## General
    parser.add_argument('--name', type=str, default='debug',
                        help='The unique identifier for the model')
    parser.add_argument('--seed', type=int, default=1377, help='random seed')
    parser.add_argument('--output_dir', type=str, default='models/',
                        help='Model saved directory. Default set as models/')
    # parser.add_argument('--overwrite', type=int, default=1,
    #                     help='if set as 1, then it would remove the previous models with same identifier.' \
    #                          'If 0, then use the stored model to make test prediction.')
    parser.add_argument('--check_in_records', type=int, default=1,
                        help='If set as 1, then check if the output csv already has the result. '
                             'If so, skip.')
    parser.add_argument('--fold', type=int, default=0,
                        help='Choose from 0 to 4, as we only support 5-fold CV.')
    # parser.add_argument('--n_splits', type=int, default=5,
    #                     help='Rerun the experiment for this number of splits')

    ## Which model and dataset to run
    parser.add_argument('--model_name', type=str, default='xgb-o10')
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--data_subsample', type=float, default=1.,
                        help='Between 0 and 1. Percentage of training data used. '
                             'If bigger than 1, then select these '
                             'number of samples.')

    ## What resource to use: just for recording
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--mem', type=int, default=8)
    parser.add_argument('--cpu', type=int, default=4)

    args = parser.parse_args()

    # Set seed
    lib.utils.seed_everything(args.seed)

    # Remove debug folder
    if args.name == 'debug':
        print("WATCHOUT!!! YOU ARE RUNNING IN TEST RUN MODE!!!")
        if pexists('./logs/debug'):
            shutil.rmtree('./logs/debug', ignore_errors=True)

        args.data_subsample = 0.01

    # on v server
    if not pexists(pjoin('logs', args.name)) \
            and 'SLURM_JOB_ID' in os.environ \
            and pexists('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID']):
        if os.path.islink(pjoin('logs', args.name)):
            os.remove(pjoin('logs', args.name))
        os.symlink('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID'],
                   pjoin('logs', args.name))
    return args


def main(args) -> None:
    # Create directory
    os.makedirs(pjoin('logs', args.name), exist_ok=True)

    # Data
    data = lib.DATASETS[args.dataset.upper()](path='./data', fold=args.fold)
    X_train, X_test, y_train, y_test = \
        data['X_train'], data['X_test'], data['y_train'], data['y_test']

    extra_args = {}
    if 'X_valid' in data: # For those NODE datasets
        X_train = pd.concat([X_train, data['X_valid']], axis=0)
        y_train = np.concatenate([y_train, data['y_valid']], axis=0)

        if args.model_name.startswith('xgb') or args.model_name.startswith('ebm'):
            # Change the validation size
            extra_args['validation_size'] = \
                data['X_valid'].shape[0] * 1.0 / X_train.shape[0]
            print('Change val ratio:', extra_args['validation_size'])

    # Do target transform
    if 'cat_features' in data:
        preprocessor = lib.MyPreprocessor(
            cat_features=data.get('cat_features', None),
        )

        preprocessor.fit(X_train, y_train)
        X_train, y_train = preprocessor.transform(X_train, y_train)
        X_test, y_test = preprocessor.transform(X_test, y_test)
        # Save preprocessor
        with open(pjoin('logs', args.name, 'preprocessor.pkl'), 'wb') as op:
            pickle.dump(preprocessor, op)

    ## Subsampling
    if args.data_subsample > 1.:
        args.data_subsample = int(args.data_subsample)
    # Do not subsample data in the pretraining!
    if args.data_subsample != 1. and args.data_subsample < X_train.shape[0]:
        print(f'Subsample the data by ds={args.data_subsample}')
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=args.data_subsample, random_state=1377,
            stratify=(y_train if data['problem'] == 'classification' else None))

    args.in_features = X_train.shape[1]
    args.problem = data['problem']

    # Record hparams
    print("experiment:", args.name)
    print("Args:")
    print(args)
    saved_args = pjoin('logs', args.name, 'hparams.json')
    json.dump(vars(args), open(saved_args, 'w'))
    # record hparams again, since logs/{args.name} will be deleted!
    os.makedirs(pjoin('logs', 'hparams'), exist_ok=True)
    json.dump(vars(args), open(pjoin('logs', 'hparams', args.name), 'w'))

    # Train model
    start_time = time.time()
    model = model_utils.get_model(
        X_train, y_train, args.problem, args.model_name, random_state=args.seed,
        **extra_args
    )
    with open(pjoin('logs', args.name, 'model.pkl'), 'wb') as op:
        pickle.dump(model, op)

    # Evaluate
    result = {}
    if args.problem == 'classification':
        y_pred = model.predict_proba(X_test)[:, 1]
        result['test_auc'] = roc_auc_score(y_test, y_pred)
        result['test_acc'] = np.mean(np.round(y_pred) == y_test)
        print('Test AUC: {:.2f}%, Test acc: {:.2f}%'.format(
            result['test_auc'] * 100, result['test_acc'] * 100))
    else:
        result['test_mse'] = ((y_test - model.predict(X_test)) ** 2).mean()

    result['name'] = args.name
    result['model_name'] = args.model_name
    result['dataset'] = args.dataset
    result['fold'] = args.fold
    result['seed'] = args.seed
    result['time(s)'] = '%d' % (time.time() - start_time)

    print('finish %s %s with %ds' % (args.dataset, args.model_name,
                                      time.time() - start_time))
    os.makedirs(f'results', exist_ok=True)
    ds_str = f'_ds{args.data_subsample}' if args.data_subsample != 1. else ''
    lib.utils.output_csv(f'results/baselines_{args.dataset}{ds_str}.csv', result)

    open(pjoin('logs', args.name, 'MY_IS_FINISHED'), 'a')


if __name__ == '__main__':
    args = get_args()
    main(args)
