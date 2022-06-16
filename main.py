"""Main training function.

(A) NodeGAM training.
- Set name. E.g. 0615_bikeshare.
- Set dataset. E.g. bikeshare.
- Set arch. E.g. GAMAtt.
- Set ga2m = 1 if training a GA2M, or 0 for training GAM.

- Output:
  - The final test accuracy is stored in the results/bikeshare_GAMAtt.csv.
  - The best model is stored in the logs/0615_bikeshare/best.ckpt.
  - The hyperparameter is stored in logs/hparams/0615_bikeshare.
  - The training and validation loss figure is in loss_figs/0615_bikeshare.jpg.
  - The training and validation results are stored in logs/0615_bikeshare/recorder.json,
    loss_history.npy (training loss history per step), and err_history.npy (val err history).
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from os.path import join as pjoin, exists as pexists
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import lib
from qhoptim.pyt import QHAdam

# Don't use multiple gpus; If more than 1 gpu, just use first one
if torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use it to create figure instead of interactive
matplotlib.use('Agg')


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--name",
                        default='debug',
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training.')
    # My own arguments
    parser.add_argument("--dataset", default='bikeshare',
                        help="Choose the dataset.",
                        choices=['year', 'epsilon', 'a9a', 'higgs', 'microsoft',
                                 'yahoo', 'click', 'mimic2', 'adult', 'churn',
                                 'credit', 'compas', 'support2', 'mimic3',
                                 'rossmann', 'wine', 'bikeshare'])
    parser.add_argument('--fold', type=int, default=0,
                        help='Choose from 0 to 4, as we only support 5-fold CV.')
    parser.add_argument("--arch", type=str, default='GAM',
                        choices=['ODST', 'GAM', 'GAMAtt'])
    parser.add_argument("--num_trees", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--addi_tree_dim", type=int, default=0)
    parser.add_argument("--l2_lambda", type=float, default=0.)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lr_warmup_steps", type=int, default=-1)
    parser.add_argument("--lr_decay_steps", type=int, default=-1,
                        help='Decay learning rate by 1/5 if not improving for this step')
    parser.add_argument("--quantile_dist", type=str, default='normal',
                        choices=['normal', 'uniform'],
                        help='Which distribution to do qunatile transform')

    parser.add_argument("--early_stopping_rounds", type=int, default=11000)
    parser.add_argument("--max_rounds", type=int, default=-1)
    parser.add_argument("--max_time", type=float, default=3600 * 20)  # At most 20 hours
    parser.add_argument("--report_frequency", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_bs", type=int, default=2048,
                        help='If batch size is None, it automatically finds the right batch size '
                             'that fits into the GPU memory between max_bs and min_bs via binary '
                             'search.')
    parser.add_argument("--min_bs", type=int, default=128)

    parser.add_argument("--random_search", type=int, default=0)
    parser.add_argument('--fp16', type=int, default=0,
                        help='Uses the 16-precision to train. Slows down 5~10%, but saves memory '
                             'and is slightly better.')
    parser.add_argument('--data_subsample', type=float, default=1.,
                        help='Between 0 and 1. Percentage of training data used. If bigger than 1, '
                             'treats it as integer and select the specified number of samples.')
    parser.add_argument('--ignore_prev_runs', type=int, default=0,
                        help='If 1, in random search, it ignores previous runs and reruns the '
                             'training even it was run before. Useful when fixing a bug.')

    temp_args, _ = parser.parse_known_args()
    # Remove stuff if in debug mode
    if temp_args.name.startswith('debug'):
        clean_up(temp_args.name)

    # Load previous hparams arch
    prev_hparams = load_from_prev_hparams(temp_args)
    if prev_hparams is not None and all([not arg.startswith('--arch') for arg in sys.argv]):
        temp_args.arch = prev_hparams['arch']
    parser = getattr(lib.arch, temp_args.arch + 'Block').add_model_specific_args(parser)
    args = parser.parse_args()

    # If loading previous hparams, update prev hparams with user inputs
    user_hparams = load_user_hparams(parser)
    if prev_hparams is not None:
        update_args(args, user_hparams, prev_hparams)

    return args, user_hparams


def load_from_prev_hparams(args):
    hparams = None
    if pexists(pjoin('logs', 'hparams', args.name)):
        with open(pjoin('logs', 'hparams', args.name)) as fp:
            hparams = json.load(fp)
    elif args.load_from_hparams is not None:
        path = args.load_from_hparams
        if '/' not in args.load_from_hparams:
            path = pjoin('logs', 'hparams', args.load_from_hparams)
        with open(path) as fp:
            hparams = json.load(fp)
    return hparams


def load_user_hparams(parser):
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return vars(parser.parse_args())


def clean_up(name):
    shutil.rmtree(pjoin('logs', name), ignore_errors=True)
    shutil.rmtree(pjoin('lightning_logs', name), ignore_errors=True)
    if pexists(pjoin('logs', 'hparams', name)):
        os.remove(pjoin('logs', 'hparams', name))


def update_args(args, user_hparams, prev_hparams):
    for k, v in prev_hparams.items():
        if k not in user_hparams:
            setattr(args, k, v)


def main():
    args, user_hparams = get_args()

    if args.random_search == 0:
        try:
            train(args)
        finally:
            if pexists(pjoin('is_running', args.name)):  # release it
                os.remove(pjoin('is_running', args.name))
        sys.exit()

    def get_rs_name(hparams, rs_hparams):
        if isinstance(hparams, argparse.Namespace):
            hparams = vars(hparams)
        tmp = '_'.join([f'{v["short_name"]}{hparams[k]}'
                        for k, v in rs_hparams.items()])
        tmp += ('' if hparams['data_subsample'] == 1 else f'_ds{hparams["data_subsample"]}')
        return tmp

    # Create a directory to record what is running
    os.makedirs('is_running', exist_ok=True)

    rs_hparams = getattr(lib.arch, args.arch + 'Block').get_model_specific_rs_hparams()

    # This makes every random search as the same order!
    if args.seed is not None:
        lib.seed_everything(args.seed)

    orig_name, num_random_search = args.name, args.random_search
    args.random_search = 0  # When sending jobs, not run the random search!!

    unsearched_set = {k for k in user_hparams if k in rs_hparams and k not in ['seed']}
    if len(unsearched_set) > 0:
        print('Do not random search following attributes:', unsearched_set)

    for r in range(num_random_search):
        for _ in range(50):  # Try 50 times if can't found, quit
            for k, v in rs_hparams.items():
                if 'gen' in v and v['gen'] is not None:
                    if k in user_hparams and k not in ['seed']:
                        continue
                    setattr(args, k, v['gen'](args))

            args.name = orig_name + '_' + get_rs_name(args, rs_hparams)

            if pexists(pjoin('is_running', args.name)):
                continue

            if (not args.ignore_prev_runs) and pexists(pjoin('logs', args.name, 'MY_IS_FINISHED')):
                continue

            Path(pjoin('is_running', args.name)).touch()

            train(args)
            break
        else:
            print('Can not find any more parameters! Quit.')
            sys.exit()


def train(args) -> None:
    # Create directory
    os.makedirs(pjoin('logs', args.name), exist_ok=True)

    if pexists(pjoin('logs', args.name, 'MY_IS_FINISHED')):
        print('Quit! Already finish running for %s' % args.name)
        return

    # Set seed
    if args.seed is not None:
        lib.utils.seed_everything(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data
    with lib.utils.Timer(f'Load dataset {args.dataset}'):
        data = lib.DATASETS[args.dataset.upper()](path='./data', fold=args.fold)

    # Dataset-dependent quantile noise. If it's set too small, the categorical features
    # will not get enough value. In general 1e-3 is a good value.
    qn = data.get('quantile_noise', 1e-3)
    preprocessor = lib.MyPreprocessor(
        cat_features=data.get('cat_features', None),
        y_normalize=(data['problem'] == 'regression'),
        random_state=1337, quantile_transform=True,
        output_distribution=args.quantile_dist,
        quantile_noise=qn,
    )

    X_train, y_train = data['X_train'], data['y_train']
    preprocessor.fit(X_train, y_train)
    if args.data_subsample > 1.:
        args.data_subsample = int(args.data_subsample)

    if args.data_subsample != 1. and args.data_subsample < X_train.shape[0]:
        print(f'Subsample the data by ds={args.data_subsample}')
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=args.data_subsample, random_state=1377,
            stratify=(y_train if data['problem'] == 'classification' else None))

    use_data_val = ('X_valid' in data and 'y_valid' in data)
    if use_data_val:
        X_valid, y_valid = data['X_valid'], data['y_valid']
    else:
        # Merge with the valid set, and cut it ourselves
        if 'X_valid' in data:
            X_train = pd.concat([X_train, data['X_valid']], axis=0)
            y_train = np.concatenate([y_train, data['y_valid']], axis=0)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=1377,
            stratify=(y_train if data['problem'] == 'classification' else None)
        )

    # Transform dataset
    X_train, y_train = preprocessor.transform(X_train, y_train)
    X_valid, y_valid = preprocessor.transform(X_valid, y_valid)
    X_test, y_test = preprocessor.transform(data['X_test'], data['y_test'])

    # Save preprocessor
    with open(pjoin('logs', args.name, 'preprocessor.pkl'), 'wb') as op:
        pickle.dump(preprocessor, op)

    metric = data.get('metric', ('classification_error'
                                 if data['problem'] == 'classification' else 'mse'))

    # Modify args based on the dataset
    args.in_features = X_train.shape[1]
    args.problem = data['problem']
    args.num_classes = data.get('num_classes', 1)
    args.data_addi_tree_dim = data.get('addi_tree_dim', 0)

    print(f'X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}')
    # Model
    model, step_callbacks = getattr(lib.arch, args.arch + 'Block').load_model_by_hparams(
        args, ret_step_callback=True)

    # Initialize bias before sending to cuda
    if 'init_bias' in args and args.init_bias and args.problem == 'classification':
        model.set_bias(y_train)

    model.to(device)

    optimizer_params = {'nus': (0.7, 1.0), 'betas': (0.95, 0.998)}

    trainer = lib.Trainer(
        model=model,
        experiment_name=args.name,
        warm_start=True,  # To handle the interruption on v server
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        verbose=False,
        n_last_checkpoints=5,
        step_callbacks=step_callbacks,  # Temp annelaing
        fp16=args.fp16,
        problem=args.problem,
    )

    assert metric in ['negative_auc', 'classification_error', 'mse']
    eval_fn = getattr(trainer, 'evaluate_' + metric)

    # Before we start, we will need to select the batch size if unspecified
    if args.batch_size is None or args.batch_size < 0:
        assert device != 'cpu', 'Have to specify batch size when using CPU'
        args.batch_size = choose_batch_size(trainer, X_train, y_train, device,
                                            max_bs=args.max_bs, min_bs=args.min_bs)
    else:
        # trigger data-aware init
        with torch.no_grad():
            res = model(torch.as_tensor(X_train[:(2 * args.batch_size)], device=device))

    # Then show hparams after deciding the batch size
    print("experiment:", args.name)
    print("Args:")
    print(args)

    # Then record hparams
    saved_args = pjoin('logs', args.name, 'hparams.json')
    json.dump(vars(args), open(saved_args, 'w'))

    # record hparams again, since logs/{args.name} will be deleted!
    os.makedirs(pjoin('logs', 'hparams'), exist_ok=True)
    json.dump(vars(args), open(pjoin('logs', 'hparams', args.name), 'w'))

    # To make sure when rerunning the err history and time are accurate,
    # we save the whole history in training.json.
    recorder = lib.Recorder(path=pjoin('logs', args.name))

    ntf_diff, ntf = 0., None  # Record number of trees assigned to each feature
    st_time = time.time()
    for batch in lib.iterate_minibatches(X_train, y_train,
                                         batch_size=args.batch_size,
                                         shuffle=True, epochs=float('inf')):
        # Handle removing missing by sampling from a Gaussian!
        metrics = trainer.train_on_batch(*batch, device=device)

        if recorder.loss_history is not None:
            recorder.loss_history.append(float(metrics['loss']))

        if trainer.step % args.report_frequency == 0:
            trainer.save_checkpoint()
            trainer.remove_old_temp_checkpoints()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')

            err = eval_fn(X_valid, y_valid,
                          device=device, batch_size=args.batch_size * 2)
            if err < recorder.best_err:
                recorder.best_err = err
                recorder.best_step_err = trainer.step
                trainer.save_checkpoint(tag='best')
            if recorder.err_history is not None:
                recorder.err_history.append(err)

            recorder.step = trainer.step
            recorder.run_time += float(time.time() - st_time)
            st_time = time.time()

            recorder.save_record()

            trainer.load_checkpoint()  # last
            if recorder.loss_history is not None and recorder.err_history is not None:
                save_loss_fig(recorder.loss_history, recorder.err_history,
                              pjoin('loss_figs', f'{args.name}.jpg'))

            cur_ntf = trainer.model.get_num_trees_assigned_to_each_feature()
            if cur_ntf is None:  # ODST no NTF
                ntf_diff = 0.
            else:
                if ntf is not None:
                    ntf_diff = (torch.sum(torch.abs(cur_ntf - ntf)) * 100.0 / torch.sum(cur_ntf)).item()
                ntf = cur_ntf

            if trainer.step == 1:
                print("Step\tVal_Err\tTime(s)\tNTF(%)")
            print('{}\t{}\t{:.0f}\t{:.2f}%'.format(
                trainer.step, np.around(err, 5), recorder.run_time, ntf_diff))

        bstep = recorder.best_step_err
        if isinstance(bstep, list):
            bstep = np.max(bstep)

        min_steps = max(bstep, getattr(args, 'anneal_steps', -1))
        if trainer.step > min_steps + args.early_stopping_rounds:
            print('BREAK. There is no improvment for {} steps'.format(
                args.early_stopping_rounds))
            break

        if args.lr_decay_steps > 0 \
                and trainer.step > bstep + args.lr_decay_steps \
                and trainer.step > (recorder.lr_decay_step + args.lr_decay_steps):
            lr_before = trainer.lr
            trainer.decrease_lr(ratio=0.2, min_lr=1e-6)
            recorder.lr_decay_step = trainer.step
            print('LR: %.2e -> %.2e' % (lr_before, trainer.lr))

        if 0 < args.max_rounds < trainer.step:
            print('End. It reaches the maximum rounds %d' % args.max_rounds)
            break

        if recorder.run_time > args.max_time:
            print('End. It reaches the maximum run time %d (s)' % args.max_time)
            break

    print("Best step: ", recorder.best_step_err)
    print("Best Val Error: ", recorder.best_err)

    max_step = trainer.step
    # Run test time
    trainer.load_checkpoint(tag='best')
    test_err = eval_fn(X_test, y_test,
                       device=device, batch_size=2 * args.batch_size)
    print("Test Error rate: {}".format(test_err))

    # Save csv results
    results = dict()
    results['test_err'] = test_err
    results['val_err'] = recorder.best_err
    results['best_step_err'] = recorder.best_step_err
    results['max_step'] = max_step
    results['time(s)'] = '%d' % recorder.run_time
    results['fold'] = args.fold
    results['fp16'] = args.fp16
    results['batch_size'] = args.batch_size
    # Append the hyperparameters
    rs_hparams = getattr(lib.arch, args.arch + 'Block').get_model_specific_rs_hparams()
    for k in rs_hparams:
        results[k] = getattr(args, k)

    results = getattr(lib.arch, args.arch + 'Block').add_model_specific_results(results, args)
    results['name'] = args.name

    os.makedirs(f'results', exist_ok=True)
    dataset_postfix = f'_ds{args.data_subsample}' if args.data_subsample != 1. else ''
    csv_file = f'results/{args.dataset}{dataset_postfix}_{args.arch}.csv'
    lib.utils.output_csv(csv_file, results)
    print('output results to %s' % csv_file)

    # Clean up
    open(pjoin('logs', args.name, 'MY_IS_FINISHED'), 'a')
    trainer.remove_old_temp_checkpoints(number_ckpts_to_keep=0)


def choose_batch_size(trainer, X_train, y_train, device, max_bs=4096, min_bs=64):
    def clean_up_memory():
        for p in trainer.model.parameters():
            p.grad = None
        torch.cuda.empty_cache()

    # Starts with biggest batch size. Capped by training size
    bs = min(max_bs, X_train.shape[0])
    min_bs = min(min_bs, X_train.shape[0])

    shuffle_indices = np.random.permutation(X_train.shape[0])

    while True:
        try:
            if bs < min_bs:
                raise RuntimeError('The batch size %d is smaller than mininum %d'
                                   % (bs, min_bs))
            print('Trying batch size %d ...' % bs)
            trainer.train_on_batch(
                X_train[shuffle_indices[:bs]], y_train[shuffle_indices[:bs]],
                device=device, update=False)
            break
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise e

            print('| batch size %d failed.' % (bs))
            bs = bs // 2
            if bs < min_bs:
                raise e
            continue
        finally:
            clean_up_memory()

    print('Choose batch size %d.' % (bs))
    return bs


def save_loss_fig(loss_history, err_history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # At last, save the loss figure
    plt.figure(figsize=[18, 6])
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('Loss')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(err_history)
    plt.title('Error')
    plt.grid()
    plt.savefig(path, bbox_inches='tight')
    # plt.show()
    plt.close()



if __name__ == '__main__':
    main()
