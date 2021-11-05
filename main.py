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

# Don't use multiple gpus; If more than 1 gpu, just use first one
if torch.cuda.device_count() > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Use it to create figure instead of interactive
matplotlib.use('Agg')

# Detect anomaly
# torch.autograd.set_detect_anomaly(True)


def get_args():
    # Big Transfer arg parser
    parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
    parser.add_argument("--name",
                        default='debug',
                        # default='0517_mimic2_GAM_ga2m_s44_nl3_nt1333_td1_d2_od0.1_ld0.3_cs0.5_lr0.005_lo0_la0.0_pt0_pr0_mn0_ol0_ll1',
                        help="Name of this run. Used for monitoring and checkpointing.")
    parser.add_argument('--seed', type=int, default=None,
                        help='seed for initializing training.')
    # My own arguments
    parser.add_argument("--dataset", default='mimic2',
                        help="Choose the dataset.",
                        choices=['year', 'epsilon', 'a9a', 'higgs', 'microsoft',
                                 'yahoo', 'click', 'mimic2', 'adult', 'churn',
                                 'credit', 'compas', 'support2', 'mimic3',
                                 'rossmann', 'wine', 'bikeshare', 'sarcos', 'sarcos0',
                                 'sarcos1', 'sarcos2', 'sarcos3', 'sarcos4',
                                 'sarcos5', 'sarcos6'])
    parser.add_argument('--fold', type=int, default=0,
                        help='Choose from 0 to 4, as we only support 5-fold CV.')
    parser.add_argument("--arch", type=str, default='GAM',
                        choices=['ODST', 'GAM', 'GAMAtt', 'GAMAtt2','GAMAtt3'])
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
    parser.add_argument("--quantile_noise", type=float, default=None)
    parser.add_argument("--n_quantiles", type=int, default=2000)

    parser.add_argument("--early_stopping_rounds", type=int, default=11000)
    parser.add_argument("--max_rounds", type=int, default=-1)
    parser.add_argument("--max_time", type=float, default=3600 * 20) # At most 20 hours
    parser.add_argument("--report_frequency", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_bs", type=int, default=2048)
    parser.add_argument("--min_bs", type=int, default=128)

    parser.add_argument("--random_search", type=int, default=0)
    parser.add_argument('--fp16', type=int, default=1,
                        help='Slows down 5~10%. But saves memory and is slightly better')
    parser.add_argument('--data_subsample', type=float, default=1.,
                        help='Between 0 and 1. Percentage of training data used. '
                             'If bigger than 1, then select these '
                             'number of samples.')

    # slurm hparams. Used to run jobs in slurm
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--mem', type=int, default=8)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--partition', type=str, default=None)

    # Load hparams from a model name
    parser.add_argument("--load_from_hparams", type=str, default=None)

    # For old compatability purpose
    parser.add_argument("--save_memory", type=int, default=1)

    # Pretraining
    parser.add_argument('--pretrain', type=int, default=0, choices=[0, 1, 2, 3],
                        help='0: no pretrain. 1: pretrain with mask loss.'
                             '2: pretrain with MSE loss. 3: pretrain with MSE; mask outputs')
    parser.add_argument('--pretraining_ratio', type=float, default=0,
                        help='Between 0 and 1, percentage of feature to mask for reconstruction')
    parser.add_argument('--masks_noise', type=float, default=0,
                        help='Between 0 and 1, percentage of masks that do not mask inputs')
    parser.add_argument('--load_from_pretrain', type=str, default=None)
    parser.add_argument('--opt_only_last_layer', type=int, default=0)
    parser.add_argument('--finetune_lr', type=float, default=None, nargs='+')
    parser.add_argument('--finetune_freeze_steps', type=int, default=None, nargs='+')
    parser.add_argument('--finetune_data_subsample', type=float, default=None, nargs='+')
    parser.add_argument('--finetune_zip', type=int, default=0,
                        help='If set to 0, after pretraining it sends all combinations of '
                             'fds,flr and frs. If set to 1, then the length of fds, flr and frs '
                             'need to be the same, and only run the zip(fds, flr, frs). '
                             'Comes quite handy when running multiple folds.')
    parser.add_argument('--send_pt0', type=int, default=0,
                        help='If it is 1 and pretrain != 0, it sends another job with same hparams'
                             'but with pretrain=0. It serves as a baseline w/ same hparams.')
    parser.add_argument('--freeze_steps', type=int, default=0,
                        help='In finetuning, freeze the pretrained weights and only train'
                             'the last layer for these first steps.')
    parser.add_argument('--split_train_as_val', type=int, default=0,
                        help='For ODST 6 datasets, they have their own val set. '
                             'Set to 1 will split the train 20% as val set.')
    parser.add_argument('--ignore_prev_runs', type=int, default=0,
                        help='In random search, ignore previous runs that a param is already searched '
                             'and finished')

    temp_args, _ = parser.parse_known_args()
    if temp_args.name == 'debug':
        temp_args.arch = 'GAM'

    parser = getattr(lib.arch, temp_args.arch + 'Block').add_model_specific_args(parser)
    args = parser.parse_args()

    if args.name.startswith('debug'):
        print("WATCHOUT!!! YOU ARE RUNNING IN TEST RUN MODE!!!")
        if pexists('./logs/%s' % args.name):
            shutil.rmtree('./logs/%s' % args.name, ignore_errors=True)
        if pexists('./logs/hparams/%s' % args.name):
            os.remove('./logs/hparams/%s' % args.name)

    hparams = None
    if pexists(pjoin('logs', 'hparams', args.name)):
        hparams = json.load(open(pjoin('logs', 'hparams', args.name)))
    elif pexists(pjoin('logs', args.name, 'hparams.json')):
        hparams = json.load(open(pjoin('logs', args.name, 'hparams.json')))
    elif args.load_from_hparams is not None:
        hparams = json.load(open(pjoin('logs', 'hparams', args.load_from_hparams)))
    elif args.load_from_pretrain is not None:
        assert pexists(pjoin('logs', args.load_from_pretrain, 'MY_IS_FINISHED'))
        hparams = json.load(open(pjoin('logs', 'hparams', args.load_from_pretrain)))

    # Remove default value. Only parse user inputs
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    user_hparams = vars(parser.parse_args())

    if hparams is not None:
        cur_hparams = vars(args)

        # Update hparams from the previous hparams except user-specified one
        print('Reload and update from inputs: ' + str(user_hparams))
        cur_hparams.update({k: v for k, v in hparams.items() if k not in user_hparams})
        args = argparse.Namespace(**cur_hparams)

    # on v server
    if not pexists(pjoin('logs', args.name)) \
            and 'SLURM_JOB_ID' in os.environ \
            and pexists('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID']):
        if os.path.islink(pjoin('logs', args.name)):
            os.remove(pjoin('logs', args.name))
        os.symlink('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID'],
                   pjoin('logs', args.name))

    # Avoid I am too stupid....
    if args.load_from_pretrain is not None:
        args.pretrain = 0
    if args.arch.startswith('GAMAtt'):
        assert args.dim_att > 0

    return args, user_hparams


def main(args) -> None:
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
    qn = args.quantile_noise if getattr(args, 'quantile_noise', None) is not None \
        else data.get('quantile_noise', 1e-3)
    preprocessor = lib.MyPreprocessor(
        cat_features=data.get('cat_features', None),
        y_normalize=(data['problem'] == 'regression'),
        random_state=1337, quantile_transform=True,
        output_distribution=args.quantile_dist,
        quantile_noise=qn,
        n_quantiles=args.n_quantiles,
    )

    X_train, y_train = data['X_train'], data['y_train']
    preprocessor.fit(X_train, y_train)
    if args.data_subsample > 1.:
        args.data_subsample = int(args.data_subsample)

    # Do not subsample data in the pretraining!
    if args.pretrain == 0 and args.data_subsample != 1. \
            and args.data_subsample < X_train.shape[0]:
        print(f'Subsample the data by ds={args.data_subsample}')
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=args.data_subsample, random_state=1377,
            stratify=(y_train if data['problem'] == 'classification' else None))

    use_data_val = ('X_valid' in data and 'y_valid' in data
                    and (not args.split_train_as_val))
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

    # Modify based on if doing pretraining!
    if args.pretrain > 0:
        assert args.pretraining_ratio > 0.
        if args.pretrain == 1:
            args.problem = 'pretrain_mask'
        elif args.pretrain == 2:
            args.problem = 'pretrain_recon'
        elif args.pretrain == 3:
            args.problem = 'pretrain_recon2'
        else:
            raise NotImplementedError('Wrong pretrain: ' + str(args.pretrain))

        metric = 'pretrain_loss'
        args.num_classes = args.in_features
        args.data_addi_tree_dim = (-args.in_features) + 1
        # Use both train/val as training set, and use test as val
        X_train, X_valid = np.concatenate([X_train, X_valid], axis=0), X_test
        y_train, y_valid = X_train, X_valid

    print(f'X_train: {X_train.shape}, X_valid: {X_valid.shape}, X_test: {X_test.shape}')
    # Model
    model, step_callbacks = getattr(lib.arch, args.arch + 'Block').load_model_by_hparams(
        args, ret_step_callback=True)

    # Initialize bias before sending to cuda
    if 'init_bias' in args and args.init_bias and args.problem == 'classification':
        model.set_bias(y_train)

    model.to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # Load from pretrained model. Since last fc layer has diff size
    if getattr(args, 'load_from_pretrain', None) is not None:
        print("=> using pre-trained model '{}'".format(args.load_from_pretrain))
        path = pjoin('logs', args.load_from_pretrain, "checkpoint_best.pth")
        checkpoint = torch.load(path)

        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in checkpoint['model'].items()
                            if k in model_state and v.size() == model_state[k].size()}
        print('Pre-load the following weights:')
        print(list(pretrained_state.keys()))
        print('Ignore the following weights:')
        print([k for k in model_state if k not in pretrained_state])
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

    from qhoptim.pyt import QHAdam
    optimizer_params = {'nus': (0.7, 1.0), 'betas': (0.95, 0.998)}

    trainer = lib.Trainer(
        model=model,
        experiment_name=args.name,
        warm_start=True, # To handle the interruption on v server
        Optimizer=QHAdam,
        optimizer_params=optimizer_params,
        lr=args.lr,
        lr_warmup_steps=args.lr_warmup_steps,
        verbose=False,
        n_last_checkpoints=5,
        step_callbacks=step_callbacks, # Temp annelaing
        fp16=args.fp16,
        problem=args.problem,
        pretraining_ratio=args.pretraining_ratio,
        opt_only_last_layer=(args.load_from_pretrain is not None
                             and args.opt_only_last_layer),
        freeze_steps=(0 if args.load_from_pretrain is None else args.freeze_steps),
    )

    assert metric in ['negative_auc', 'classification_error', 'mse',
                      'multiple_mse', 'pretrain_loss']
    eval_fn = getattr(trainer, 'evaluate_' + metric)

    # Before we start, we will need to select the batch size if unspecified
    if args.batch_size is None or args.batch_size < 0:
        assert device != 'cpu', 'Have to specify batch size when using CPU'
        args.batch_size = choose_batch_size(trainer, X_train, y_train, device,
                                            max_bs=args.max_bs, min_bs=args.min_bs)
    else:
        try:
            with torch.no_grad():
                res = model(torch.as_tensor(X_train[:(2 * args.batch_size)], device=device))
            # trigger data-aware init
        except RuntimeError as e:
            handle_oom_error(e, args)

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

    ntf_diff, ntf = 0., None # Record number of trees assigned to each feature
    st_time = time.time()
    for batch in lib.iterate_minibatches(X_train, y_train,
                                         batch_size=args.batch_size,
                                         shuffle=True, epochs=float('inf')):
        # Handle removing missing by sampling from a Gaussian!
        try:
            metrics = trainer.train_on_batch(*batch, device=device)
        except RuntimeError as e:
            handle_oom_error(e, args)

        if recorder.loss_history is not None:
            recorder.loss_history.append(float(metrics['loss']))

        if trainer.step % args.report_frequency == 0:
            trainer.save_checkpoint()
            trainer.remove_old_temp_checkpoints()
            trainer.average_checkpoints(out_tag='avg')
            trainer.load_checkpoint(tag='avg')

            err = eval_fn(X_valid, y_valid,
                          device=device, batch_size=args.batch_size * 2)

            # Handle per-task early stopping when metric='multiple_mse'
            if metric == 'multiple_mse':
                # Initialize
                if not isinstance(recorder.best_err, list):
                    recorder.best_err = [float('inf') for _ in range(len(err))]
                    recorder.best_step_err = [0 for _ in range(len(err))]

                for idx, (e, be) in enumerate(zip(err, recorder.best_err)):
                    if e < be:
                        recorder.best_err[idx] = e
                        recorder.best_step_err[idx] = trainer.step
                        trainer.save_checkpoint(tag='best_t%d' % idx)
                if recorder.err_history is not None:
                    recorder.err_history.append(np.mean(err))

            else:
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
            if cur_ntf is None: # ODST no NTF
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

    if args.pretrain:
        # Submit another sbatch job for the real training
        print('***** FINISH pretraining! *****')
    else:
        max_step = trainer.step
        # Run test time
        if metric != 'multiple_mse':
            trainer.load_checkpoint(tag='best')
            test_err = eval_fn(X_test, y_test,
                               device=device, batch_size=2 * args.batch_size)
        else:
            test_err = []
            for idx in range(len(recorder.best_err)):
                trainer.load_checkpoint(tag='best_t%d' % idx)
                tmp = eval_fn(X_test, y_test,
                              device=device, batch_size=2 * args.batch_size)
                test_err.append(tmp[idx])

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
        results['finetuned'] = int(args.load_from_pretrain is not None)
        # Append the hyperparameters
        rs_hparams = getattr(lib.arch, args.arch + 'Block').get_model_specific_rs_hparams()
        for k in rs_hparams:
            results[k] = getattr(args, k)

        results = getattr(lib.arch, args.arch + 'Block').add_model_specific_results(results, args)
        results['name'] = args.name

        os.makedirs(f'results', exist_ok=True)
        dataset_postfix = f'_ds{args.data_subsample}' if args.data_subsample != 1. else ''
        if metric != 'multiple_mse':
            csv_file = f'results/{args.dataset}{dataset_postfix}_{args.arch}_new10.csv'
            lib.utils.output_csv(csv_file, results)
        else:
            csv_file = f'results/{args.dataset}{dataset_postfix}_{args.arch}_new10.ssv'
            lib.utils.output_csv(csv_file, results, delimiter=';')
        print('output results to %s' % csv_file)

    # Clean up
    open(pjoin('logs', args.name, 'MY_IS_FINISHED'), 'a')
    trainer.remove_old_temp_checkpoints(number_ckpts_to_keep=0)
    # recorder.clear()


def choose_batch_size(trainer, X_train, y_train, device,
                      max_bs=4096, min_bs=64):
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


def handle_oom_error(e, args, reduce_bs=True):
    if 'out of memory' not in str(e):
        raise e
    if 'SLURM_JOB_ID' not in os.environ:  # Not In SLURM
        print('Out of memory! But not in the slurm env.')
        raise e

    bs = args.batch_size
    if reduce_bs:
        bs = args.batch_size // 2
    if bs < args.min_bs:
        print('OOM but the current batch size %d can not be cut '
              'smaller than min batch size %d. Exit!'
              % (args.batch_size, args.min_bs))
        raise e

    print('OUT of memory! Cut bs in half and resend the job.')
    os.system(
        './my_sbatch --cpu {} --gpus {} --mem {} --name {} -p {} python -u main.py '
        '--batch_size {}'.format(
            args.cpu, args.gpus, args.mem, args.name,
            os.environ['SLURM_JOB_PARTITION'], bs))
    sys.exit()


if __name__ == '__main__':
    args, user_hparams = get_args()

    if args.random_search == 0:
        try:
            main(args)
        finally:
            if pexists(pjoin('is_running', args.name)): # release it
                os.remove(pjoin('is_running', args.name))

        def run_pretrain_cmd(args):
            if pexists(pjoin('logs', args.name, 'MY_IS_FINISHED')):
                print(f'{args.name} already finishes!')
                return
            main(args)

        if args.pretrain:
            orig_name = args.load_from_pretrain = args.name
            args.pretrain = 0

            flrs = args.finetune_lr if args.finetune_lr is not None else [args.lr]
            frss = args.finetune_freeze_steps \
                if args.finetune_freeze_steps is not None \
                else [args.freeze_steps]
            fdss = args.finetune_data_subsample \
                if args.finetune_data_subsample is not None \
                else [args.data_subsample]

            print(f'Sending lr={flrs} freeze_steps={frss} data_subsample={fdss}')
            if args.finetune_zip:
                assert len(flrs) == len(frss) == len(fdss), 'Lengths are not equal!'
                for flr, frs, fds in zip(flrs, frss, fdss):
                    args.lr = flr
                    args.freeze_steps = frs
                    args.data_subsample = fds
                    args.name = orig_name + f'_fds{fds}_flr{flr}_frs{frs}_ft'
                    run_pretrain_cmd(args)

            for flr in flrs:
                args.lr = flr
                for frs in frss:
                    args.freeze_steps = frs
                    for fds in fdss:
                        args.data_subsample = fds
                        # Change name coorespondingly
                        args.name = orig_name + f'_fds{fds}_flr{flr}_frs{frs}_ft'

                        run_pretrain_cmd(args)

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
    args.random_search = 0 # When sending jobs, not run the random search!!

    unsearched_set = {k for k in user_hparams if k in rs_hparams and k not in ['seed']}
    if len(unsearched_set) > 0:
        print('Do not random search following attributes:', unsearched_set)

    for r in range(num_random_search):
        for _ in range(50): # Try 50 times if can't found, quit
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

            main(args)
            break
        else:
            print('Can not find any more parameters! Quit.')
            sys.exit()
