"""All utilities including minibatches, files, seeds, model storages, and GAM extractions."""

import contextlib
import gc
import glob
import hashlib
import json
import os
import pickle
import random
import time
from logging import log
from os.path import join as pjoin, exists as pexists

import numpy as np
import pandas as pd
import torch

from .gams.utils import extract_GAM, bin_data


def iterate_minibatches(*tensors, batch_size, shuffle=True, epochs=1,
                        allow_incomplete=True, callback=lambda x:x):
    """Run the minibatches.

    Args:
        *tensors: the tensors to run minibatch.
        batch_size: the batch size.
        shuffle: if True, shuffle the tensors before each epoch starts.
        epochs: the number of epochs to iterate minibatches.
        allow_incomplete: if True, the last batch of each epoch can be smaller than the batch_size.
        callback: f(list of batch start idxes). Could be useful to change the batch start idxes.

    Example:
        >>> for x, y in iterate_minibatches(X, Y, batch_size=256, shuflle=True, epochs=10):
        >>>     train(x, y)
    """
    indices = np.arange(len(tensors[0]))
    upper_bound = int((np.ceil if allow_incomplete else np.floor) (len(indices) / batch_size)) * batch_size
    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in callback(range(0, upper_bound, batch_size)):
            batch_ix = indices[batch_start: batch_start + batch_size]
            batch = [tensor[batch_ix] if tensor is not None else None
                     for tensor in tensors]
            yield batch if len(tensors) > 1 else batch[0]
        epoch += 1
        if epoch >= epochs:
            break


def process_in_chunks(function, *args, batch_size, out=None, **kwargs):
    """Computes output by applying batch-parallel function to large data tensor in chunks.

    Args:
        function: a function(*[x[indices, ...] for x in args]) -> out[indices, ...].
        args: one or many tensors, each [num_instances, ...].
        batch_size: maximum chunk size processed in one go.
        out: memory buffer for out, defaults to torch.zeros of appropriate size and type.

    Returns:
        out: the outputs of function(data), computed in a memory-efficient (mini-batch) way.
    """
    total_size = args[0].shape[0]
    first_output = function(*[x[0: batch_size] for x in args])
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    if out is None:
        out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                          layout=first_output.layout, **kwargs)

    out[0: batch_size] = first_output
    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = function(*[x[batch_ix] for x in args])
    return out


def check_numpy(x):
    """Makes sure x is a numpy array. If not, make it as one."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    assert isinstance(x, np.ndarray)
    return x


def sigmoid_np(x):
    """A sigmoid function for numpy array.

    Args:
        x: numpy array.

    Returns:
        the sigmoid value.
    """
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


@contextlib.contextmanager
def nop_ctx():
    yield None


def get_latest_file(pattern):
    """Get the lattest files under the regex pattern.

    Args:
        pattern: the regex pattern. E.g. '*.csv'.
    """
    list_of_files = glob.glob(pattern) # * means all if need specific format then *.csv
    if len(list_of_files) == 0:
        return None
    return max(list_of_files, key=os.path.getctime)


def md5sum(fname):
    """Computes mdp checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def free_memory(sleep_time=0.1):
    """Black magic function to free torch memory and some jupyter whims."""
    gc.collect()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(sleep_time)

def to_float_str(element):
    try:
        return str(float(element))
    except ValueError:
        return element


def seed_everything(seed=None) -> int:
    """Seed everything.

    It includes pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable. Borrow
    it from the pytorch_lightning.

    Args:
        seed: the seed. If None, it generates one.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    seed = random.randint(min_seed_value, max_seed_value)
    print(f"No correct seed found, seed set to {seed}")
    return seed


def output_csv(the_path, data_dict, order=None, delimiter=','):
    """Output a csv file from a python dictionary.

    If the csv file exists, it outputs another row under this csv file.

    Args:
        the_path: the filename of the csv file.
        data_dict: the data dictionary.
        order: if specified, the columns of the csv follow the specified order. Default: None.
        delimiter: the seperated delimiter. Defulat: ','.
    """
    if the_path.endswith('.tsv'):
        delimiter = '\t'

    is_file_exists = os.path.exists(the_path)
    with open(the_path, 'a+') as op:
        keys = list(data_dict.keys())
        if order is not None:
            keys = order + [k for k in keys if k not in order]

        col_title = delimiter.join([str(k) for k in keys])
        if not is_file_exists:
            print(col_title, file=op)
        else:
            old_col_title = open(the_path, 'r').readline().strip()
            if col_title != old_col_title:
                old_order = old_col_title.split(delimiter)

                no_key = [k for k in old_order if k not in keys]
                if len(no_key) > 0:
                    print('The data_dict does not have the '
                          'following old keys: %s' % str(no_key))

                additional_keys = [k for k in keys if k not in old_order]
                if len(additional_keys) > 0:
                    print('WARNING! The data_dict has following additional '
                          'keys %s.' % (str(additional_keys)))
                    col_title = delimiter.join([
                        str(k) for k in old_order + additional_keys])
                    print(col_title, file=op)

                keys = old_order + additional_keys

        vals = []
        for k in keys:
            val = data_dict.get(k, -999)
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                val = val.item()
            vals.append(str(val))

        print(delimiter.join(vals), file=op)


class Timer:
    def __init__(self, name, remove_start_msg=True):
        """A simple timer.

        Args:
            name: the name of the timer.
            remove_start_msg: if True, it will remove the start message of running.

        Usage:
            >>> with Timer('model training'):
            >>>     train()
            Run model training.........
            Finish model training in 1.3s
        """
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))


def load_best_model_from_trained_dir(the_dir):
    """Load the best NodeGAM model from a trained directory.

    Follow the filenames of checkpoints in 'main.py'.

    Args:
        the_dir: the saved direcotry.

    Returns:
        model: a pytorch NodeGAM model.
    """
    hparams = load_hparams(the_dir)

    from . import arch
    model, step_callbacks = getattr(arch, hparams['arch'] + 'Block').load_model_by_hparams(
        hparams, ret_step_callback=True)

    # Set the step!
    if step_callbacks is not None and len(step_callbacks) > 0:
        bstep = json.load(open(pjoin(the_dir, 'recorder.json')))['best_step_err']
        for c in step_callbacks:
            c(bstep)

    best_ckpt = pjoin(the_dir, 'checkpoint_{}.pth'.format('best'))
    if not pexists(best_ckpt):
        print('NO BEST CHECKPT EXISTS in {}!'.format(best_ckpt))
        return None

    tmp = torch.load(best_ckpt, map_location='cpu')
    model.load_state_dict(tmp['model'])
    model.train(False)
    return model


def extract_GAM_from_saved_dir(saved_dir, max_n_bins=256, **kwargs):
    """Extract the GAM dataframe from a saved model directory (either NodeGAM or EBM or Spline).

    Args:
        saved_dir: the saved directory.
        max_n_bins: max number of bins for each feature when extracting.
        kwargs: additional arguments passed into NodeGAM.extract_additive_terms().

    Returns:
        df: a GAM dataframe.
    """
    if not pexists(saved_dir) or not pexists(pjoin(saved_dir, 'hparams.json')):
        with Timer('copying from v'):
            cmd = 'rsync -avzL v:/h/kingsley/node/%s/ %s' % (
                saved_dir, saved_dir)
            print(cmd)
            os.system(cmd)

    assert pexists(saved_dir), 'Either path is wrong or copy fails'

    hparams = load_hparams(saved_dir)

    if 'num_trees' in hparams: # NODE model
        df = extract_GAM_from_NODE(saved_dir, max_n_bins, **kwargs)
        # Wierd bug that sometimes the feat_idx gets np.int64 instead of int causing bugs
        df.feat_idx = df.feat_idx.apply(lambda x: int(x) if (type(x) is np.int64) else x)
        return df

    return extract_GAM_from_baselines(saved_dir, max_n_bins, **kwargs)


def extract_GAM_from_NODE(saved_dir, max_n_bins=256, way='blackbox', cache=False, **kwargs):
    """Extract the GAM dataframe from the NodeGAM model.

    Args:
        saved_dir: the saved directory of the NodeGAM.
        max_n_bins: max number of bins of each feature.
        way: choice from ['blackbox', 'mine']. 'blackbox' treats the model as a blackbox to extract
            a GAM dataframe. 'mine' can only be applied to NodeGAM that uses the internal knowledge
            of NodeGAM to extract the GAM/GA2M dataframe.
        cache: if True, it stores 'df_cache_bins{max_n_bins}.pkl' under the saved_dir.
        kwargs: the additional arguments when calling model.extract_additive_terms().
    
    Returns:
        df: the GAM dataframe.
    """
    if max_n_bins is None:
        max_n_bins = -1

    cache_path = pjoin(saved_dir, f'df_cache_bins{max_n_bins}.pkl')
    if pexists(cache_path):
        with open(cache_path, 'rb') as fp, Timer(f'load cache: {cache_path}'):
            return pickle.load(fp)
    from .data import DATASETS

    hparams = json.load(open(pjoin(saved_dir, 'hparams.json')))

    assert pexists(pjoin(saved_dir, 'checkpoint_best.pth')), 'No best ckpt exists!'
    model = load_best_model_from_trained_dir(saved_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    pp = pickle.load(open(pjoin(saved_dir, 'preprocessor.pkl'), 'rb'))

    dataset = DATASETS[hparams['dataset'].upper()](path='./data/')
    all_X = pd.concat([dataset['X_train'], dataset['X_test']], axis=0)

    if max_n_bins is not None and max_n_bins > 0:
        all_X = bin_data(all_X, max_n_bins=max_n_bins)

    # If it's a GA2M, can only use 'mine' method to extract
    if hparams.get('ga2m', 0):
        way = 'mine'

    if way == 'blackbox':
        def predict_fn(X):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=all_X.columns)

            X = pp.transform(X)
            X = torch.as_tensor(X, device=device)
            with torch.no_grad():
                logits = process_in_chunks(model, X, batch_size=2*hparams['batch_size'])
                logits = check_numpy(logits)

            ret = logits
            if len(logits.shape) == 2 and logits.shape[1] == 2:
                ret = logits[:, 1] - logits[:, 0]
            elif len(logits.shape) == 1: # regression or binary cls
                if pp.y_mu is not None and pp.y_std is not None:
                    ret = (ret * pp.y_std) + pp.y_mu
            return ret

        df = extract_GAM(all_X, predict_fn)
    elif way == 'mine':
        df = model.extract_additive_terms(all_X, pp, device=device,
                                          batch_size=2*hparams['batch_size'],
                                          **kwargs)
    else:
        raise NotImplementedError('No such way: ' + way)

    if cache:
        with Timer('Dump the dataframe to cache'):
            with open(cache_path, 'wb') as op:
                pickle.dump(df, op)
    return df


def make_predictions(model_name, X):
    """Make predictions of some model.

    Args:
        model_name: the model name. It's saved under logs/{model_name}/.
        X (pandas dataframe): the input data.

    Returns:
        ret (numpy array): the prediction on X.
    """
    saved_dir = pjoin('logs', model_name)
    hparams = json.load(open(pjoin(saved_dir, 'hparams.json')))

    assert pexists(pjoin(saved_dir, 'checkpoint_best.pth')), 'No best ckpt exists!'
    model = load_best_model_from_trained_dir(saved_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    pp = pickle.load(open(pjoin(saved_dir, 'preprocessor.pkl'), 'rb'))

    X = pp.transform(X)
    X = torch.as_tensor(X, device=device)
    with torch.no_grad():
        logits = process_in_chunks(model, X, batch_size=2 * hparams['batch_size'])
        logits = check_numpy(logits)

    ret = logits
    if len(logits.shape) == 2 and logits.shape[1] == 2:
        ret = logits[:, 1] - logits[:, 0]
    elif len(logits.shape) == 1:  # regression or binary cls
        if pp.y_mu is not None and pp.y_std is not None:
            ret = (ret * pp.y_std) + pp.y_mu
    return ret

def extract_GAM_from_baselines(saved_dir, max_n_bins=256, **kwargs):
    """Extract the dataframe from other GAM baselines like EBM and Spline.

    Args:
        saved_dir: the saved model's directory.
        max_n_bins: the max number of bins for each feature.

    Returns:
        df: the GAM dataframe.
    """
    from .data import DATASETS
    model = pickle.load(open(pjoin(saved_dir, 'model.pkl'), 'rb'))

    hparams = load_hparams(saved_dir)

    pp = None
    if pexists(pjoin(saved_dir, 'preprocessor.pkl')):
        pp = pickle.load(open(pjoin(saved_dir, 'preprocessor.pkl'), 'rb'))

    dataset = DATASETS[hparams['dataset'].upper()](path='./data/')
    all_X = pd.concat([dataset['X_train'], dataset['X_test']], axis=0)

    def predict_fn(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=all_X.columns)

        if pp is not None:
            X = pp.transform(X)

        if dataset['problem'] == 'classification':
            prob = model.predict_proba(X)
            return prob[:, 1]
        return model.predict(X)

    predict_type = 'binary_prob' \
        if dataset['problem'] == 'classification' else 'regression'
    df = extract_GAM(all_X, predict_fn, max_n_bins=max_n_bins, predict_type=predict_type)
    return df


def load_hparams(the_dir):
    """Load the hyperparameters (hparams) from a directory."""
    if pexists(pjoin(the_dir, 'hparams.json')):
        hparams = json.load(open(pjoin(the_dir, 'hparams.json')))
    else:
        name = os.path.basename(the_dir)
        if not pexists(pjoin('logs', 'hparams', name)):
            cmd = 'rsync -avzL v:/h/kingsley/node/logs/hparams/%s ./logs/hparams/' % (name)
            print(cmd)
            os.system(cmd)

        if pexists(pjoin('logs', 'hparams', name)):
            hparams = json.load(open(pjoin('logs', 'hparams', name)))
        else:
            raise RuntimeError('No hparams exist: %s' % the_dir)
    return hparams


def average_GAMs(gam_dirs, **kwargs):
    """Take average of GAM models to derive mean and stdev from their model names.

    Args:
        gam_dirs: a list of model name. E.g. ['0603_bikeshare']. The model has to be stored under
            "logs/{name}".

    Returns:
        df: the averaged dataframe with mean, stdev and the importance.
    """
    all_dfs = [extract_GAM_from_saved_dir(pjoin('logs', d), **kwargs) for d in gam_dirs]

    df = average_GAM_dfs(all_dfs)
    return df


def average_GAM_dfs(all_dfs):
    """Take average of GAM dataframes to derive mean and stdev for each term.

    Args:
        all_dfs: a list of dataframes.

    Returns:
        df: the averaged dataframe with mean, stdev and the importance.
    """
    first_df = all_dfs[0]
    if len(all_dfs) == 1:
        return first_df

    all_feat_idx = first_df.feat_idx.values.tolist()
    for i in range(1, len(all_dfs)):
        all_feat_idx += all_dfs[i].feat_idx.values.tolist()
    all_feat_idx = set(all_feat_idx)

    results = []
    for feat_idx in all_feat_idx:
        # print(feat_idx)
        # all_dfs_with_this_feat_idx = []
        all_dfs_with_this_feat_idx = [
            df[df.feat_idx == feat_idx].iloc[0] for df in all_dfs
            if np.any(df.feat_idx == feat_idx)
        ]

        all_ys = [df.y for df in all_dfs_with_this_feat_idx]
        if len(all_ys) == 0:
            import pdb; pdb.set_trace()

        if len(all_ys) < len(all_dfs):  # Not every df has the index
            diff = len(all_dfs) - len(all_ys)
            # print(f'Add {diff} times 0 arr in {feat_idx}')
            for _ in range(diff):
                all_ys.append(np.zeros(len(all_ys[0])).tolist())

        y_mean = np.mean(all_ys, axis=0)
        y_std = np.std(all_ys, axis=0)

        row = all_dfs_with_this_feat_idx[0]
        result = {
            'feat_name': row.feat_name,
            'feat_idx': row.feat_idx,
            'x': row.x,
            'y': y_mean,
            'y_std': y_std,
        }
        if 'counts' in row:
            result['counts'] = row.counts
            result['importance'] = np.average(np.abs(y_mean), weights=row.counts)

        results.append(result)

    df = pd.DataFrame(results)
    # sort it
    df['tmp'] = df.feat_idx.apply(
        lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x, tuple) else int(x))
    df = df.sort_values('tmp').drop('tmp', axis=1)
    df = df.reset_index(drop=True)
    return df


def get_gpu_stat(pitem: str, device_id=0):
    """Get the GPU stats.

    Borrow from pytorch lightning:
    https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.9.0/pytorch_lightning/callbacks/gpu_usage_logger.py#L30-L166

    Args:
        pitem: the gpu partition.
        device_id: the device id of gpu.

    Returns:
        gpu_usage: the GPU memory consumption.
    """
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={pitem}", "--format=csv,nounits,noheader"],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )

    gpu_usage = [float(x) for x in result.stdout.strip().split(os.linesep)]
    return gpu_usage[device_id]
