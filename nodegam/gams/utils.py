"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import copy
import time

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from scipy import interpolate


def get_GAM_df_by_models(models, x_values_lookup=None, aggregate=True):
    models = iter(models)

    first_model = next(models)

    first_df = first_model.get_GAM_df(x_values_lookup)

    is_x_values_lookup_none = x_values_lookup is None
    if is_x_values_lookup_none:
        x_values_lookup = first_df[['feat_name', 'x']].set_index('feat_name').x.to_dict()
    
    all_dfs = [first_df]
    for model in models:
        the_df = model.get_GAM_df(x_values_lookup)
        all_dfs.append(the_df)
    
    if not aggregate:
        return all_dfs
    
    if len(all_dfs) == 1:
        return first_df

    all_ys = [np.concatenate(df.y) for df in all_dfs]

    split_pts = first_df.y.apply(lambda x: len(x)).cumsum()[:-1]
    first_df['y'] = np.split(np.mean(all_ys, axis=0), split_pts)
    first_df['y_std'] = np.split(np.std(all_ys, axis=0), split_pts)
    return first_df


def predict_score(model, X):
    result = predict_score_with_each_feature(model, X)
    return result.values.sum(axis=1)


def predict_score_by_df(GAM_plot_df, X):
    result = predict_score_with_each_feature_by_df(GAM_plot_df, X, sum_directly=True)
    return result


def predict_score_with_each_feature(model, X):
    x_values_lookup = get_x_values_lookup(X, model.feature_names)
    GAM_plot_df = model.get_GAM_df(x_values_lookup)
    return predict_score_with_each_feature_by_df(GAM_plot_df, X)


def predict_score_with_each_feature_by_df(GAM_plot_df, X, sum_directly=False):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=GAM_plot_df.feat_name.iloc[1:(X.shape[1]+1)].values.tolist())

    from tqdm import tqdm

    if sum_directly:
        scores = np.zeros((X.shape[0]))
    else:
        scores = np.empty((X.shape[0], GAM_plot_df.shape[0]))

    for f_idx, attrs in tqdm(GAM_plot_df.iterrows()):
        if attrs.feat_idx == -1:
            offset = attrs.y[0]
            if sum_directly:
                scores += offset
            else:
                scores[:, f_idx] = offset
            continue

        feat_idx = attrs.feat_idx if not isinstance(attrs.feat_idx, tuple) else list(attrs.feat_idx)
        truncated_X = X.iloc[:, feat_idx]
        if isinstance(attrs.feat_idx, tuple):
            score_lookup = pd.Series(attrs.y, index=attrs.x)
            truncated_X = pd.MultiIndex.from_frame(truncated_X) # list(truncated_X.itertuples(index=False, name=None))
        else:
            score_lookup = pd.Series(attrs.y, index=attrs.x)
            truncated_X = truncated_X.values

        if sum_directly:
            scores += score_lookup[truncated_X].values
        else:
            scores[:, (f_idx)] = score_lookup[truncated_X].values

    if sum_directly:
        return scores
    else:
        return pd.DataFrame(scores, columns=GAM_plot_df.feat_name.values.tolist())


def sigmoid(x):
    "Numerically stable sigmoid function."
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def get_X_values_counts(X, feature_names=None):
    if feature_names is None:
        feature_names = ['f%d' % i for i in range(X.shape[1])] \
            if isinstance(X, np.ndarray) else X.columns
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
        # return {'f%d' % idx: dict(zip(*np.unique(X[:, idx], return_counts=True))) for idx in range(X.shape[1])}
    result = X.apply(lambda x: x.value_counts().sort_index().to_dict(), axis=0)
    result.index = feature_names
    return result


def bin_data(X, max_n_bins=256):
    """Do a quantile binning for the X.

    Args:
        X: the pandas table or numpy array with shape as [N, D] where N is number of samples
            and D is number of features.
        max_n_bins: the maximum number of bins per feature. Default: 256.

    Returns:
        Binned X with the same input type (pandas table or numpy array).
    """
    X = X.copy()
    for col_name, dtype in zip(X.dtypes.index, X.dtypes):
        if is_string_dtype(dtype): # categorical
            continue

        col_data = X[col_name].astype(np.float32)

        uniq_vals = np.unique(col_data[~np.isnan(col_data)])
        if len(uniq_vals) > max_n_bins:
            print(f'bin features {col_name} with uniq val {len(uniq_vals)} to only {max_n_bins}')
            bins = np.unique(
                np.quantile(
                    col_data, q=np.linspace(0, 1, max_n_bins + 1),
                )
            )

            _, bin_edges = np.histogram(col_data, bins=bins)

            digitized = np.digitize(col_data, bin_edges, right=False)
            digitized[digitized == 0] = 1
            digitized -= 1

            # NOTE: NA handling done later.
            # digitized[np.isnan(col_data)] = self.missing_constant
            X.loc[:, col_name] = pd.Series(bins)[digitized].values.astype(np.float32)
    return X


def get_x_values_lookup(X, feature_names=None):
    """Get x values lookup.

    Args:
        X: input features. Numpy array or pandas dataframe.

    Returns:
        x_values_lookup: a dictionary with key as feature name and the value is all unique values
            of that feature.
    """
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = ['f%d' for idx in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    else:
        feature_names = X.columns

    return {
        feat_name : np.unique(X.iloc[:, feat_idx]).astype(X.dtypes[feat_idx])
        for feat_idx, feat_name in enumerate(feature_names)
    }

def my_interpolate(x, y, new_x):
    """Handle edge cases for interpolation."""
    assert len(x) == len(y)

    if len(x) == 1:
        y = np.full(len(new_x), y[0])
    else:
        f = interpolate.interp1d(x, y, fill_value='extrapolate', kind='nearest')
        y = f(new_x.astype(float))
    return y


class Timer:
    def __init__(self, name, remove_start_msg=True):
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v)

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo=memo))


def extract_GAM(X, predict_fn, predict_type='binary_logodds', max_n_bins=None):
    """
    X: input 2d array
    predict_fn: the model prediction function
    predict_type: choose from ["binary_logodds", "binary_prob", "regression"]
        This corresponds to which predict_fn to pass in.
    max_n_bins: default set as None (No binning). It bins the value into
        this number of buckets to reduce the resulting GAM graph clutterness.
        Should set large enough to not change prediction too much.
    """
    assert isinstance(X, pd.DataFrame)

    if max_n_bins is not None:
        X = bin_data(X, max_n_bins=max_n_bins)

    X_values_counts = get_X_values_counts(X)
    keys = list(X_values_counts.keys())

    # Use the X_values_counts to produce the Xs
    log_odds = {'offset': {'y_val': 0.}}
    for feat_name in keys:
        all_xs = list(X_values_counts[feat_name].keys())

        log_odds[feat_name] = {
            'x_val': np.array(all_xs),
            'y_val': np.zeros(len(all_xs), dtype=np.float32),
        }

    # Extract the GAM value from the model
    split_lens = [len(log_odds[f_name]['x_val']) for f_name in keys]
    cum_lens = np.cumsum(split_lens)

    first_record = X.iloc[0].values
    all_X = first_record.reshape((1, -1)).repeat(1 + np.sum(split_lens), axis=0)

    for f_idx, (feature_name, s_idx, e_idx) in enumerate(
            zip(keys, [0] + cum_lens[:-1].tolist(), cum_lens)):
        x = log_odds[feature_name]['x_val']

        all_X[(1 + s_idx):(1 + e_idx), f_idx] = x

    if predict_type in ['binary_logodds', 'regression']:
        score = predict_fn(all_X)
    elif predict_type == 'binary_prob':
        eps = 1e-8
        prob = predict_fn(all_X)

        prob = np.clip(prob, eps, 1. - eps)
        score = np.log(prob) - np.log(1. - prob)
    else:
        raise NotImplementedError(f'Unknoen {predict_type}')

    log_odds['offset']['y_val'] = score[0]
    score[1:] -= score[0]

    ys = np.split(score[1:], np.cumsum(split_lens[:-1]))
    for f_idx, feature_name in enumerate(keys):
        log_odds[feature_name]['y_val'] = ys[f_idx]

    # Centering and importances
    for feat_idx, feat_name in enumerate(keys):
        v = log_odds[feat_name]

        model_y_val = v['y_val']

        # Calculate importance
        weights = np.array(list(X_values_counts[feat_name].values()))
        weighted_mean = np.average(model_y_val, weights=weights)
        importance = np.average(np.abs(model_y_val - weighted_mean), weights=weights)
        log_odds[feat_name]['importance'] = importance
        log_odds[feat_name]['counts'] = weights.tolist()

        # Centering
        log_odds[feat_name]['y_val'] -= weighted_mean
        log_odds['offset']['y_val'] += weighted_mean

    results = [{
        'feat_name': 'offset',
        'feat_idx': -1,
        'x': None,
        'y': np.full(1, log_odds['offset']['y_val']),
        'importance': -1,
        'counts': [X.shape[0]],
    }]

    for feat_idx, feat_name in enumerate(X.columns):
        results.append({
            'feat_name': feat_name,
            'feat_idx': feat_idx,
            'x': log_odds[feat_idx]['x_val'],
            'y': np.array(log_odds[feat_idx]['y_val']),
            'importance': log_odds[feat_idx]['importance'],
            'counts': log_odds[feat_idx]['counts'],
        })

    return pd.DataFrame(results)
