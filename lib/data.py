import os
from os.path import join as pjoin, exists as pexists
import bz2
import numpy as np
import pandas as pd
import gzip
import shutil
import torch
import random
import warnings

from sklearn.model_selection import train_test_split

from .utils import download, Timer
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import LeaveOneOutEncoder
from zipfile import ZipFile
import requests


class MyPreprocessor:

    def __init__(self, random_state=1377, cat_features=None, normalize=False,
                 y_normalize=False, quantile_transform=False,
                 output_distribution='normal', n_quantiles=2000,
                 quantile_noise=0, **kwargs):
        """
        Preprocessor is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATSETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param random_state: global random seed for an experiment
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param quantile_transform: transforms the features to follow a normal distribution.
        :param output_distribution: if quantile_transform == True, data is projected onto this distribution
            See the same param of sklearn QuantileTransformer
        :param quantile_noise: if specified, fits QuantileTransformer on data with added gaussian noise
            with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable
            Please not that this transformation does NOT apply gaussian noise to the resulting data,
            the noise is only applied for QuantileTransformer
        :param kwargs: depending on the dataset, you may select train size, test size or other params
            If dataset is not in DATASETS, provide six keys: X_train, y_train, X_valid, y_valid, X_test and y_test
        """
        self.random_state = random_state
        self.cat_features = cat_features
        self.normalize = normalize
        self.y_normalize = y_normalize
        self.quantile_transform = quantile_transform
        self.output_distribution = output_distribution
        self.quantile_noise = quantile_noise
        self.n_quantiles = n_quantiles

        self.transformers = []
        self.y_mu, self.y_std = None, None
        self.feature_names = None

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), 'X is not a dataframe! %s' % type(X)
        self.feature_names = X.columns

        if self.cat_features is not None:
            cat_encoder = LeaveOneOutEncoder(cols=self.cat_features)
            cat_encoder.fit(X, y)
            self.transformers.append(cat_encoder)

        if self.normalize:
            scaler = StandardScaler(copy=False)
            scaler.fit(X)
            self.transformers.append(scaler)

        if self.quantile_transform:
            quantile_train = X.copy()
            if self.cat_features is not None:
                quantile_train = cat_encoder.transform(quantile_train)

            if self.quantile_noise:
                r = np.random.RandomState(self.random_state)
                stds = np.std(quantile_train.values, axis=0, keepdims=True)
                noise_std = self.quantile_noise / np.maximum(stds, self.quantile_noise)
                quantile_train += noise_std * r.randn(*quantile_train.shape)

            qt = QuantileTransformer(random_state=self.random_state,
                                     n_quantiles=self.n_quantiles,
                                     output_distribution=self.output_distribution,
                                     copy=False)
            # if self.cat_features is not None:
            #     conti_fs = [f for f in self.feature_names if f not in self.cat_features]
            #     qt = ColumnTransformer(transformers=[("quantile", qt, conti_fs)],
            #                            remainder='passthrough')
            qt.fit(quantile_train)
            self.transformers.append(qt)

        if y is not None and self.y_normalize:
            self.y_mu, self.y_std = y.mean(axis=0), y.std(axis=0)
            print("Normalize y. mean = {}, std = {}".format(self.y_mu, self.y_std))

    def transform(self, *args):
        assert len(args) <= 2

        X = args[0]
        if len(self.transformers) > 0:
            X = X.copy()
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)

            for i, t in enumerate(self.transformers):
                # Leave one out transform when it's training set
                X = t.transform(X)
            # The LeaveOneOutEncoder makes it as np.float64 instead of 32
            X = X.astype(np.float32)

        if len(args) == 1:
            return X

        y = args[1]
        if y is None:
            return X, None

        if self.y_normalize and self.y_mu is not None and self.y_std is not None:
            y = (y - self.y_mu) / self.y_std

        return X, y


def download_file_from_google_drive(id, destination):
    '''
    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    '''
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def fetch_A9A(path='./data/', train_size=None, valid_size=None, test_size=None, fold=0):
    path = pjoin(path, 'A9A')

    train_path = pjoin(path, 'a9a')
    test_path = pjoin(path, 'a9a.t')
    if not all(pexists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/9cqdx166iwonrj9/a9a?dl=1", train_path)
        download("https://www.dropbox.com/s/sa0ds895c0v4xc6/a9a.t?dl=1", test_path)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=123)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=123)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = pjoin(path, 'stratified_train_idx.txt')
        valid_idx_path = pjoin(path, 'stratified_valid_idx.txt')
        if not all(pexists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/xy4wwvutwikmtha/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/nthpxofymrais5s/stratified_test_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
        problem='classification',
    )


def fetch_EPSILON(path='./data/', train_size=None, valid_size=None, test_size=None, fold=0):
    path = pjoin(path, 'EPSILON')

    train_path = pjoin(path, 'epsilon_normalized')
    test_path = pjoin(path, 'epsilon_normalized.t')
    if not all(pexists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        train_archive_path = pjoin(path, 'epsilon_normalized.bz2')
        test_archive_path = pjoin(path, 'epsilon_normalized.t.bz2')
        if not all(pexists(fname) for fname in (train_archive_path, test_archive_path)):
            download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2", train_archive_path)
            download("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2", test_archive_path)
        print("unpacking dataset")
        for file_name, archive_name in zip((train_path, test_path), (train_archive_path, test_archive_path)):
            zipfile = bz2.BZ2File(archive_name)
            with open(file_name, 'wb') as f:
                f.write(zipfile.read())

    with Timer("reading dataset (it may take a long time)"):
        X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=2000)
        X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=2000)
        X_train, X_test = X_train.toarray(), X_test.toarray()
        y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)
        y_train[y_train == -1] = 0
        y_test[y_test == -1] = 0

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = pjoin(path, 'stratified_train_idx.txt')
        valid_idx_path = pjoin(path, 'stratified_valid_idx.txt')
        if not all(pexists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/wxgm94gvm6d3xn5/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/fm4llo5uucdglti/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    return dict(
        X_train=X_train.iloc[train_idx], y_train=y_train[train_idx],
        X_valid=X_train.iloc[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
        problem='classification',
    )


def fetch_PROTEIN(path='./data/', train_size=None, valid_size=None, test_size=None, fold=0):
    """
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#protein
    """
    path = pjoin(path, 'PROTEIN')

    train_path = pjoin(path, 'protein')
    test_path = pjoin(path, 'protein.t')
    if not all(pexists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/pflp4vftdj3qzbj/protein.tr?dl=1", train_path)
        download("https://www.dropbox.com/s/z7i5n0xdcw57weh/protein.t?dl=1", test_path)
    for fname in (train_path, test_path):
        raw = open(fname).read().replace(' .', '0.')
        with open(fname, 'w') as f:
            f.write(raw)

    X_train, y_train = load_svmlight_file(train_path, dtype=np.float32, n_features=357)
    X_test, y_test = load_svmlight_file(test_path, dtype=np.float32, n_features=357)
    X_train, X_test = X_train.toarray(), X_test.toarray()
    y_train, y_test = y_train.astype(np.int), y_test.astype(np.int)

    if all(sizes is None for sizes in (train_size, valid_size, test_size)):
        train_idx_path = pjoin(path, 'stratified_train_idx.txt')
        valid_idx_path = pjoin(path, 'stratified_valid_idx.txt')
        if not all(pexists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/wq2v9hl1wxfufs3/small_stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/7o9el8pp1bvyy22/small_stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)
        if test_size is not None:
            warnings.warn('Test set is fixed for this dataset.', Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    return dict(
        X_train=X_train.iloc[train_idx], y_train=y_train[train_idx],
        X_valid=X_train.iloc[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test
    )


def fetch_YEAR(path='./data/', train_size=None, valid_size=None, test_size=51630, fold=0):
    path = pjoin(path, 'YEAR')

    data_path = pjoin(path, 'data.csv')
    if not pexists(data_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/l09pug0ywaqsy0e/YearPredictionMSD.txt?dl=1', data_path)
    n_features = 91
    types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    if all(sizes is None for sizes in (train_size, valid_size)):
        train_idx_path = pjoin(path, 'stratified_train_idx.txt')
        valid_idx_path = pjoin(path, 'stratified_valid_idx.txt')
        if not all(pexists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/00u6cnj9mthvzj1/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/420uhjvjab1bt7k/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return dict(
        X_train=X_train.iloc[train_idx], y_train=y_train[train_idx].astype(np.float32),
        X_valid=X_train.iloc[valid_idx], y_valid=y_train[valid_idx].astype(np.float32),
        X_test=X_test, y_test=y_test.astype(np.float32),
        problem='regression',
    )


def fetch_HIGGS(path='./data/', train_size=None, valid_size=None, test_size=5 * 10 ** 5, fold=0):
    path = pjoin(path, 'HIGGS')

    data_path = pjoin(path, 'higgs.csv')
    if not pexists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = pjoin(path, 'HIGGS.csv.gz')
        download('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', archive_path)
        with gzip.open(archive_path, 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    n_features = 29
    types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    if all(sizes is None for sizes in (train_size, valid_size)):
        train_idx_path = pjoin(path, 'stratified_train_idx.txt')
        valid_idx_path = pjoin(path, 'stratified_valid_idx.txt')
        if not all(pexists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/i2uekmwqnp9r4ix/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/wkbk74orytmb2su/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return dict(
        X_train=X_train.iloc[train_idx], y_train=y_train[train_idx],
        X_valid=X_train.iloc[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
        problem='classification',
    )


def fetch_MICROSOFT(path='./data/', fold=0):
    path = pjoin(path, 'MICROSOFT')

    train_path = pjoin(path, 'msrank_train.tsv')
    test_path = pjoin(path, 'msrank_test.tsv')
    if not all(pexists(fname) for fname in (train_path, test_path)):
        os.makedirs(path, exist_ok=True)
        download("https://www.dropbox.com/s/izpty5feug57kqn/msrank_train.tsv?dl=1", train_path)
        download("https://www.dropbox.com/s/tlsmm9a6krv0215/msrank_test.tsv?dl=1", test_path)

        for fname in (train_path, test_path):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

    data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
    data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

    train_idx_path = pjoin(path, 'train_idx.txt')
    valid_idx_path = pjoin(path, 'valid_idx.txt')
    if not all(pexists(fname) for fname in (train_idx_path, valid_idx_path)):
        download("https://www.dropbox.com/s/pba6dyibyogep46/train_idx.txt?dl=1", train_idx_path)
        download("https://www.dropbox.com/s/yednqu9edgdd2l1/valid_idx.txt?dl=1", valid_idx_path)
    train_idx = pd.read_csv(train_idx_path, header=None)[0].values
    valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values

    X_train, y_train, query_train = data_train.iloc[train_idx, 2:], data_train.iloc[train_idx, 0].values, data_train.iloc[train_idx, 1]
    X_valid, y_valid, query_valid = data_train.iloc[valid_idx, 2:], data_train.iloc[valid_idx, 0].values, data_train.iloc[valid_idx, 1]
    X_test, y_test, query_test = data_test.iloc[:, 2:], data_test.iloc[:, 0].values, data_test.iloc[:, 1]

    return dict(
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.float32), query_train=query_train,
        X_valid=X_valid.astype(np.float32), y_valid=y_valid.astype(np.float32), query_valid=query_valid,
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.float32), query_test=query_test,
        problem='regression',
    )


def fetch_YAHOO(path='./data/', fold=0):
    path = pjoin(path, 'YAHOO')

    train_path = pjoin(path, 'yahoo_train.tsv')
    valid_path = pjoin(path, 'yahoo_valid.tsv')
    test_path = pjoin(path, 'yahoo_test.tsv')
    if not all(pexists(fname) for fname in (train_path, valid_path, test_path)):
        os.makedirs(path, exist_ok=True)
        train_archive_path = pjoin(path, 'yahoo_train.tsv.gz')
        valid_archive_path = pjoin(path, 'yahoo_valid.tsv.gz')
        test_archive_path = pjoin(path, 'yahoo_test.tsv.gz')
        if not all(pexists(fname) for fname in (train_archive_path, valid_archive_path, test_archive_path)):
            download("https://www.dropbox.com/s/7rq3ki5vtxm6gzx/yahoo_set_1_train.gz?dl=1", train_archive_path)
            download("https://www.dropbox.com/s/3ai8rxm1v0l5sd1/yahoo_set_1_validation.gz?dl=1", valid_archive_path)
            download("https://www.dropbox.com/s/3d7tdfb1an0b6i4/yahoo_set_1_test.gz?dl=1", test_archive_path)

        for file_name, archive_name in zip((train_path, valid_path, test_path), (train_archive_path, valid_archive_path, test_archive_path)):
            with gzip.open(archive_name, 'rb') as f_in:
                with open(file_name, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        for fname in (train_path, valid_path, test_path):
            raw = open(fname).read().replace('\\t', '\t')
            with open(fname, 'w') as f:
                f.write(raw)

    data_train = pd.read_csv(train_path, header=None, skiprows=1, sep='\t')
    data_valid = pd.read_csv(valid_path, header=None, skiprows=1, sep='\t')
    data_test = pd.read_csv(test_path, header=None, skiprows=1, sep='\t')

    X_train, y_train, query_train = data_train.iloc[:, 2:], data_train.iloc[:, 0].values, data_train.iloc[:, 1]
    X_valid, y_valid, query_valid = data_valid.iloc[:, 2:], data_valid.iloc[:, 0].values, data_valid.iloc[:, 1]
    X_test, y_test, query_test = data_test.iloc[:, 2:], data_test.iloc[:, 0].values, data_test.iloc[:, 1]

    return dict(
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.float32), query_train=query_train,
        X_valid=X_valid.astype(np.float32), y_valid=y_valid.astype(np.float32), query_valid=query_valid,
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.float32), query_test=query_test,
        problem='regression',
    )


def fetch_CLICK(path='./data/', valid_size=100_000, validation_seed=None, fold=0):
    # based on: https://www.kaggle.com/slamnz/primer-airlines-delay
    path = pjoin(path, 'CLICK')

    csv_path = pjoin(path, 'click.csv')
    if not pexists(csv_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', csv_path)

    data = pd.read_csv(csv_path, index_col=0)
    X, y = data.drop(columns=['target']), data['target']
    X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
    y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

    y_train = (y_train.values.reshape(-1) == 1).astype('int64')
    y_test = (y_test.values.reshape(-1) == 1).astype('int64')

    cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                    'keyword_id', 'title_id', 'description_id', 'user_id']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=validation_seed)

    return dict(
        X_train=X_train, y_train=y_train,
        X_valid=X_val, y_valid=y_val,
        X_test=X_test, y_test=y_test,
        problem='classification',
        cat_features=cat_features,
        # transformers=[cat_encoder],
    )


def fetch_MIMIC2(path='./data/', fold=0):
    '''
    '''
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' \
                                 % fold

    data_path = pjoin(path, 'mimic2', 'mimic2.data')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'mimic2'), exist_ok=True)
        download('https://docs.google.com/uc?id=1pmF0HF7LPuxzXqnJhoiiS7ll_e1dOsbb',
                 pjoin(path, 'mimic2.zip'))
        with ZipFile(pjoin(path, 'mimic2.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'mimic2.zip'))

    cols = ['Age', 'GCS', 'SBP', 'HR', 'Temperature',
            'PFratio', 'Renal', 'Urea', 'WBC', 'CO2', 'Na', 'K',
            'Bilirubin', 'AdmissionType', 'AIDS',
            'MetastaticCancer', 'Lymphoma', 'HospitalMortality']

    table = pd.read_csv(data_path, delimiter=' ', header=None)
    table.columns = cols

    X_df = table.iloc[:, :-1]
    y_df = table.iloc[:, -1].values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'mimic2', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'mimic2', 'test%d.txt' % fold),
                           header=None)[0].values

    cat_features = ['GCS', 'Temperature', 'AdmissionType', 'AIDS',
                    'MetastaticCancer', 'Lymphoma', 'Renal']
    for c in cat_features:
        X_df[c] = X_df[c].astype(object)

    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='classification',
        cat_features=cat_features,
        metric='negative_auc',
        quantile_noise=1e-6,
    )


def fetch_ADULT(path='./data/', fold=0):
    '''
    '''
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' \
                                 % fold
    data_path = pjoin(path, 'adult', 'adult.data')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'adult'), exist_ok=True)
        download('https://docs.google.com/uc?id=1kzx-ckH1bzTByINTGjtFPNnU5Dkb2qgx',
                 pjoin(path, 'adult.zip'))
        with ZipFile(pjoin(path, 'adult.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'adult.zip'))

    cols = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    table = pd.read_csv(data_path, header=None)
    table.columns = cols

    X_df = table.iloc[:, :-1]

    y_df = table.iloc[:, -1]
    # Make it as 0 or 1
    y_df.loc[y_df == ' >50K'] = 1.
    y_df.loc[y_df == ' <=50K'] = 0.
    y_df = y_df.values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'adult', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'adult', 'test%d.txt' % fold),
                           header=None)[0].values

    cat_features = X_df.columns[X_df.dtypes == object]

    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='classification',
        cat_features=cat_features,
        metric='negative_auc',
        quantile_noise=1e-3,
    )


def fetch_COMPAS(path='./data/', fold=0):
    '''
    '''
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' \
                                 % fold
    data_path = pjoin(path, 'recid', 'recid.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'recid'), exist_ok=True)
        download('https://docs.google.com/uc?id=1jqkVhzHZDPqDUYwjb7v3zIVoKQNvZZgD',
                 pjoin(path, 'recid.zip'))
        with ZipFile(pjoin(path, 'recid.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'recid.zip'))

    df = pd.read_csv(data_path, delimiter=',')
    target_variables = ['two_year_recid']

    X_df = df.drop(target_variables, axis=1)
    y_df = df[target_variables[0]].values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'recid', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'recid', 'test%d.txt' % fold),
                           header=None)[0].values

    cat_features = X_df.columns[X_df.dtypes == object]

    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='classification',
        cat_features=cat_features,
        metric='negative_auc',
        quantile_noise=1e-5,
    )


def fetch_CHURN(path='./data/', fold=0):
    '''
    '''
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' \
                                 % fold
    data_path = pjoin(path, 'churn', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'churn'), exist_ok=True)
        download('https://docs.google.com/uc?id=1eZnuF2NZ4RgfMEXWuqfgp_Q1UlVQfO20',
                 pjoin(path, 'churn.zip'))
        with ZipFile(pjoin(path, 'churn.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'churn.zip'))

    df = pd.read_csv(data_path)

    X_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]

    # Handle special case of TotalCharges wronly assinged as object
    X_df['TotalCharges'][X_df['TotalCharges'] == ' '] = 0.
    X_df.loc[:, 'TotalCharges'] = pd.to_numeric(X_df['TotalCharges'])

    # Make it as 0 or 1
    y_df[y_df == 'Yes'] = 1.
    y_df[y_df == 'No'] = 0.
    y_df = y_df.values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'churn', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'churn', 'test%d.txt' % fold),
                           header=None)[0].values

    cat_features = X_df.columns[X_df.dtypes == object]
    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='classification',
        cat_features=cat_features,
        metric='negative_auc',
        quantile_noise=1e-6,
    )


def fetch_CREDIT(path='./data/', fold=0):
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' \
                                 % fold
    data_path = pjoin(path, 'credit', 'creditcard.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'credit'), exist_ok=True)
        # Since this file is large, needs to use the custom function
        print('Downloading the file and extract....')
        download_file_from_google_drive('1TdxRae273iTnYVQnOZzb9bmHBllcMOT6',
                                        pjoin(path, 'credit.zip'))
        with ZipFile(pjoin(path, 'credit.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'credit.zip'))

    df = pd.read_csv(data_path)

    X_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]
    y_df = y_df.values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'credit', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'credit', 'test%d.txt' % fold),
                           header=None)[0].values
    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='classification',
        metric='negative_auc',
        quantile_noise=1e-5,
    )


def fetch_SUPPORT2(path='./data/', fold=0):
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' \
                                 % fold
    data_path = pjoin(path, 'support2', 'support2.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'support2'), exist_ok=True)
        download('https://docs.google.com/uc?id=10J1xl6ii3dZS-RZ9AIX2z0l5C4rsKLn-',
                 pjoin(path, 'support2.zip'))
        with ZipFile(pjoin(path, 'support2.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'support2.zip'))

    df = pd.read_csv(data_path)

    cat_cols = ['sex', 'dzclass', 'race', 'ca', 'income']
    target_variables = ['hospdead']
    remove_features = ['death', 'slos', 'd.time', 'dzgroup', 'charges', 'totcst',
                       'totmcst', 'aps', 'sps', 'surv2m', 'surv6m', 'prg2m', 'prg6m',
                       'dnr', 'dnrday', 'avtisst', 'sfdm2']

    df = df.drop(remove_features, axis=1)

    rest_colmns = [c for c in df.columns if c not in (cat_cols + target_variables)]
    # Impute the missing values for 0.
    df[rest_colmns] = df[rest_colmns].fillna(0.)

    df['income'][df['income'].isna()] = 'NaN'
    df['income'][df['income'] == 'under $11k'] = ' <$11k'
    df['race'][df['race'].isna()] = 'NaN'

    X_df = df.drop(target_variables, axis=1)
    y_df = df[target_variables[0]].values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'support2', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'support2', 'test%d.txt' % fold),
                           header=None)[0].values
    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        cat_features=cat_cols,
        problem='classification',
        metric='negative_auc',
        quantile_noise=1e-4,
    )


def fetch_MIMIC3(path='./data/', fold=0):
    'https://drive.google.com/file/d//view?usp=sharing'
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' % fold
    data_path = pjoin(path, 'mimic3', 'adult_icu.gz')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'mimic3'), exist_ok=True)
        download('https://docs.google.com/uc?id=16c0VTnZxw1xwzMzx2jwEClPBNeW_fc8-',
                 pjoin(path, 'mimic3.zip'))
        with ZipFile(pjoin(path, 'mimic3.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'mimic3.zip'))

    df = pd.read_csv(data_path, compression='gzip')

    train_cols = [
        'age', 'first_hosp_stay', 'first_icu_stay', 'adult_icu', 'eth_asian',
        'eth_black', 'eth_hispanic', 'eth_other', 'eth_white',
        'admType_ELECTIVE', 'admType_EMERGENCY', 'admType_NEWBORN',
        'admType_URGENT', 'heartrate_min', 'heartrate_max', 'heartrate_mean',
        'sysbp_min', 'sysbp_max', 'sysbp_mean', 'diasbp_min', 'diasbp_max',
        'diasbp_mean', 'meanbp_min', 'meanbp_max', 'meanbp_mean',
        'resprate_min', 'resprate_max', 'resprate_mean', 'tempc_min',
        'tempc_max', 'tempc_mean', 'spo2_min', 'spo2_max', 'spo2_mean',
        'glucose_min', 'glucose_max', 'glucose_mean', 'aniongap', 'albumin',
        'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
        'hematocrit', 'hemoglobin', 'lactate', 'magnesium', 'phosphate',
        'platelet', 'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc']

    label = 'mort_icu'

    X_df = df[train_cols]
    y_df = df[label].values.astype(np.int64)

    train_idx = pd.read_csv(pjoin(path, 'mimic3', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'mimic3', 'test%d.txt' % fold),
                           header=None)[0].values
    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='classification',
        metric='negative_auc',
        quantile_noise=1e-7,
    )


def fetch_WINE(path='./data/', fold=0):
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' % fold

    data_path = pjoin(path, 'wine', 'winequality-white.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'wine'), exist_ok=True)
        download('https://docs.google.com/uc?id=1BV2xG1JzBCo6OjchUSzxJO877zILj4oZ',
                 pjoin(path, 'wine.zip'))
        with ZipFile(pjoin(path, 'wine.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'wine.zip'))

    df = pd.read_csv(data_path, delimiter=';')

    y_df = df['quality'].values.astype(np.float32)
    X_df = df.drop(['quality'], axis=1)

    train_idx = pd.read_csv(pjoin(path, 'wine', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'wine', 'test%d.txt' % fold),
                           header=None)[0].values
    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='regression',
        quantile_noise=1e-8,
    )


def fetch_BIKESHARE(path='./data/', fold=0):
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' % fold

    data_path = pjoin(path, 'bikeshare', 'hour.csv')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'wine'), exist_ok=True)
        download('https://docs.google.com/uc?id=1sHpV9q3_tK0Uov2iRxT6TzTiRZxyx9K8',
                 pjoin(path, 'bikeshare.zip'))
        with ZipFile(pjoin(path, 'bikeshare.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'bikeshare.zip'))

    df = pd.read_csv(data_path).set_index('instant')
    train_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
                  'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    label = 'cnt'

    X_df = df[train_cols]
    y_df = df[label].values.astype(np.float32)

    train_idx = pd.read_csv(pjoin(path, 'bikeshare', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'bikeshare', 'test%d.txt' % fold),
                           header=None)[0].values
    return dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='regression',
        quantile_noise=1e-6,
    )


def fetch_ROSSMANN(path='./data/', fold=0):
    train_path = pjoin(path, 'rossmann-store-sales-preprocessed', 'train')
    if not pexists(train_path):
        os.makedirs(pjoin(path, 'rossmann-store-sales-preprocessed'), exist_ok=True)
        download('https://docs.google.com/uc?id=1CSI7ETLo50fksaK7Z_YxVzc7BaQNwtjw',
                 pjoin(path, 'rossmann-store-sales-preprocessed.zip'))
        with ZipFile(pjoin(path, 'rossmann-store-sales-preprocessed.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'rossmann-store-sales-preprocessed.zip'))

    def load_X_y(path):
        df = pd.read_csv(path, delimiter='\t')
        X_df = df.drop(['Sales'], axis=1)
        y_df = df['Sales'].values.astype(np.float32)
        return X_df, y_df

    X_train, y_train = load_X_y(train_path)
    test_path = pjoin(path, 'rossmann-store-sales-preprocessed', 'test')
    X_test, y_test = load_X_y(test_path)

    cat_features = [
        'Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'StoreType', 'Assortment', 'Promo2', 'Promo2Start_Jan', 'Promo2Start_Feb',
        'Promo2Start_Mar', 'Promo2Start_Apr', 'Promo2Start_May', 'Promo2Start_Jun',
        'Promo2Start_Jul', 'Promo2Start_Aug', 'Promo2Start_Sept', 'Promo2Start_Oct',
        'Promo2Start_Nov', 'Promo2Start_Dec']

    return dict(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        problem='regression',
        cat_features=cat_features,
        quantile_noise=1e-4,
    )


def fetch_SARCOS(path='./data/', fold=0, target_id=None):
    assert 0 <= fold <= 4, 'fold is only allowed btw 0 and 4, but get %d' % fold

    data_path = pjoin(path, 'sarcos', 'sarcos_inv.mat')
    if not pexists(data_path):
        os.makedirs(pjoin(path, 'sarcos'), exist_ok=True)
        download('https://docs.google.com/uc?id=1RjCYB87f2L1vL6lx2evA2Wqtpqcb2gYj',
                 pjoin(path, 'sarcos.zip'))
        with ZipFile(pjoin(path, 'sarcos.zip'), 'r') as zipObj:
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path)
        os.remove(pjoin(path, 'sarcos.zip'))

    import mat4py
    df = pd.DataFrame(mat4py.loadmat(data_path)['sarcos_inv'])

    y_df = df.iloc[:, 21:].values.astype(np.float32)
    X_df = df.iloc[:, :21].astype(np.float32)

    train_idx = pd.read_csv(pjoin(path, 'sarcos', 'train%d.txt' % fold),
                            header=None)[0].values
    test_idx = pd.read_csv(pjoin(path, 'sarcos', 'test%d.txt' % fold),
                           header=None)[0].values
    result = dict(
        X_train=X_df.iloc[train_idx], y_train=y_df[train_idx],
        X_test=X_df.iloc[test_idx], y_test=y_df[test_idx],
        problem='regression',
        metric='multiple_mse',
        num_classes=7,
        addi_tree_dim=-6,
    )
    if target_id is None:
        return result

    assert 0 <= target_id < 7, f'Only has 7 tasks! {target_id}'
    result['y_train'] = result['y_train'][:, target_id]
    result['y_test'] = result['y_test'][:, target_id]
    del result['metric']
    del result['num_classes']
    del result['addi_tree_dim']
    return result


DATASETS = {
    # NODE datasets
    'A9A': fetch_A9A,
    'EPSILON': fetch_EPSILON,
    'PROTEIN': fetch_PROTEIN, # multi-class
    'YEAR': fetch_YEAR,
    'HIGGS': fetch_HIGGS,
    'MICROSOFT': fetch_MICROSOFT,
    'YAHOO': fetch_YAHOO,
    'CLICK': fetch_CLICK,
    # The rest are GAMs datasets
    'MIMIC2': fetch_MIMIC2,
    'COMPAS': fetch_COMPAS,
    'ADULT': fetch_ADULT,
    'CHURN': fetch_CHURN,
    'CREDIT': fetch_CREDIT,
    'SUPPORT2': fetch_SUPPORT2,
    'MIMIC3': fetch_MIMIC3,
    # My found
    'ROSSMANN': fetch_ROSSMANN,
    'WINE': fetch_WINE,
    'BIKESHARE': fetch_BIKESHARE,
    # Multi-task regression
    'SARCOS': fetch_SARCOS,
    'SARCOS0': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=0, **kwargs),
    'SARCOS1': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=1, **kwargs),
    'SARCOS2': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=2, **kwargs),
    'SARCOS3': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=3, **kwargs),
    'SARCOS4': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=4, **kwargs),
    'SARCOS5': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=5, **kwargs),
    'SARCOS6': lambda *args, **kwargs: fetch_SARCOS(*args, target_id=6, **kwargs),
}
