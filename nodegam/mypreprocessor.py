"""The preprocessor that normalizes and imputes the data."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, QuantileTransformer


class MyPreprocessor:
    def __init__(self, random_state=1377, cat_features=None, normalize=False,
                 y_normalize=False, quantile_transform=False,
                 output_distribution='normal', n_quantiles=2000,
                 quantile_noise=1e-3):
        """Preprocessor does the data preprocessing like input and target normalization.

        Args:
            random_state: Global random seed for an experiment.
            cat_features: If passed in, it does the ordinal encoding for these features before other
                input normalization like quantile transformation. Default: None.
            normalize: standardize features by removing the mean and scaling to unit variance.
            y_normalize: If True, it standardizes the targets y by setting the mean and stdev to 0
                and 1. Useful in the regression setting.
            quantile_transform: If True, transforms the features to follow a normal or uniform
                distribution.
            output_distribution: Choose between ['normal', 'uniform']. Data is projected onto this
                distribution. See the same param of sklearn QuantileTransformer. 'normal' is better.
            n_quantiles: Number of quantiles to estimate the distribution. Default: 2000.
            quantile_noise: If specified, fits QuantileTransformer on data with added gaussian noise
                with std = :quantile_noise: * data.std; this will cause discrete values to be more
                separable. Please note that this transformation does NOT apply gaussian noise to the
                resulting data, the noise is only applied for QuantileTransformer.

        Example:
            >>> preprocessor = nodegam.mypreprocessor.MyPreprocessor(
            >>>     cat_features=['ethnicity', 'gender'],
            >>>     y_normalize=True,
            >>>     random_state=1337,
            >>> )
            >>> preprocessor.fit(X_train, y_train)
            >>> X_train, y_train = preprocessor.transform(X_train, y_train)
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
        self.y_mu, self.y_std = 0, 1
        self.feature_names = None

    def fit(self, X, y=None):
        """Fit the transformer.

        Args:
            X (pandas daraframe): Input data.
            y (numpy array): Optional. If passed in, it will record the y_mean and y_std if
                'y_normalize' is set to True.
        """
        assert isinstance(X, pd.DataFrame), 'X is not a dataframe! %s' % type(X)
        self.feature_names = X.columns

        if self.cat_features is not None:
            cat_encoder = MyOrdinalEncoder(cat_features=self.cat_features)
            cat_encoder.fit(X)

            self.transformers.append(cat_encoder)
            X = cat_encoder.transform(X)

        if self.normalize:
            scaler = StandardScaler(copy=False)
            scaler.fit(X)
            self.transformers.append(scaler)

        if self.quantile_transform:
            quantile_train = X.copy()

            if self.quantile_noise:
                r = np.random.RandomState(self.random_state)
                stds = np.std(quantile_train.values, axis=0, keepdims=True)
                noise_std = self.quantile_noise / np.maximum(stds, self.quantile_noise)
                quantile_train += noise_std * r.randn(*quantile_train.shape)

            qt = QuantileTransformer(random_state=self.random_state,
                                     n_quantiles=self.n_quantiles,
                                     output_distribution=self.output_distribution,
                                     copy=False)
            qt.fit(quantile_train)
            self.transformers.append(qt)

        if y is not None and self.y_normalize:
            self.y_mu, self.y_std = y.mean(axis=0), y.std(axis=0)
            print("Normalize y. mean = {}, std = {}".format(self.y_mu, self.y_std))

    def transform(self, *args):
        """Transform the data.

        Args:
            X (pandas daraframe): Input data.
            y (numpy array): Optional. If passed in, it will do target normalization.

        Returns:
            X (pandas daraframe): Normalized Input data.
            y (numpy array): Optional. Normalized y.
        """
        assert len(args) <= 2

        X = args[0]
        if len(self.transformers) > 0:
            X = X.copy()
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names)

            for i, t in enumerate(self.transformers):
                # Leave one out transform when it's training set
                X = t.transform(X)

        # Make everything as numpy and float32
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = X.astype(np.float32)

        if len(args) == 1:
            return X

        y = args[1]
        if y is None:
            return X, None

        if self.y_normalize and self.y_mu is not None and self.y_std is not None:
            y = (y - self.y_mu) / self.y_std
            y = y.astype(np.float32)

        return X, y


class MyOrdinalEncoder:
    def __init__(self, cat_features=None):
        """My ordinal encoding that imputes the unseen category as the most frequent one.

        Args:
            cat_features: a list of column names to do ordinal encoding. If None, transform all.
        """
        self.cat_features = cat_features  # array of column names to encode
        self.cat_to_int_map = {}

    def fit(self, X):
        if self.cat_features is None:
            self.cat_features = X.columns

        # Create a per-feature ordinal encoder
        for c in self.cat_features:
            tmp = X[c].astype('category').cat
            self.cat_to_int_map[c] = pd.Series(range(len(tmp.categories)),
                                               index=tmp.categories.values)
        return self

    def transform(self, X):
        """Transforms columns of X specified in self.columns.

        Args:
            X (pandas datafranme): Input features.

        Returns:
            output (pandas dataframe): Transformed X.
        """
        output = X.copy()
        for col in self.cat_features:
            x_value = output[col]

            not_in_category = (~x_value.isin(set(self.cat_to_int_map[col].index))).copy()
            if not_in_category.any():
                x_value.loc[not_in_category] = x_value.mode().iloc[0]
            val = self.cat_to_int_map[col][x_value].values

            output[col] = val
        return output
