"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import numpy as np
import pandas as pd
from interpret.glassbox.ebm.ebm import EBMPreprocessor
from interpret.utils import unify_data, autogen_schema
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler

from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin, \
    OnehotEncodingRegressorMixin, OnehotEncodingClassifierMixin
from .base import MyGAMPlotMixinBase, MyCommonBase
from .utils import my_interpolate


class MyTransformMixin(object):
    def transform(self, X):
        return X


class MyTransformClassifierMixin(MyTransformMixin):
    def predict_proba(self, X):
        X = self.transform(X)
        return super().predict_proba(X)


class MyTransformRegressionMixin(MyTransformMixin):
    def predict(self, X):
        X = self.transform(X)
        return super().predict(X)


class MyStandardizedTransformMixin(object):
    def __init__(self, *args, **kwargs):
        assert isinstance(self, (MyTransformClassifierMixin, MyTransformRegressionMixin))
        super().__init__(*args, **kwargs)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        return super().fit(X, y)

    def transform(self, X):
        X = self.scaler.transform(X)
        return super().transform(X)


class MyMaxMinTransformMixin(object):
    def __init__(self, *args, **kwargs):
        assert isinstance(self, (MyTransformClassifierMixin, MyTransformRegressionMixin))
        super().__init__(*args, **kwargs)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        return super().fit(X, y)

    def transform(self, X):
        X = self.scaler.transform(X)
        return super().transform(X)


class MyEBMPreprocessorTransformMixin(object):
    def __init__(self, binning='uniform', **kwargs):
        assert isinstance(self, (MyTransformClassifierMixin, MyTransformRegressionMixin))
        super().__init__(**kwargs)

        self.prepro_feature_names = None
        self.feature_types = None
        self.schema = None
        self.binning = binning

    def fit(self, X, y):
        X, y, self.prepro_feature_names, _ = unify_data(
            X, y
        )

        # Build preprocessor
        self.schema_ = self.schema
        if self.schema_ is None:
            self.schema_ = autogen_schema(
                X, feature_names=self.prepro_feature_names, feature_types=self.feature_types
            )

        self.preprocessor_ = EBMPreprocessor(schema=self.schema_, binning=self.binning)
        self.preprocessor_.fit(X)

        X = self.preprocessor_.transform(X)
        return super().fit(X, y)

    def transform(self, X):
        X, _, _, _ = unify_data(X, None, self.prepro_feature_names, self.feature_types)
        X = self.preprocessor_.transform(X)
        return super().transform(X)


class MyMarginalizedTransformMixin(object):
    def __init__(self, *args, **kwargs):
        assert isinstance(self, (MyTransformClassifierMixin, MyTransformRegressionMixin))
        super().__init__(*args, **kwargs)
        self.X_mapping = {}

    def fit(self, X, y):
        # My marginal transformation
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        original_columns = X.columns
        X['label'] = y
        for col_idx, col in enumerate(original_columns):
            self.X_mapping[col_idx] = X.groupby(col).label.apply(lambda x: x.mean())
        X = X.drop('label', axis=1)

        X = self._transform(X)
        return super().fit(X, y)

    def transform(self, X):
        X = self._transform(X)
        return super().transform(X)

    def _transform(self, X):
        assert len(self.X_mapping) > 0

        if isinstance(X, pd.DataFrame):
            X = X.values

        new_X = np.empty(X.shape, dtype=np.float)
        for col_idx in range(X.shape[1]):
            x_unique = np.sort(np.unique(X[:, col_idx]))
            x_map = self.X_mapping[col_idx]
            if len(x_map) != len(x_unique) or np.any(x_map.index != x_unique):
                new_y = my_interpolate(x_map.index, x_map.values, x_unique)
                x_map = pd.Series(new_y, index=x_unique)

            new_X[:, col_idx] = x_map[X[:, col_idx]].values
        return new_X


class MyIndicatorTransformMixin(object):
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        categorical_features = X.nunique() > 2
        if not np.any(categorical_features):
            self.enc = FunctionTransformer()
        else:
            self.enc = ColumnTransformer([
                ('onehot', OneHotEncoder(handle_unknown='error'), categorical_features)
            ], remainder='passthrough')

        X = self.enc.fit_transform(X.values)
        return super().fit(X, y)

    def transform(self, X):
        X = self.enc.transform(X)
        return super().transform(X)


''' ----------------------- Transformations Mixin Class Ends ----------------- '''

# Somehow we need to set all the arguments on the parameter lists __init__ to avoid the error
class MyLogisticRegressionCVBase(LogisticRegressionCV):
    def __init__(self, Cs=12, cv=5, penalty='l2', random_state=1377, solver='lbfgs', max_iter=3000,
                 n_jobs=-1, **kwargs):
        super().__init__(Cs=Cs, cv=cv, penalty=penalty, random_state=random_state, solver=solver,
            max_iter=max_iter, n_jobs=n_jobs, **kwargs)

    def _my_predict_logodds(self, X):
        return self.decision_function(self.transform(X))


class MyLinearRegressionCVBase(RidgeCV):
    def __init__(self, alphas=np.logspace(-3, 3, 12), **kwargs):
        super().__init__(alphas, **kwargs)
    

class MyLogisticRegressionCV(OnehotEncodingClassifierMixin, MyGAMPlotMixinBase,
                             MyStandardizedTransformMixin, MyTransformClassifierMixin,
                             MyLogisticRegressionCVBase):
    pass

class MyLinearRegressionRidgeCV(OnehotEncodingRegressorMixin, MyGAMPlotMixinBase,
                                MyStandardizedTransformMixin, MyTransformRegressionMixin,
                                MyLinearRegressionCVBase):
    pass

class MyMarginalLogisticRegressionCV(LabelEncodingClassifierMixin, MyGAMPlotMixinBase,
                                     MyEBMPreprocessorTransformMixin, MyMarginalizedTransformMixin,
                                     MyTransformClassifierMixin, MyLogisticRegressionCVBase):
    pass

class MyMarginalLinearRegressionCV(LabelEncodingRegressorMixin, MyGAMPlotMixinBase,
                                   MyEBMPreprocessorTransformMixin, MyMarginalizedTransformMixin,
                                   MyTransformRegressionMixin, MyLinearRegressionCVBase):
    pass

class MyIndicatorLogisticRegressionCV(LabelEncodingClassifierMixin, MyGAMPlotMixinBase,
                                      MyEBMPreprocessorTransformMixin, MyIndicatorTransformMixin,
                                      MyTransformClassifierMixin, MyLogisticRegressionCVBase):
    pass

class MyIndicatorLinearRegressionCV(LabelEncodingRegressorMixin, MyGAMPlotMixinBase,
                                    MyEBMPreprocessorTransformMixin, MyIndicatorTransformMixin,
                                    MyTransformRegressionMixin, MyLinearRegressionCVBase):
    pass

class MyRandomForestClassifier(LabelEncodingClassifierMixin, MyCommonBase, RandomForestClassifier):
    @property
    def is_GAM(self):
        return False

class MyRandomForestRegressor(LabelEncodingRegressorMixin, MyCommonBase, RandomForestRegressor):
    @property
    def is_GAM(self):
        return False
