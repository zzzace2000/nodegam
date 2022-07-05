"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin, \
    OnehotEncodingClassifierMixin, OnehotEncodingRegressorMixin
from .base import MyGAMPlotMixinBase


class MyXGBMixin(object):
    def __init__(
        self,
        max_depth=1,
        random_state=1377,
        n_estimators=5000,
        n_jobs=-1,
        # My own parameter
        model_cls=XGBClassifier,
        validation_size=0.15,
        early_stopping_rounds=50,
        objective=None,
        **kwargs,
    ):
        if objective is None:
            objective = 'binary:logistic' if model_cls == XGBClassifier else 'reg:squarederror'

        self.objective = objective
        self.model = model_cls(
            max_depth=max_depth, random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=n_jobs, objective=objective,
            **kwargs)

        self.validation_size = validation_size
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.onehot_columns = None
        self.clean_feat_names = None

    def fit(self, X, y, verbose=False, **kwargs):
        stratify = None if isinstance(self.model, XGBRegressor) else y
        the_X_train, the_X_val, the_y_train, the_y_val = train_test_split(
            X, y,
            random_state=self.random_state,
            test_size=self.validation_size,
            stratify=stratify)

        eval_metric = 'logloss' if isinstance(self.model, XGBClassifier) else 'rmse'
        self.model.fit(
            the_X_train, the_y_train,
            eval_set=[(the_X_val, the_y_val)],
            eval_metric=eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=verbose,
            **kwargs)

    @property
    def is_GAM(self):
        return self.model.max_depth == 1

    def get_params(self, *args, **kwargs):
        return self.model.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        return self.model.set_params(*args, **kwargs)


class MyXGBClassifier(MyGAMPlotMixinBase, MyXGBMixin):
    def __init__(self, *args, **kwargs):
        kwargs['model_cls'] = XGBClassifier
        super().__init__(*args, **kwargs)

    def predict_proba(self, data, ntree_limit=None, validate_features=True):
        return self.model.predict_proba(data, ntree_limit=ntree_limit, validate_features=False)


class MyXGBRegressor(MyGAMPlotMixinBase, MyXGBMixin):
    def __init__(self, *args, **kwargs):
        kwargs['model_cls'] = XGBRegressor
        super().__init__(*args, **kwargs)

    def predict(self, data, output_margin=False, ntree_limit=None, validate_features=True):
        return self.model.predict(data, output_margin=output_margin,
                                  ntree_limit=ntree_limit, validate_features=False)


class MyXGBOnehotClassifier(OnehotEncodingClassifierMixin, MyXGBClassifier):
    """XGB-GAM Classifier with one-hot encoding for categorical features.

    Args:
        max_depth=1: The tree depth of the package. Should be set to 1 to remain as a GAM.
        random_state=1377: Seed.
        n_estimators=5000: Maximum number of rounds to fit.
        n_jobs=-1: Set to -1 to use multi-thread parallel training.
        validation_size=0.15: The validation porportion.
        early_stopping_rounds=50: Early stopping rounds.
        objective='binary\:logistic': The validation objective.
    """

class MyXGBOnehotRegressor(OnehotEncodingRegressorMixin, MyXGBRegressor):
    """XGB-GAM Regressor with one-hot encoding for categorical features.

    Args:
        max_depth=1: The tree depth of the package. Should be set to 1 to remain as a GAM.
        random_state=1377: Seed.
        n_estimators=5000: Maximum number of rounds to fit.
        n_jobs=-1: Set to -1 to use multi-thread parallel training.
        validation_size=0.15: The validation porportion.
        early_stopping_rounds=50: Early stopping rounds.
        objective='reg\:squarederror': The validation objective.
    """

class MyXGBLabelEncodingClassifier(LabelEncodingClassifierMixin, MyXGBClassifier):
    pass

class MyXGBLabelEncodingRegressor(LabelEncodingRegressorMixin, MyXGBRegressor):
    pass
