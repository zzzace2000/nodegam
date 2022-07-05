"""Adapted from https://github.com/zzzace2000/GAMs_models/.

It implements the bagging of GAM models. Unlike sklearn implmementation of BaggingClassifier, it
averages the logits instead of the probability to make sure the bagging of GAMs is still a GAM. It
also implements the get_GAM_df() that automatically takes average of the GAMs under bagging.

Usage:
>>> from nodegam.gams.MyXGB import MyXGBClassifier
>>> from nodegam.gams.MyBagging import MyBaggingClassifier
>>> base_model = MyXGBClassifier()
>>> # Train an XGB-GAM with 10 times bagging
>>> bag_model = MyBaggingClassifier(base_model=base_model, n_estimators=10)
>>> bag_model.fit(X, y)
>>> df = bag_model.get_GAM_df()
"""


import numpy as np
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

from .EncodingBase import LabelEncodingRegressorMixin, LabelEncodingClassifierMixin, \
    OnehotEncodingClassifierMixin, OnehotEncodingRegressorMixin
from .base import eps, MyGAMPlotMixinBase, MyCommonBase
from .utils import get_GAM_df_by_models, sigmoid


class MyBaggingMixin(MyGAMPlotMixinBase):
    def get_GAM_df(self, x_values_lookup=None, get_y_std=True):
        """Get the GAM graph parameter.

        Args:
            x_values_lookup: a dictionary of mapping feature name to its correpsonding unique
                increasing x. E.g. {'BUN': [1.1, 1.5, 3.1, 5.0], 'cancer': [0, 1]}.
            get_y_std: to get the error bar of the y. It's slower if this is set to true.
                Default: True

        Return:
            A dataframe of GAM graph.
        """

        assert self.is_GAM, 'Only supports visualization when it is a GAM'

        if not get_y_std: # Use the predict function to derive the GAM
            df = super().get_GAM_df(x_values_lookup)

            # Modify the feature name to be the outer feature name
            df.feat_name = ['offset'] + self.feature_names
            return self.revert_dataframe(df)

        # Since the submodel is fitted by np array, the x_values_lookup need to change
        if x_values_lookup is None:
            x_values_lookup = {
                'f%d' % idx : np.array(list(self.X_values_counts[feat_name].keys()))
                for idx, feat_name in enumerate(self.feature_names)
            }

        if 'f0' not in x_values_lookup:
            x_values_lookup = {
                'f%d' % idx : x_values_lookup[feat_name]
                for idx, feat_name in enumerate(self.feature_names)
            }

        all_old_dfs = get_GAM_df_by_models(self.estimators_, x_values_lookup, aggregate=False)

        # Loop through all_dfs, making them into the original form
        all_dfs = []
        for df in all_old_dfs:
            df.feat_name = ['offset'] + self.feature_names
            all_dfs.append(self.revert_dataframe(df))

        # aggregate them
        first_df = all_dfs[0]
        all_ys = [np.concatenate(df.y) for df in all_dfs]

        split_pts = first_df.y.apply(lambda x: len(x)).cumsum()[:-1]
        first_df['y'] = np.split(np.mean(all_ys, axis=0), split_pts)
        first_df['y_std'] = np.split(np.std(all_ys, axis=0), split_pts)

        # Calculate the importances
        importances = [-1.]
        for feat_name, x, y in zip(first_df.feat_name.iloc[1:].values, first_df.x.iloc[1:].values,
                                   first_df.y.iloc[1:].values):
            if feat_name in self.X_values_counts:
                model_xs = np.unique(list(self.X_values_counts[feat_name].keys()))

                # TODO: only caculate importance on X values that's the same this model for now.
                if len(model_xs) != len(x) or np.any(model_xs != x):
                    importance = np.nan
                else:
                    # new_map = pd.Series(y, index=x)
                    # model_ys = new_map[model_xs]
                    importance = np.average(np.abs(y),
                                            weights=list(self.X_values_counts[feat_name].values()))
            else:
                print('No counts recorded for feature: %s. Skip' % feat_name)
                importance = np.nan
            importances.append(importance)
        
        first_df['importance'] = importances

        return first_df

    @property
    def is_GAM(self):
        return hasattr(self.base_estimator, 'is_GAM') and self.base_estimator.is_GAM


class MyBaggingClassifierBase(MyBaggingMixin, BaggingClassifier):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        # Tell the encoding no need to do reversion in get_GAM_df()
        self.not_revert = True

    def predict_proba(self, X, parallel=False):
        """Modify it to be using the average of the log-odds instead of avg probobability.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Args:
            X: {array-like, sparse matrix} of shape = [n_samples, n_features]
                The training input samples. Sparse matrices are accepted only if they are supported
                by the base estimator.
            parallel: if True, predict outputs using parallel threads to speed up. But in xgboost,
                the base estimator already uses multiple threads so it actually slows down.

        Returns:
            p:array of shape = [n_samples, n_classes].
                The class probabilities of the input samples. The order of the classes corresponds
                to that in the attribute `classes_`.
        """

        from sklearn.ensemble.bagging import _parallel_predict_proba
        from sklearn.ensemble.base import _partition_estimators
        from sklearn.utils.validation import check_array, check_is_fitted
        from sklearn.utils._joblib import Parallel, delayed

        check_is_fitted(self, "classes_")
        # Check data
        X = check_array(
            X, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False
        )

        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # Single thread loop
        if not parallel:
            all_proba = [est.predict_proba(X) for est in self.estimators_]
        else: # Parallel loop
            n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                                self.n_estimators)

            all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_predict_proba)(
                    self.estimators_[starts[i]:starts[i + 1]],
                    self.estimators_features_[starts[i]:starts[i + 1]],
                    X,
                    self.n_classes_)
                for i in range(n_jobs))

        # Reduce
        all_proba = np.stack(all_proba).astype(np.float64)
        mean_log_odds = np.nanmean(np.log(all_proba + eps) - np.log(1. - all_proba + eps), axis=0)
        if np.any(np.isnan(mean_log_odds)):
            mean_log_odds = np.where(np.isnan(mean_log_odds), np.nanmean(mean_log_odds, axis=0),
                                     mean_log_odds)

        proba = sigmoid(mean_log_odds)

        return proba


class MyBaggingLabelEncodingClassifier(LabelEncodingClassifierMixin, MyBaggingClassifierBase,
                                       MyCommonBase):
    pass


class MyBaggingClassifier(OnehotEncodingClassifierMixin, MyBaggingClassifierBase, MyCommonBase):
    """The bagging for the base estimator GAM.

    It overwrites the sklearn.ensemble.BaggingClassifier to (1) do ensemble on the logits and NOT
         the probabilities to make it still as a GAM, and (2) support GAM df extraction.

    Args:
        base_estimator: the base estimator model.
        n_estimators: how many number of estimators to do bagging.
        max_samples (int or float) = 1.0: the number of samples to draw from X to train each base
            estimator (with replacement by default, see `bootstrap` for more details).
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
        max_features (int or float) = 1.0: The number of features to draw from X to train each base
            estimator (without replacement by default, see `bootstrap_features` for more details).
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
        bootstrap (bool) = True: Whether samples are drawn with replacement. If False, sampling
            without replacement is performed.
        bootstrap_features (bool) = False: Whether features are drawn with replacement.
        oob_score (bool) = False: Whether to use out-of-bag samples to estimate
            the generalization error. Only available if bootstrap=True.
        warm_start (bool) = False: When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.
        n_jobs (int) = None: The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
        random_state: random state.
        verbose: verbose.
    """
    pass


class MyBaggingRegressorBase(MyBaggingMixin, BaggingRegressor):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        # Tell the encoding no need to do reversion in get_GAM_plot_df() 
        self.not_revert = True

class MyBaggingLabelEncodingRegressor(LabelEncodingRegressorMixin, MyBaggingRegressorBase,
                                      MyCommonBase):
    pass

class MyBaggingRegressor(OnehotEncodingRegressorMixin, MyBaggingRegressorBase, MyCommonBase):
    """The bagging for the base estimator GAM regressor.

    It overwrites the sklearn.ensemble.BaggingRegressor to support GAM df extraction.

    Args:
        base_estimator: the base estimator model.
        n_estimators: how many number of estimators to do bagging.
        max_samples (int or float) = 1.0: the number of samples to draw from X to train each base
            estimator (with replacement by default, see `bootstrap` for more details).
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
        max_features (int or float) = 1.0: The number of features to draw from X to train each base
            estimator (without replacement by default, see `bootstrap_features` for more details).
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.
        bootstrap (bool) = True: Whether samples are drawn with replacement. If False, sampling
            without replacement is performed.
        bootstrap_features (bool) = False: Whether features are drawn with replacement.
        oob_score (bool) = False: Whether to use out-of-bag samples to estimate
            the generalization error. Only available if bootstrap=True.
        warm_start (bool) = False: When set to True, reuse the solution of the previous call to fit
            and add more estimators to the ensemble, otherwise, just fit
            a whole new ensemble.
        n_jobs (int) = None: The number of jobs to run in parallel for both :meth:`fit` and
            :meth:`predict`. ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors.
        random_state: random state.
        verbose: verbose.
    """
    pass
