"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, \
    RandomizedSearchCV
from sklearn.utils import parallel_backend

from .EncodingBase import LabelEncodingFitMixin
from .MyBagging import MyBaggingClassifier, MyBaggingRegressor, \
    MyBaggingLabelEncodingClassifier, MyBaggingLabelEncodingRegressor
from .MyBaselines import MyLogisticRegressionCV, MyLinearRegressionRidgeCV, \
    MyMarginalLogisticRegressionCV, MyMarginalLinearRegressionCV, \
    MyIndicatorLinearRegressionCV, MyIndicatorLogisticRegressionCV, \
    MyRandomForestClassifier, MyRandomForestRegressor
from .utils import Timer


def get_rf_model(model_name, problem, random_state=1377, **kwargs):
    assert model_name.split('-')[0] == 'rf', \
        'the model_name is not in the supported format %s' % model_name

    assert problem in ['regression', 'classification']
    the_cls = MyRandomForestRegressor if problem == 'regression' \
        else MyRandomForestClassifier

    # Understand the format
    params = {'random_state': random_state, 'n_jobs': -1}
    for param_str in model_name.split('-')[1:]:
        if param_str.startswith('n'):
            params['n_estimators'] = int(param_str[1:])
        else:
            raise NotImplementedError('the param_str is not supported: %s' % param_str)

    params.update(kwargs)
    model = the_cls(**params)
    return model


def _get_lr_model_template(model_name, problem, cls_model_cls,
                           reg_model_cls, random_state=1377, **kwargs):
    assert problem in ['regression', 'classification']
    the_cls = reg_model_cls if problem == 'regression' else cls_model_cls

    # Understand the format
    params = {} if problem == 'regression' else {'random_state': random_state}

    is_bag = False
    bag_params = {'random_state': random_state, 'n_jobs': None}
    split_model_name = model_name.split('-')
    for param_str in split_model_name[1:]:
        if param_str.startswith('o'):
            bag_params['n_estimators'] = int(param_str[1:])
            is_bag = True
        elif param_str.startswith('r'):
            if problem == 'classification':
                params['random_state'] = int(param_str[1:])
            bag_params['random_state'] = int(param_str[1:])
        elif param_str == 'l1':
            if problem == 'classification':
                params['penalty'] = 'l1'
                params['solver'] = 'saga'
        elif param_str.startswith('q'):
            params['binning'] = 'quantile'
        else:
            raise NotImplementedError('the param_str is not supported %s' % param_str)

    params.update(kwargs)
    model = the_cls(**params)
    if is_bag:
        if problem == 'regression':
            bag_cls = MyBaggingLabelEncodingRegressor \
                if isinstance(model, LabelEncodingFitMixin) else MyBaggingRegressor
        else:
            bag_cls = MyBaggingLabelEncodingClassifier \
                if isinstance(model, LabelEncodingFitMixin) else MyBaggingClassifier

        model = bag_cls(base_estimator=model, **bag_params)

    return model


def get_lr_model(model_name, problem, random_state=1377, **kwargs):
    assert model_name.split('-')[0] == 'lr', 'the model_name is not in the supported format %s' \
                                             % model_name

    reg_model_cls = MyLinearRegressionRidgeCV
    if model_name.startswith('lr-l1') and problem == 'regression':
        raise NotImplementedError(
            'Somehow I can not override LassoCV. It has error key_error: no intercept...'
            'Not sure how to fix it.')

    return _get_lr_model_template(model_name, problem, random_state,
                                  cls_model_cls=MyLogisticRegressionCV,
                                  reg_model_cls=reg_model_cls, **kwargs)


def get_mlr_model(model_name, problem, random_state=1377, **kwargs):
    ''' Get Marginal Logistic Regression '''
    assert model_name.split('-')[0] == 'mlr', 'the model_name is not in the supported format %s' \
                                              % model_name
    return _get_lr_model_template(model_name, problem, random_state,
                                  cls_model_cls=MyMarginalLogisticRegressionCV,
                                  reg_model_cls=MyMarginalLinearRegressionCV,
                                  **kwargs)


def get_ilr_model(model_name, problem, random_state=1377, **kwargs):
    ''' Get Indicator Logistic Regression '''
    assert model_name.split('-')[0] == 'ilr', \
        'the model_name is not in the supported format %s' % model_name
    return _get_lr_model_template(model_name, problem, random_state,
                                  cls_model_cls=MyIndicatorLogisticRegressionCV,
                                  reg_model_cls=MyIndicatorLinearRegressionCV, **kwargs)


def get_xgb_model(model_name, problem, random_state=1377, **kwargs):
    from .MyXGB import MyXGBClassifier, MyXGBRegressor, \
        MyXGBLabelEncodingClassifier, MyXGBLabelEncodingRegressor

    assert model_name.split('-')[0] == 'xgb', \
        'the model_name is not in the supported format %s' % model_name

    assert problem in ['regression', 'classification']
    the_cls = MyXGBRegressor if problem == 'regression' else MyXGBClassifier

    # Understand the format
    params = {'random_state': random_state}

    is_bag = False
    bag_params = {'random_state': random_state, 'n_jobs': None}
    for param_str in model_name.split('-')[1:]:
        if param_str.startswith('d'):
            params['max_depth'] = int(param_str[1:])
        elif param_str == 'l':  # label-encoding instead of one-hot
            the_cls = MyXGBLabelEncodingRegressor if problem == 'regression' \
                else MyXGBLabelEncodingClassifier
        elif param_str.startswith('o'):
            bag_params['n_estimators'] = int(param_str[1:])
            is_bag = True
        elif param_str.startswith('cols'):  # 'cols0.9'
            params['colsample_bytree'] = float(param_str[4:])
        elif param_str.startswith('cv'):
            continue
        elif param_str.startswith('reg'):
            params['reg_lambda'] = float(param_str[3:])
        elif param_str.startswith('r'):
            params['random_state'] = int(param_str[1:])
            bag_params['random_state'] = int(param_str[1:])
        elif param_str.startswith('cw'):
            params['min_child_weight'] = float(param_str[2:])
        elif param_str.startswith('lr'):
            params['learning_rate'] = float(param_str[2:])
        elif param_str.startswith('nj'):
            bag_params['n_jobs'] = int(param_str[2:])
        else:
            raise NotImplementedError('the param_str is not in the supported format %s'
                                      % param_str)

    params.update(kwargs)
    model = the_cls(**params)
    if is_bag:
        if problem == 'regression':
            bag_cls = MyBaggingLabelEncodingRegressor \
                if isinstance(model, LabelEncodingFitMixin) else MyBaggingRegressor
        else:
            bag_cls = MyBaggingLabelEncodingClassifier \
                if isinstance(model, LabelEncodingFitMixin) else MyBaggingClassifier

        model = bag_cls(base_estimator=model, **bag_params)

    return model


def get_spline_model(model_name, problem, random_state=1377, **kwargs):
    assert model_name.split('-')[0] == 'spline', \
        'the model_name is not in the supported format %s' % model_name

    assert problem in ['regression', 'classification']
    from .MySpline import MySplineLogisticGAM, MySplineGAM
    the_cls = MySplineGAM if problem == 'regression' else MySplineLogisticGAM

    # Understand the format
    spline_params = {'search': True}

    is_bag = False
    bag_params = {'random_state': random_state, 'n_jobs': None}
    for param_str in model_name.split('-')[1:]:
        if param_str.startswith('lam'):
            spline_params['lam'] = float(param_str[3:])
            spline_params['search'] = False
        elif param_str == 'b':  # spline-b
            spline_params['fit_binary_feat_as_factor_term'] = True
        elif param_str.startswith('cv'):
            continue
        elif param_str.startswith('o'):
            bag_params['n_estimators'] = int(param_str[1:])
            is_bag = True
        elif param_str == 'v2':
            continue  # spline-v2
        elif param_str.startswith('r'):
            bag_params['random_state'] = int(param_str[1:])
        else:
            raise NotImplementedError('the param_str is not in the supported format %s'
                                      % param_str)

    spline_params.update(kwargs)
    model = the_cls(**spline_params)
    if is_bag:
        if problem == 'regression':
            bag_cls = MyBaggingLabelEncodingRegressor \
                if isinstance(model, LabelEncodingFitMixin) else MyBaggingRegressor
        else:
            bag_cls = MyBaggingLabelEncodingClassifier \
                if isinstance(model, LabelEncodingFitMixin) else MyBaggingClassifier

        model = bag_cls(base_estimator=model, **bag_params)

    return model


def get_ebm_model(model_name, problem, random_state=1377, **kwargs):
    from .MyEBM import MyExplainableBoostingClassifier, MyExplainableBoostingRegressor, \
        MyOnehotExplainableBoostingRegressor, MyOnehotExplainableBoostingClassifier

    the_cls = MyExplainableBoostingRegressor if problem == 'regression' \
        else MyExplainableBoostingClassifier

    assert model_name.split('-')[0] == 'ebm', 'the model_name is not in the supported format %s' \
                                              % model_name

    # Understand the format
    params = {'random_state': random_state}
    for param_str in model_name.split('-')[1:]:
        if param_str.startswith('o'):
            params['outer_bags'] = int(param_str[1:])
        elif param_str.startswith('cv'):
            continue
        elif param_str.startswith('it'):
            params['interactions'] = int(param_str[2:])
        elif param_str.startswith('i'):
            params['inner_bags'] = int(param_str[1:])
        elif param_str.startswith('r'):
            params['random_state'] = int(param_str[1:])
        elif param_str == 'h':  # onehot encoding
            the_cls = MyOnehotExplainableBoostingRegressor if problem == 'regression' \
                else MyOnehotExplainableBoostingClassifier
        else:
            raise NotImplementedError('the param_str is not in the supported format %s' % param_str)

    params.update(kwargs)
    ebm = the_cls(**params)
    return ebm


def get_model(X_train, y_train, problem, model_name, random_state=1377, **kwargs):
    assert np.any([model_name.startswith(k) \
                   for k in ['ebm', 'spline', 'skgbt', 'xgb', 'lr', 'mlr', 'ilr',
                             'rf', 'flam', 'rspline']]), \
        'Model name is wierd! %s' % model_name

    for k in ['ebm', 'spline', 'skgbt', 'xgb', 'lr', 'mlr', 'ilr', 'rf', 'flam', 'rspline']:
        if model_name.startswith(k):
            the_model = eval('get_%s_model' % k)(model_name, problem, random_state=random_state,
                                                 **kwargs)
            break
    else:
        raise RuntimeError('No model class found with name %s' % model_name)

    if not hasattr(the_model, 'param_distributions') \
            or the_model.param_distributions is None \
            or '-cv' not in model_name:
        the_model.fit(X_train, y_train)
    else:
        with Timer('Use cv to select hyperparameters'):
            cv_cls = StratifiedShuffleSplit if problem == 'classification' else ShuffleSplit
            scoring = 'roc_auc' if problem == 'classification' else 'neg_mean_squared_error'

            cv = cv_cls(n_splits=3, test_size=0.2, random_state=random_state)
            cv_model = RandomizedSearchCV(
                the_model, param_distributions=the_model.param_distributions,
                n_iter=8, n_jobs=8, scoring=scoring, cv=cv, refit=True,
                random_state=random_state, error_score=np.nan)

            with parallel_backend('loky'):
                cv_model.fit(X_train, y_train)

        the_model = cv_model.best_estimator_

    return the_model
