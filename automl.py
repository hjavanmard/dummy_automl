import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

def fit_estimator(X, y, estimator_name, metric_name, hyper_parameter_tuning=False, regression=None):
    '''
    X: a data frame of features (only numeric columns)
    y: target column
    estimator_name: the name of estimator wanted to be fitted on data (should be one of supported models)
    metric_name: is used to score model during cross validation but not in optimizing each model
    hyper_parameter_tuning: if True it will tune model params otherwise will use default values
    regression: if True it will use regressors otherwise classifiers; if None it will check if it is regression
    '''
    # we check regression is True or False in case there is no user input on that one
    if regression is None:
        is_numeric = np.issubdtype(y.dtype, np.number)
        print (type(y))
        if is_numeric and y.nunique() > 5:
            regression = True
        else:
            regression = False
    
    if regression:
        estimator = regressors[estimator_name]['regressor']
        param_distributions = regressors[estimator_name]['ParameterRanges'] if hyper_parameter_tuning else {}
        metric = supported_metrics['regression'][metric_name]
        scorer = metrics.make_scorer(metric, greater_is_better=False) if '_error' in metric_name else metrics.make_scorer(metric)
    else:
        estimator = classifiers[estimator_name]['classifier']
        param_distributions = classifiers[estimator_name]['ParameterRanges'] if hyper_parameter_tuning else {}
        metric = supported_metrics['classification'][metric_name]
        scorer = metrics.make_scorer(metric, greater_is_better=False) if '_error' in metric_name else metrics.make_scorer(metric)
    # in case we need to do some preprocessing (later will be more)
    one_hot_encoder = OneHotEncoder(n_values=10)
    pipeline = Pipeline([('one_hot_encoder', one_hot_encoder), ('estimator', estimator)])
    param_distributions = {('estimator__' + key):value for key, value in param_distributions.items()}
    print(pipeline.fit(X,y))
    if hyper_parameter_tuning:
        rsv = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, scoring=scorer, n_jobs=-1, cv=3, random_state=1234,verbose=0)
    else:
        rsv = RandomizedSearchCV(pipeline, param_distributions={}, n_iter=10, scoring=scorer, n_jobs=-1, cv=3, random_state=1234, verbose=0)
    rsv.fit(X, y)
    fitted_model = rsv.best_estimator_
    training_score = -1 * rsv.best_score_ if '_error' in metric_name else rsv.best_score_
    return fitted_model, training_score

def pick_best_model(X, y, metric_name=None, hyper_parameter_tuning=False, regression=None, add_models ={}, drop_models=[]):
    # not supporting drop all models now
    models = {}
    # we check regression is True or False in case there is no user input on that one
    if regression is None:
        is_numeric = np.issubdtype(y.dtype, np.number)
        if is_numeric and y.nunique() > 5:
            regression = True
        else:
            regression = False
    if regression:
        models = regressors
    else:
        models = classifiers
    # drop user requested models
    models = {name:value for name, value in models.items() if name not in drop_models}
    # add extra models not included by default
    for model, value in add_models.items():
        models[model] = value
    fitted_results = {}
    best_model_name = ''
    maximize = '_score' in metric_name
    best_score_sofar = - float('inf') if maximize else float('inf')
    for estimator_name in models.keys():
        print ('fitting a ' + estimator_name + ' model...')
        fitted_model, training_score = fit_estimator(X.copy(), y.copy(), estimator_name, metric_name, hyper_parameter_tuning, regression)
        fitted_results[estimator_name] = {'fitted_model': fitted_model, 'training_score': training_score}
        if maximize:
            if training_score > best_score_sofar:
                best_model_name = estimator_name
                best_score_sofar = training_score
        else:
            if training_score < best_score_sofar:
                best_model_name = estimator_name
                best_score_sofar = training_score
    print (fitted_results[best_model_name])
    return fitted_results, best_model_name
    

