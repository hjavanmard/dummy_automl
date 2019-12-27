# dummy_automl
A simple auto model tuning and hyper parameter tuning helper function based on scikit-learn

## Examples:

### 1. Tune model hyperparameters
If one wants just tune a specific model parameteres, say a random forest model

from automl import fit_estimator

fitted_model, training_score = fit_estimator(X, y, estimator_name='random_forest', metric_name='mean_squared_error', hyper_parameter_tuning=True)

### 2. Pick the best performing model
If one wants to find tune both models and their parameters to find the best one

from automl import pick_best_model

fitted_results, best_model_name = pick_best_model(X, y, metric_name='mean_squared_error', hyper_parameter_tuning=True, add_models ={}, drop_models=[])


