from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
#from xgboost import XGBClassifier, XGBRegressor


# those with '_score' will be maximized and those with '_error' minimized
supported_metrics = {
    'classification':{'accuracy_score': metrics.accuracy_score,
                      'balanced_accuracy_score': metrics.balanced_accuracy_score,
                      'balanced_accuracy_score': metrics.balanced_accuracy_score,
                      'average_precision_score': metrics.average_precision_score,
                      'f1_score': metrics.f1_score,
                      'neg_log_loss': metrics.log_loss,
                      'roc_auc_score': metrics.roc_auc_score
                     },
    'regression':{'r2_score': metrics.r2_score,
                 'median_absolute_error': metrics.median_absolute_error,
                 'mean_squared_error': metrics.mean_squared_error,
                 'mean_absolute_error': metrics.mean_absolute_error
                 }
}



regressors = {'random_forest': {'regressor': RandomForestRegressor(random_state=1234, n_jobs=-1),
                                'ParameterRanges': {'n_estimators': range(10, 1000,50),
                                                    'max_depth': [*range(4, 20)] + [None],
                                                    'min_samples_split': range(2, 6),
                                                    'max_features': ['sqrt', 'log2', None],
                                                    'criterion': ['mse', 'mae']}
                               },
              'gradient_boositng': {'regressor': GradientBoostingRegressor(random_state=1234),
                                   'ParameterRanges':{'learning_rate': np.arange(0.05, 0.2, 0.05),
                                                     'n_estimators': range(100, 1000, 100),
                                                     'max_depth': range(3, 10),
                                                     'min_samples_split': range(2, 5),
                                                     'max_features': ['sqrt', 'log2', None],
                                                     'loss': ['ls', 'lad', 'huber']},
                                  },
             'linear_regression': {'regressor': LinearRegression(),
                                   'ParameterRanges':{'normalize':[True, False]}
                                  },
             'svm': {'regressor': SVR(max_iter=1000),
                     'ParameterRanges':{'C': np.arange(0.2, 1.1, 0.2),
                                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
                    }
             }
classifiers = {'random_forest': {'classifier': RandomForestClassifier(random_state=1234, n_jobs=-1),
                                 'ParameterRanges': {'n_estimators': range(10, 1000,50),
                                                    'max_depth': [*range(4, 20)] + [None],
                                                    'min_samples_split': range(2, 6),
                                                    'max_features': ['sqrt', 'log2', None],
                                                    'criterion': ['gini', 'entropy']}
                                },
             'gradient_boositng': {'classifier': GradientBoostingClassifier(random_state=1234),
                                   'ParameterRanges':{'learning_rate': np.arange(0.05, 0.2, 0.05),
                                                     'n_estimators': range(100, 1000, 100),
                                                     'max_depth': range(3, 10),
                                                     'min_samples_split': range(2, 5),
                                                     'max_features': ['sqrt', 'log2', None],
                                                     'loss': ['deviance', 'exponential'],
                                                     'criterion': ['friedman_mse', 'mse', 'mae']}
                                  },
             'logistic_regression': {'classifier': LogisticRegression(random_state=1234, n_jobs=-1),
                                   'ParameterRanges':{'penalty': ['l1', 'l2'],
                                                      'C': np.arange(0.2, 1.1, 0.2),
                                                      'max_iter': range(100, 1000)}
                                  },
             'svm': {'classifier': SVC(random_state=1234, max_iter=1000),
                     'ParameterRanges':{'C': np.arange(0.2, 1.1, 0.2),
                                        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
                    }
              }

