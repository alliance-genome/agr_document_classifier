import numpy as np
from scipy.stats import loguniform, expon, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


POSSIBLE_CLASSIFIERS = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': loguniform(1e-4, 1e+4),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'class_weight': [None, 'balanced'],
            'l1_ratio': np.linspace(0, 1, 10),
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'warm_start': [True, False]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': randint(5, 500),
            'max_depth': list(range(2, 10, 2)),
            'min_samples_split': randint(2, 20)
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': randint(10, 200),
            'learning_rate': loguniform(0.01, 0.5),
            'max_depth': list(range(2, 10, 2))
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {
            'n_estimators': randint(10, 200),
            'learning_rate': loguniform(0.01, 0.5),
            'max_depth': list(range(2, 10, 2))
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (500,),
                                   (50, 50), (100, 100), (500, 500),
                                   (50, 50, 50), (100, 100, 100), (500, 500, 500),
                                   (50, 100, 50), (100, 500, 100), (500, 100, 500)],
            'activation': ['tanh', 'relu', 'logistic', 'identity'],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'alpha': loguniform(1e-5, 1e-1),
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': loguniform(1e-4, 1e-1),
            'beta_1': loguniform(1e-3, 0.9),
            'beta_2': loguniform(1e-3, 0.999),
            'epsilon': loguniform(1e-8, 1e-1)
        }
    },
    'SVC': {
        'model': SVC(probability=True),
        'params': {
            'C': expon(scale=100),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'] + list(expon(scale=0.001).rvs(200)),
            'degree': randint(1, 10),  # Only used if kernel is 'poly'
            'coef0': uniform(0.0, 5.0),  # Independent term in kernel function. Used in 'poly' and 'sigmoid'.
            'shrinking': [True, False],
            'tol': uniform(1e-4, 1e-2),
            'class_weight': [None, 'balanced'],
            'decision_function_shape': ['ovo', 'ovr'],
            'break_ties': [True, False],
            'random_state': randint(0, 100)
        }
    }
}
