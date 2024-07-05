import numpy as np
from scipy.stats import loguniform, expon, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


POSSIBLE_CLASSIFIERS = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': loguniform(1e-4, 1e+4),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'penalty': ['l2', None],
            'class_weight': [None, 'balanced'],
            'l1_ratio': np.linspace(0, 1, 10),
            'warm_start': [True, False]
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': randint(5, 1000),
            'max_depth': list(range(2, 20, 2)),
            'min_samples_split': randint(2, 50),
            'min_samples_leaf': randint(1, 20),
            'max_features': list(range(1, 100)) + ['sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],
            'min_impurity_decrease': np.linspace(0.0, 0.1, 10),
            'bootstrap': [True, False]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': randint(10, 200),
            'learning_rate': loguniform(0.05, 1),
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
    },
    'MultinomialNB': {
        'model': MultinomialNB(),
        'params': {
            'alpha': np.linspace(0.1, 2, 20),  # Additive (Laplace/Lidstone) smoothing parameter
            'fit_prior': [True, False]  # Whether to learn class prior probabilities or not.
        }
    },
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],  # Function to measure the quality of a split
            'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
        }
    },
    'KNeighborsClassifier': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to use
            'weights': ['uniform', 'distance'],  # Weight function used in prediction
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
            'p': [1, 2]  # Power parameter for the Minkowski metric
        }
    },
    'SGDClassifier': {
        'model': SGDClassifier(random_state=42, max_iter=1000),
        'params': {
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],  # Loss function to be used
            'penalty': ['l2', 'l1', 'elasticnet'],  # The penalty (aka regularization term) to be used
            'alpha': np.logspace(-6, -1, 10),  # Constant that multiplies the regularization term
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],  # Learning rate schedule
            'eta0': [0.01, 0.1, 1]  # Initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules
        }
    },
    'Perceptron': {
        'model': Perceptron(random_state=42, max_iter=1000),
        'params': {
            'penalty': [None, 'l2', 'l1', 'elasticnet'],  # The penalty (aka regularization term) to be used
            'alpha': np.logspace(-6, -1, 10),  # Constant that multiplies the regularization term
            'fit_intercept': [True, False],  # Whether the intercept should be estimated or not
            'shuffle': [True, False],  # Whether or not the training data should be shuffled after each epoch
            'eta0': [0.01, 0.1, 1],  # Constant by which the updates are multiplied
            'warm_start': [True, False]  # Whether to reuse the solution of the previous call to fit as initialization
        }
    }
}
