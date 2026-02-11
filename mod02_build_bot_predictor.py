# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
import numpy as np

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.001,
        n_estimators=2000,
        max_depth=2,
        subsample=.75,
        min_samples_leaf=1,
        min_samples_split=50,
        random_state=seed
    )
    model.fit(X, y)
    return model

def train_model_RandomCV(X, y, seed=seed):
    param_dist = {
        'learning_rate' : uniform(0.01, 0.3),
        'n_estimators' : randint(100, 1000),
        'max_depth' : randint(2, 10),
        'subsample' : uniform(.1, .9),
        'min_samples_leaf' : randint(1, 25),
        'min_samples_split' : randint(2, 100),
        'min_weight_fraction_leaf' : uniform(0.0, 0.05),
        'max_leaf_nodes' : randint(2, 100),
        'validation_fraction' : uniform(0.1, 0.3),
        'n_iter_no_change' : randint(5, 30),
    }

    model = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=seed),
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring='accuracy',
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X, y)
    return model

def train_model_GridCV(X, y, seed=seed):
    param_grid = {
        "learning_rate" : [0.01, 0.05, 0.1],
        "n_estimators" : [200, 500, 1000],
        'max_depth' : [2, 4, 6],
        "subsample" : [.5, .75, 1],
        "min_samples_leaf": [1, 5, 10],
        "min_samples_split" : [2, 10, 50]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    model = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=seed),
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1
    )

    model.fit(X, y)
    return model



