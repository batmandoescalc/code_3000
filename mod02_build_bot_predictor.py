# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    
    """
    model = GradientBoostingClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        min_samples_leaf=15,
        min_samples_split=30,
        max_features="sqrt",
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )
    
    model.fit(X, y)
    return model
