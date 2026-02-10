# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    #this is used to train the model every time 
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=2,
        subsample=1,
        min_samples_leaf=1,
        random_state=seed
    )
    #where model finds patterns in X, or the training file
    #that are then used to test the files in Y
    model.fit(X, y)
    return model