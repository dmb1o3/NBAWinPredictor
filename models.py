import numpy as np
import pandas as pd

# https://www.youtube.com/watch?v=egTylm6C2is

def run_knn():
    # Set up cross validation
    # Set random_state so that we can reproduce results
    # Can remove later if we want
    kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=45)


