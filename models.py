import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    fighters_fname = "fighters_preprocessed.csv"
    fighters = pd.read_csv(fighters_fname)
    print(fighters)
    
    # Predicting winning_pct
    target = "winning_pct"
    X = fighters.drop(target, axis=1)
    Y = fighters[target].copy()
    params = {"max_depth": [2,4,8], "max_leaf_nodes": [4,16,64]}
    grid_cv = GridSearchCV(RandomForestRegressor(), params, cv=5, n_jobs=-1, scoring="explained_variance")
    grid_cv.fit(X, Y)
    cv_fvaf = grid_cv.best_score_
    rfr = grid_cv.best_estimator_
    
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
