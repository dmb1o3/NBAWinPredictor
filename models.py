import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from data_collector import gather_data_for_model
from sklearn.pipeline import Pipeline


# https://www.youtube.com/watch?v=egTylm6C2is


def get_best_parameters(model, param_grid, x_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.2f}")

    return best_params


def run_model(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error: {mse:.2f}")
    """
    # Example of printing predictions
    print("Predictions:")
    for i in range(len(x_test)):
        print(f"Features: {x_test[i]}, Actual: {y_test[i]}, Predicted: {y_pred[i]}")
    """
    return model


def logistic_regression(x, y):
    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    # Max_iter settings for saga solvers. Other models can go lower but saga solvers will through convergence errors
    sag_max_iter_start = 1400
    sag_max_iter_end = 4000
    sag_max_iter_step = 200

    # Max_iter settings for sag, liblinear, newton-cholesky, newton-cg and lbfgs solvers
    max_iter_start = 100
    max_iter_end = 1000
    max_iter_step = 100

    # Best settings on 2020 - 2023 is {'max_iter': 150, 'penalty': 'l1', 'solver': 'liblinear'}
    # Best cross-validation score: 0.60
    # Accuracy: 0.65
    # Mean squared error: 0.35
    param_grid = [
        {
            'solver': ["saga"],
            'penalty': ['l1', 'l2', None],
            'l1_ratio': [None],
            'max_iter': range(sag_max_iter_start, sag_max_iter_end, sag_max_iter_step)
        },
        {
            'solver': ["saga"],
            'penalty': ['elasticnet'],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': range(sag_max_iter_start, sag_max_iter_end, sag_max_iter_step)
        },
        {
            'solver': ["sag"],
            'penalty': ['l2', None],
            'l1_ratio': [None],
            'max_iter': range(sag_max_iter_start, sag_max_iter_end, sag_max_iter_end)
        },
        {
            'solver': ["liblinear"],
            'penalty': ['l1', 'l2'],
            'l1_ratio': [None],
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step)
        },
        {
            'solver': ["newton-cholesky", "newton-cg", "lbfgs"],
            'penalty': ['l2', None],
            'l1_ratio': [None],
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step)
        },
    ]

    print("LogisticRegression")
    best_params = get_best_parameters(LogisticRegression(), param_grid, x_train, y_train)
    best_solver = best_params['solver']
    best_penalty = best_params['penalty']
    best_l1_ratio = best_params['l1_ratio']
    best_max_iter = best_params['max_iter']
    model = LogisticRegression(penalty=best_penalty, solver=best_solver, l1_ratio=best_l1_ratio, max_iter=best_max_iter)

    model = run_model(x_train, x_test, y_train, y_test, model)

    print(model.coef_, model.intercept_)


def random_forest(x, y):
    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'n_estimators': range(10, 30, 10),
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': list(range(20, 60, 20)),  # + [None],
        'max_features': ["sqrt", "log2", None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3, 4],
        'bootstrap': [True, False]
    }
    print("RandomForestClassifier")
    # Set up data for testing and training
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    # Find best parameters for model
    best_params = get_best_parameters(RandomForestClassifier(), param_grid, x_train, y_train)
    best_n_estimators = best_params['n_estimators']
    best_criterion = best_params['criterion']
    best_max_depth = best_params['max_depth']
    best_max_features = best_params['max_features']
    best_min_samples_split = best_params['min_samples_split']
    best_min_samples_leaf = best_params['min_samples_leaf']
    best_bootstrap = best_params['bootstrap']

    # Run mode on best parameters
    model = RandomForestClassifier(n_estimators=best_n_estimators, criterion=best_criterion, max_depth=best_max_depth,
                                   max_features=best_max_features, min_samples_split=best_min_samples_split,
                                   min_samples_leaf=best_min_samples_leaf, bootstrap=best_bootstrap)
    model.fit(x_train, y_train)
    # @TODO extract features used from RF
    selected_features = x.columns[model.support_]
    print(selected_features)
    run_model(x_train, x_test, y_train, y_test, model)



def gradient_boosting(x, y):
    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'loss': ["exponential", "log_loss"],
        'learning_rate': [0.3, 0.5, 0.7],
        'n_estimators': list(range(50, 100, 50)),
        'criterion': ["friedman_mse", "squared_error"],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
        'max_depth': [3, 7]  # [None]
    }
    print("GradientBoostingClassifier")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    best_params = get_best_parameters(GradientBoostingClassifier(), param_grid, x_train, y_train)
    best_loss = best_params['loss']
    best_learning_rate = best_params['learning_rate']
    best_n_estimators = best_params['n_estimators']
    best_criterion = best_params['criterion']
    best_min_samples_split = best_params['min_samples_split']
    best_min_samples_leaf = best_params['min_samples_leaf']
    best_max_depth = best_params['max_depth']

    model = GradientBoostingClassifier(loss=best_loss, learning_rate=best_learning_rate, n_estimators=best_n_estimators,
                                       criterion=best_criterion, min_samples_split=best_min_samples_split,
                                       min_samples_leaf=best_min_samples_leaf, max_depth=best_max_depth)

    run_model(x_train, x_test, y_train, y_test, model)


def svc(x, y):
    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    max_iter_start = -1  # -1 is unlimited
    max_iter_end = 10
    max_iter_step = 11

    # Set up param grid
    param_grid = [
        {
            'kernel': ["linear", "rbf", "sigmoid", "precomputed"],
            'gamma': ["scale"],
            'C': [0.5, 1, 1.5],
            'degree': [3],
            "coef0": [0.0],
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step)

        },
        {
            'kernel': ["poly"],
            'gamma': ["scale"],
            'C': [0.5, 1, 1.5],
            'degree': range(1, 5, 1),
            "coef0": [0.0],
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step)
        },
        {
            'kernel': ["poly", "rbf", "sigmoid"],
            'gamma': ["scale", "auto", 0.3, 0.6],
            'C': [0.5, 1, 1.5],
            'degree': [3],
            "coef0": [0.0],
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step)
        },
        {
            'kernel': ["poly", "rbf", "sigmoid"],
            'gamma': ["scale", "auto", 0.3, 0.6],
            'C': [0.5, 1, 1.5],
            'degree': [3],
            "coef0": [0.1, 0.4, 0.7],
            'max_iter': range(max_iter_start, max_iter_end, max_iter_step)
        },

    ]
    print("SVC")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    best_params = get_best_parameters(SVC(), param_grid, x_train, y_train)
    best_kernel = best_params['kernel']
    best_gamma = best_params['gamma']
    best_C = best_params['C']
    best_degree = best_params['degree']
    best_coef0 = best_params['coef0']
    best_max_iter= best_params['max_iter']

    model = SVC(kernel=best_kernel, gamma=best_gamma, C=best_C, degree=best_degree, coef0=best_coef0,
                max_iter=best_max_iter)

    run_model(x_train, x_test, y_train, y_test, model)


def knn(x, y):
    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    param_grid = {
        'n_neighbors': range(2, 20, 6),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    print("KNN")
    # Get best parameters for model
    best_params = get_best_parameters(KNeighborsClassifier(), param_grid, x_train, y_train)

    # Get best values
    best_k = best_params['n_neighbors']
    best_weights = best_params['weights']
    best_p = best_params['p']
    best_algorithm = best_params['algorithm']
    best_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, p=best_p, algorithm=best_algorithm)
    # Run model
    run_model(x_train, x_test, y_train, y_test, best_model)


def bet_on_home_team(results):
    # Get number of wins
    home_team_wins = 0
    for result in results:
        if result == 1:
            home_team_wins += 1
    # Divide by number of games
    pct = home_team_wins / results.shape[0]
    return str(pct)


def main():
    x, y = gather_data_for_model(["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"])
    print("If you were to just bet on the home team over these seasons your accuracy would be " + bet_on_home_team(y))
    #logistic_regression(x, y)
    #svc(x, y)
    #knn(x, y)
    random_forest(x, y)
    #gradient_boosting(x, y)


if __name__ == "__main__":
    main()
