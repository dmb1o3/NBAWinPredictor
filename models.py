import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from data_collector import get_data_for_model
from sklearn.pipeline import Pipeline


# https://www.youtube.com/watch?v=egTylm6C2is

# @TODO implement sequential feature selection and test out
# @TODO add way to save and compare model stats based on years used, setttings for years and settings for model probaly some type of chart

def get_best_parameters(model, param_grid, x_train, y_train):
    """
    Uses GridSearchCV to find optimal parameters for a model

    :param model: Scikit learn model we would like to test
    :param param_grid: A dictionary or list of dictionaries containing parameters for model to test
    :param x_train: 1d Array containing training data
    :param y_train: 1d Array contain target values for training data
    :return: Returns dictionary containing the best configuration of parameters based on param_grid
    """

    # Test model
    grid_search = GridSearchCV(model, param_grid)
    grid_search.fit(x_train, y_train)

    # Print out the best parameters for a model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score:.2f}")

    return best_params


def run_model(x_train, x_test, y_train, y_test, model, notClassifier):
    """
    Will run model on data given

    :param x_train: Array containing training data
    :param x_test:  Array containing testing data
    :param y_train: Array containing target values for training data
    :param y_test:  Array containing target values for testing data
    :param model:   Scikit-learn model to run on data
    :param notClassifier: Boolean to know if model is classification or regression. Determines if we calc and print MSE
    :return: Returns a scikit-learn model after being trained and tested
    """
    # Train model
    model.fit(x_train, y_train)
    # Test model
    y_pred = model.predict(x_test)

    # Get and print accuracy, mean squared error and classification report
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["Home Team Lost", "Home Team Won"])
    print(f"Accuracy: {accuracy:.2f}")
    if notClassifier:
        print(f"Mean squared error: {mse:.2f}")
    print(class_report)

    """
    # Printing predictions
    print("Predictions:")
    for i in range(0, 10):
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
    """
    return model


def logistic_regression(x_train, x_test, y_train, y_test, features):
    # Max_iter settings for saga solvers. Other models can go lower but saga solvers will through convergence errors
    sag_max_iter_start = 500
    sag_max_iter_end = 1500
    sag_max_iter_step = 50

    # Max_iter settings for sag, liblinear, newton-cholesky, newton-cg and lbfgs solvers
    max_iter_start = 100
    max_iter_end = 1000
    max_iter_step = 50

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
        }
    ]

    print("\nLogisticRegression")
    best_params = get_best_parameters(LogisticRegression(), param_grid, x_train, y_train,)
    best_solver = best_params['solver']
    best_penalty = best_params['penalty']
    best_l1_ratio = best_params['l1_ratio']
    best_max_iter = best_params['max_iter']
    model = LogisticRegression(penalty=best_penalty, solver=best_solver, l1_ratio=best_l1_ratio, max_iter=best_max_iter)

    model = run_model(x_train, x_test, y_train, y_test, model, False)

    # Set up data frame so easier to understand what features are being used
    d = pd.DataFrame({"FEATURES": features, "COEFFICIENT": model.coef_[0]})
    # Sort them by coefficients absolute value and then save
    d = d.sort_values(["COEFFICIENT"], key=abs, ascending=False)
    d.to_csv("data/models/Linear_Regression_Features_COEF.csv", index=False)


def ridge_classification(x_train, x_test, y_train, y_test, features):
    param_grid = [
        {
            'solver': ["sag", "saga"],
            'alpha': [0.1, 0.5, 1.5],
            'max_iter': range(800, 1500, 100),
            'positive': [False],
        },
        {
            'solver': ["auto", "svd", "cholesky", "lsqr", "sparse_cg"],
            'alpha': [0.1, 0.5, 1.5],
            'max_iter': range(50, 600, 50),
            'positive': [False],
        },
        {
            'solver': ["lbfgs"],
            'alpha': [0.1, 0.5, 1.5],
            'max_iter': range(50, 600, 50),
            'positive': [True],

        }
    ]

    print("\nRidge Classifier")
    best_params = get_best_parameters(RidgeClassifier(), param_grid, x_train, y_train)
    best_solver = best_params['solver']
    best_alpha = best_params['alpha']
    best_max_iter = best_params['max_iter']
    best_positive = best_params['positive']
    model = RidgeClassifier(solver=best_solver, alpha=best_alpha, max_iter=best_max_iter, positive=best_positive)

    model = run_model(x_train, x_test, y_train, y_test, model, False)

    # Set up data frame so easier to understand what features are being used
    d = pd.DataFrame({"FEATURES": features, "COEFFICIENT": model.coef_[0]})
    # Sort them by coefficients absolute value and then save
    d = d.sort_values(["COEFFICIENT"], key=abs, ascending=False)
    d.to_csv("data/models/Ridge_Classifier_Features_COEF.csv", index=False)


def random_forest(x_train, x_test, y_train, y_test):
    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'n_estimators': range(10, 50, 10),
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': list(range(20, 160, 20)),  # + [None],
        'max_features': ["sqrt", "log2"],  # , None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4],
        'bootstrap': [True, False]
    }
    print("\nRandomForestClassifier")

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
    """
    selected_features = model.decision_path(x_train)
    print(selected_features)
    """
    run_model(x_train, x_test, y_train, y_test, model, False)


def gradient_boosting(x_train, x_test, y_train, y_test):
    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'loss': ["exponential", "log_loss"],
        'learning_rate': [0.3, 0.5, 0.7],
        'n_estimators': list(range(20, 100, 20)),
        'criterion': ["friedman_mse", "squared_error"],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4],
        'max_depth': [2, 3, 7]  # [None]
    }
    print("\nGradientBoostingClassifier")
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

    run_model(x_train, x_test, y_train, y_test, model, False)


def knn(x_train, x_test, y_train, y_test):
    param_grid = {
        'n_neighbors': range(2, 44, 6),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    print("\nKNN")
    # Get best parameters for model
    best_params = get_best_parameters(KNeighborsClassifier(), param_grid, x_train, y_train)

    # Get best values
    best_k = best_params['n_neighbors']
    best_weights = best_params['weights']
    best_p = best_params['p']
    best_algorithm = best_params['algorithm']
    best_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, p=best_p, algorithm=best_algorithm)
    # Run model
    run_model(x_train, x_test, y_train, y_test, best_model, False)


def bet_on_home_team(results):
    """
    Given the results will calculate the odds of winning by betting on the home team. It does this by calculating all
    the wins, indicated by a 1, and dividing by the number of games played.

    :param results: An array with results of games 1 indicating a win
    :return: Will return a string with the float value calculated. 55% will return 0.55
    """

    # Get number of wins
    home_team_wins = 0
    for result in results:
        if result == 1:
            home_team_wins += 1
    # Divide by number of games
    pct = home_team_wins / results.shape[0]
    return str(pct)


def main():
    # @TODO test out model using training data in order for certain years maybe 2020-2022 and testing data as 2023
    # Default years to use as data
    years_to_examine = ["2020", "2021", "2022", "2023"]

    # Ask user if they would like to use default or set their own
    print("Do you want to use current years or input new years? (Enter number of choice)")
    print("Current years: " + str(years_to_examine))
    print("1. Use current years")
    print("2. Input new years")
    user_answer = input("")
    # If they would like to set their own change to desired years
    if user_answer == "2":
        years_to_examine = input("What years would you like to examine? If multiple just type them with a space like "
                                 "\"2020 2021 2022\" ").split()

    # Get data, target values and features of data
    x, y, features = get_data_for_model(years_to_examine)

    # Ask user if we should scale data
    print("\nDo you want to scale the data?")
    print("1. Scale data")
    print("2. Do not Scale Data")
    user_answer = input("")
    # If they would like to scale data then scale it
    if user_answer == "1":
        # Scale data
        print("\nScaling data\n")
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    # Ask user how we should split data
    print("For the seasons entered do you want randomly split data or go sequentially?")
    print("1. Randomly Split Data")
    print("2. Sequentially")
    user_answer = input("")
    if user_answer == "1":
        print("\nRandomly Splitting Data\n")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    else:
        print("\nSplitting Data Sequential\n")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=100)

    # Find out the baseline by calculating odds home team win
    print("If you were to just bet on the home team over these seasons your accuracy would be " + bet_on_home_team(y))

    # Run models
    logistic_regression(x_train, x_test, y_train, y_test, features)
    ridge_classification(x_train, x_test, y_train, y_test, features)
    random_forest(x_train, x_test, y_train, y_test)
    knn(x_train, x_test, y_train, y_test)
    gradient_boosting(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
