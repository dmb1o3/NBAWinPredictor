import backend.models.model_data_collector as mdc
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SequentialFeatureSelector
from backend.NBA_data_collector import handle_year_input
from xgboost import XGBRegressor
import pandas as pd
import SQL.SQL_data_collector as sdc


RANDOM_STATE = 100


# https://www.youtube.com/watch?v=egTylm6C2is

#  TODO add way to save and compare model stats based on
#  years used, settings for years and settings for model probably some type of chart

def check_regression_classification(df):
    # Determine actual and predicted winners
    df["Actual_Winner"] = df.apply(
        lambda row: "HOME" if row["Actual_HOME_TEAM_PTS"] > row["Actual_AWAY_TEAM_PTS"] else "AWAY", axis=1)
    df["Predicted_Winner"] = df.apply(
        lambda row: "HOME" if row["Predicted_HOME_TEAM_PTS"] > row["Predicted_AWAY_TEAM_PTS"] else "AWAY", axis=1)

    # Compute accuracy of win prediction
    df["Correct_Winner_Prediction"] = df["Actual_Winner"] == df["Predicted_Winner"]
    win_accuracy = df["Correct_Winner_Prediction"].mean() * 100
    print(win_accuracy)



def apply_sfs_model(model, x_train, y_train, x_test, sfs_settings):
    """
    Will run a model and return the best features using Sequential feature selection.

    :param model: Model to find best features for
    :param x_train: Training data for model
    :param y_train: Outcome of training data
    :param x_test: Testing data for model
    :param sfs_settings: Dictionary containing settings for sfs
    :return: A tuple containing transformed training data and then the transformed testing data
    """
    sfs = SequentialFeatureSelector(model, n_features_to_select=sfs_settings["n_features"], n_jobs=-1,
                                    direction=sfs_settings["direction"])
    # Fit to training data
    sfs.fit(x_train, y_train)
    print("Features selected " + str(sfs.transform(x_train).shape[1]))
    # Transform training and test data and return results
    return sfs.transform(x_train), sfs.transform(x_test)


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


def run_model(x_train, x_test, y_train, y_test, model, not_classifier, settings):
    """
    Will run model on data given

    :param x_train: Array containing training data
    :param x_test:  Array containing testing data
    :param y_train: Array containing target values for training data
    :param y_test:  Array containing target values for testing data
    :param model:   Scikit-learn model to run on data
    :param not_classifier: Boolean to know if model is classification or regression. Determine if we calc and print MSE
    :param settings: String containing settings used for data collection and preparing
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
    if not_classifier:
        print(f"Mean squared error: {mse:.2f}")
    print("Testing data Classification Report")
    print(class_report)
    print(settings)

    """
    # Printing predictions
    print("Predictions:")
    for i in range(0, 10):
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
    """
    return model


def logistic_regression(x_train, x_test, y_train, y_test, sfs_settings, settings):
    # Max_iter settings for saga solvers. Other models can go lower but saga solvers will through convergence errors
    sag_max_iter_start = 1000
    sag_max_iter_end = 4000
    sag_max_iter_step = 200

    # Max_iter settings for sag, liblinear, newton-cholesky, newton-cg and lbfgs solvers
    max_iter_start = 1000
    max_iter_end = 4000
    max_iter_step = 200

    # Best settings on 2020-2023 is {'max_iter': 150, 'penalty': 'l1', 'solver': 'liblinear'}
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
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(LogisticRegression(), x_train, y_train, x_test, sfs_settings)

    best_params = get_best_parameters(LogisticRegression(), param_grid, x_train, y_train, )
    best_solver = best_params['solver']
    best_penalty = best_params['penalty']
    best_l1_ratio = best_params['l1_ratio']
    best_max_iter = best_params['max_iter']
    model = LogisticRegression(penalty=best_penalty, solver=best_solver, l1_ratio=best_l1_ratio, max_iter=best_max_iter)

    model = run_model(x_train, x_test, y_train, y_test, model, False, settings)


def ridge_classification(x_train, x_test, y_train, y_test, sfs_settings, settings):
    param_grid = [
        {
            'solver': ["sag", "saga"],
            'alpha': [0.1, 0.5, 1.5],
            'max_iter': range(1000, 4000, 200),
            'positive': [False],
        },
        {
            'solver': ["auto", "svd", "cholesky", "lsqr", "sparse_cg"],
            'alpha': [0.1, 0.5, 1.5],
            'max_iter': range(1000, 4000, 200),
            'positive': [False],
        },
        {
            'solver': ["lbfgs"],
            'alpha': [0.1, 0.5, 1.5],
            'max_iter': range(1000, 4000, 200),
            'positive': [True],

        }
    ]

    print("\nRidge Classifier")
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(RidgeClassifier(), x_train, y_train, x_test, sfs_settings)

    best_params = get_best_parameters(RidgeClassifier(), param_grid, x_train, y_train)
    best_solver = best_params['solver']
    best_alpha = best_params['alpha']
    best_max_iter = best_params['max_iter']
    best_positive = best_params['positive']
    model = RidgeClassifier(solver=best_solver, alpha=best_alpha, max_iter=best_max_iter, positive=best_positive)

    model = run_model(x_train, x_test, y_train, y_test, model, False, settings)


def random_forest_regression(x_train, x_test, y_train, y_test, sfs_settings, settings):
    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'n_estimators': range(10, 30, 10),
        'criterion': ["squared_error", "absolute_error", "friedman_mse"],  # Regression criteria
        'max_depth': list(range(20, 180, 80)),
        'max_features': ["sqrt", "log2"],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 4, 10, 20],
        'bootstrap': [True, False]
    }

    print("\nRandomForestRegressor")

    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(RandomForestRegressor(), x_train, y_train, x_test, sfs_settings)

    # Find best parameters for model
    best_params = get_best_parameters(RandomForestRegressor(), param_grid, x_train, y_train)

    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        criterion=best_params['criterion'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        bootstrap=best_params['bootstrap']
    )

    # Train model on both outputs (home & away scores)
    model.fit(x_train, y_train)
    # Try predicting
    y_pred = model.predict(x_test)

    predictions_df = pd.DataFrame({
        'Actual_HOME_TEAM_PTS': y_test['HOME_TEAM_PTS'],
        'Predicted_HOME_TEAM_PTS': y_pred[:, 0],  # Predictions for HOME_TEAM_PTS
        'Actual_AWAY_TEAM_PTS': y_test['AWAY_TEAM_PTS'],
        'Predicted_AWAY_TEAM_PTS': y_pred[:, 1]  # Predictions for AWAY_TEAM_PTS
    })

    predictions_df.to_csv('random_forest_regerssion_predictions.csv', index=False)

    check_regression_classification(predictions_df)


def gradient_boosting_regression(x_train, x_test, y_train, y_test, sfs_settings, settings):
    print("\nGradient Boosting Regressor")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(GradientBoostingRegressor(), x_train, y_train, x_test, sfs_settings)

    best_params = get_best_parameters(GradientBoostingRegressor(), param_grid, x_train, y_train)

    model = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample']
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    predictions_df = pd.DataFrame({
        'Actual_HOME_TEAM_PTS': y_test['HOME_TEAM_PTS'],
        'Predicted_HOME_TEAM_PTS': y_pred[:, 0],
        'Actual_AWAY_TEAM_PTS': y_test['AWAY_TEAM_PTS'],
        'Predicted_AWAY_TEAM_PTS': y_pred[:, 1]
    })

    predictions_df.to_csv('gradient_boosting_regression_predictions.csv', index=False)
    check_regression_classification(predictions_df)


def xgboost_regression(x_train, x_test, y_train, y_test, sfs_settings, settings):
    print("\nXGBoost Regressor")

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(XGBRegressor(), x_train, y_train, x_test, sfs_settings)

    best_params = get_best_parameters(XGBRegressor(), param_grid, x_train, y_train)

    model = XGBRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample']
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    predictions_df = pd.DataFrame({
        'Actual_HOME_TEAM_PTS': y_test['HOME_TEAM_PTS'],
        'Predicted_HOME_TEAM_PTS': y_pred[:, 0],
        'Actual_AWAY_TEAM_PTS': y_test['AWAY_TEAM_PTS'],
        'Predicted_AWAY_TEAM_PTS': y_pred[:, 1]
    })

    predictions_df.to_csv('xgboost_regression_predictions.csv', index=False)
    check_regression_classification(predictions_df)


def lasso_regression(x_train, x_test, y_train, y_test, sfs_settings, settings):
    print("\nLasso Regression")

    param_grid = {
        'alpha': [0.01, 0.1, 1.0, 10],  # Regularization strength
        'max_iter': [1000, 5000, 10000]  # Max iterations for convergence
    }

    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(Lasso(), x_train, y_train, x_test, sfs_settings)

    best_params = get_best_parameters(Lasso(), param_grid, x_train, y_train)

    model = Lasso(
        alpha=best_params['alpha'],
        max_iter=best_params['max_iter']
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    predictions_df = pd.DataFrame({
        'Actual_HOME_TEAM_PTS': y_test['HOME_TEAM_PTS'],
        'Predicted_HOME_TEAM_PTS': y_pred[:, 0],
        'Actual_AWAY_TEAM_PTS': y_test['AWAY_TEAM_PTS'],
        'Predicted_AWAY_TEAM_PTS': y_pred[:, 1]
    })

    predictions_df.to_csv('lasso_regression_predictions.csv', index=False)
    check_regression_classification(predictions_df)


def random_forest(x_train, x_test, y_train, y_test, sfs_settings, settings):
    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'n_estimators': range(10, 50, 10),
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': list(range(20, 180, 80)),  # + [None],
        'max_features': ["sqrt", "log2"],  # , None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 4, 10, 20],
        'bootstrap': [True, False]
    }
    print("\nRandomForestClassifier")
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(RandomForestClassifier(), x_train, y_train, x_test, sfs_settings)

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
    run_model(x_train, x_test, y_train, y_test, model, False, settings)


def gradient_boosting(x_train, x_test, y_train, y_test, sfs_settings, settings):
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
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(GradientBoostingClassifier(), x_train, y_train, x_test, sfs_settings)

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

    run_model(x_train, x_test, y_train, y_test, model, False, settings)


def knn(x_train, x_test, y_train, y_test, sfs_settings, settings):
    param_grid = {
        'n_neighbors': range(2, 44, 6),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    print("\nKNN")
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(KNeighborsClassifier(), x_train, y_train, x_test, sfs_settings)

    # Get best parameters for model
    best_params = get_best_parameters(KNeighborsClassifier(), param_grid, x_train, y_train)

    # Get best values
    best_k = best_params['n_neighbors']
    best_weights = best_params['weights']
    best_p = best_params['p']
    best_algorithm = best_params['algorithm']
    best_model = KNeighborsClassifier(n_neighbors=best_k, weights=best_weights, p=best_p, algorithm=best_algorithm)
    # Run model
    run_model(x_train, x_test, y_train, y_test, best_model, False, settings)


def svc(x_train, x_test, y_train, y_test, sfs_settings, settings):
    param_grid = [
        {
            'kernel': ["linear", "poly", "rbf", "sigmoid"],
            'degree': range(1, 5, 1),
            'gamma': ["scale", "auto", 0.2, 0.5, 0.7],
            'coef0': [0.0, 0.2, 0.5, 0.7],
            'shrinking': [True, False],

        },
    ]
    print("\nSVC")
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(SVC(), x_train, y_train, x_test, sfs_settings)

    # Get best parameters for model
    best_params = get_best_parameters(SVC(), param_grid, x_train, y_train)

    # Get best values
    best_kernel = best_params['kernel']
    best_degree = best_params['degree']
    best_gamma = best_params['gamma']
    best_coef0 = best_params['coef0']
    best_shrinking = best_params['shrinking']

    best_model = SVC(kernel=best_kernel, degree=best_degree, gamma=best_gamma, coef0=best_coef0,
                     shrinking=best_shrinking)
    # Run model
    run_model(x_train, x_test, y_train, y_test, best_model, False, settings)


def gaussian_process_classifier(x_train, x_test, y_train, y_test, sfs_settings, settings):
    print("\nGaussian Process Classifier")
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(GaussianProcessClassifier(), x_train, y_train, x_test, sfs_settings)

    best_model = GaussianProcessClassifier()
    # Run model
    run_model(x_train, x_test, y_train, y_test, best_model, False, settings)


def gaussian_naive_bayes(x_train, x_test, y_train, y_test, sfs_settings, settings):
    print("\nGaussian Naive bBayes")
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(GaussianNB(), x_train, y_train, x_test, sfs_settings)

    best_model = GaussianNB()
    # Run model
    run_model(x_train, x_test, y_train, y_test, best_model, False, settings)


def get_sfs_settings(sfs_settings):
    print("How many features would you like to select? Type auto to use auto setting")
    n_features = input().lower()
    if n_features != "auto":
        try:
            n_features = int(n_features)
        except TypeError:
            raise TypeError("Only integer or auto allowed as input")

    sfs_settings["n_features"] = n_features

    print("\nDo you want to perform selection forward or backwards?")
    print("1. Forward")
    print("2. Backward")
    user_input = input()
    if user_input != "1":
        sfs_settings["direction"] = "forward"
    else:
        sfs_settings["direction"] = "backward"


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


def print_get_years_to_examine():
    years_to_examine = input("What years would you like to examine? If multiple just type them with a space like "
                                 "\"2020 2021 2022\" or use a dash like 2020-2022: ")
    years_to_examine = handle_year_input(years_to_examine)

    return years_to_examine

def main():
    # @TODO Look into implementing validation set
    years_to_examine = print_get_years_to_examine()

    # Start setting up setings
    settings = "Done using data from " + str(years_to_examine)

    # Ask user what type of data from years they want to use
    print("\nWhat type of data would you like to feed models?")
    print("1. Averaged team stats")
    print("2. Averaged advanced team stats ")
    print("3. Averaged team and advanced team stats")
    print("4. Averaged Player Stats")
    # Get users choice and lowercase it to make q/Q the same
    user_selection = input("Enter number associated with choice (Enter q to exit): ").strip()
    user_selection = user_selection.lower()

    options = {
        '1': lambda: mdc.get_averaged_team_stats(years_to_examine),
        '2': lambda: mdc.get_averaged_adv_team_stats(years_to_examine),
        '3': lambda: mdc.get_averaged_team_and_adv_team_stats(years_to_examine),
        '4': lambda: mdc.get_averaged_player_stats(years_to_examine),
        'q': exit,
    }
    # Call menu option if valid if not let user know how to properly use menu
    if user_selection in options:
        print("Gathering Data")
        x = options[user_selection]()
    else:
        exit()

    # Ask user what type of models to run
    print("\nSelect the type of model to run:")
    print("1. Classification: Predict if the home team wins or loses.")
    print("2. Regression: Estimate the final point difference and classify based on that.")

    classification = input("Enter number associated with choice (Enter q to exit): ").strip()
    classification = True if classification == "1" else False

    if classification:
        y = sdc.get_home_team_won(x["GAME_ID"].to_list())
    else:
        y = sdc.get_home_away_points(x["GAME_ID"].to_list())

    # Ensure that the target is ordered same as data
    ordered_game_ids = x['GAME_ID'].tolist()
    game_id_to_index = {game_id: index for index, game_id in enumerate(ordered_game_ids)}
    y['sort_order'] = y['GAME_ID'].map(game_id_to_index)
    y = y.sort_values('sort_order').reset_index(drop=True)
    y = y.drop(columns=['sort_order'])
    # Verify the order matches
    is_matched = (x['GAME_ID'] == y['GAME_ID']).all()

    # Drop GAME_ID from data and target
    x = x.drop(["GAME_ID"], axis=1)
    y = y.drop(["GAME_ID"], axis=1)

    # Ask user how we should split data
    print("\nFor the seasons entered do you want randomly split data or go sequentially?")
    print("1. Randomly Split Data")
    print("2. Sequentially")
    user_answer = input("")
    if user_answer == "1":
        print("\nRandomly Splitting Data\n")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
        settings += ", with Randomly Split Data"
    else:
        print("\nSplitting Data Sequential\n")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False,
                                                            random_state=RANDOM_STATE)
        settings += ", with Sequentially Split Data"

    # Ask user if we should balance data
    print("\nDo you want to balance the amount of games by class?")
    print("1. Balance data")
    print("2. Do not balance Data")
    user_answer = input("")
    # If they would like to balance data then balance it
    if user_answer == "1":
        # Ask how they want to balance
        print("\nHow do you want to balance classes?")
        print("1. Oversample Minority")
        print("2. Undersample Majority")
        user_answer = input("")
        if user_answer == "1":
            ros = RandomOverSampler(random_state=RANDOM_STATE)
            x_train, y_train = ros.fit_resample(x_train, y_train)
            settings += ", with Balanced Classes by Over Sampling Minority"

        else:
            rus = RandomUnderSampler(random_state=RANDOM_STATE)
            x_train, y_train = rus.fit_resample(x_train, y_train)
            settings += ", with Balanced Classes by Under Sampling Majority"
    else:
        settings += ", with Unbalanced Classes"

    # Ask user if we should scale data
    print("\nDo you want to scale the data?")
    print("1. Scale data")
    print("2. Do not Scale Data")
    user_answer = input("")
    # If they would like to scale data then scale it
    if user_answer == "1":
        # Scale data
        print("\nScaling data")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        settings += ", with Scaled Data"
    else:
        settings += ", with Unscaled Data"

    # Ask user if we should apply sfs
    print("\nDo you want to apply sequential feature selection?")
    print("1. Apply SFS")
    print("2. Do not apply SFS")
    user_answer = input("")
    sfs_settings = {}
    if user_answer == "1":
        sfs_settings["apply_sfs"] = True
        get_sfs_settings(sfs_settings)
        settings += ", with SFS applied"
    else:
        sfs_settings["apply_sfs"] = False
        settings += ", with SFS not applied"

    # Find out the baseline by calculating odds home team win
    if classification:
        print("\nIf you were to just bet on the home team over these seasons your accuracy would be " + bet_on_home_team(y))

    # Run models
    if classification:
        logistic_regression(x_train, x_test, y_train, y_test, sfs_settings, settings)
        ridge_classification(x_train, x_test, y_train, y_test, sfs_settings, settings)
        random_forest(x_train, x_test, y_train, y_test, sfs_settings, settings)
        gaussian_process_classifier(x_train, x_test, y_train, y_test, sfs_settings, settings)
        knn(x_train, x_test, y_train, y_test, sfs_settings, settings)
        gradient_boosting(x_train, x_test, y_train, y_test, sfs_settings, settings)
        svc(x_train, x_test, y_train, y_test, sfs_settings, settings)
    else:
        #gradient_boosting_regression(x_train, x_test, y_train, y_test, sfs_settings, settings)
        random_forest_regression(x_train, x_test, y_train, y_test, sfs_settings, settings)
        xgboost_regression(x_train, x_test, y_train, y_test, sfs_settings, settings)
        #lasso_regression(x_train, x_test, y_train, y_test, sfs_settings, settings)


if __name__ == "__main__":
    main()
