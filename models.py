import model_data_collector as dc
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SequentialFeatureSelector
from NBA_data_collector import handle_year_input

RANDOM_STATE = 100


# https://www.youtube.com/watch?v=egTylm6C2is

#  TODO add way to save and compare model stats based on
#  years used, settings for years and settings for model probably some type of chart


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
    sag_max_iter_start = 500
    sag_max_iter_end = 1500
    sag_max_iter_step = 50

    # Max_iter settings for sag, liblinear, newton-cholesky, newton-cg and lbfgs solvers
    max_iter_start = 100
    max_iter_end = 1000
    max_iter_step = 50

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
    if sfs_settings["apply_sfs"]:
        x_train, x_test = apply_sfs_model(RidgeClassifier(), x_train, y_train, x_test, sfs_settings)

    best_params = get_best_parameters(RidgeClassifier(), param_grid, x_train, y_train)
    best_solver = best_params['solver']
    best_alpha = best_params['alpha']
    best_max_iter = best_params['max_iter']
    best_positive = best_params['positive']
    model = RidgeClassifier(solver=best_solver, alpha=best_alpha, max_iter=best_max_iter, positive=best_positive)

    model = run_model(x_train, x_test, y_train, y_test, model, False, settings)



def random_forest(x_train, x_test, y_train, y_test, sfs_settings, settings):
    # Set up param_grid for hyperparameter tuning
    param_grid = {
        'n_estimators': range(10, 50, 10),
        'criterion': ["gini", "entropy", "log_loss"],
        'max_depth': list(range(20, 100, 20)),  # + [None],
        'max_features': ["sqrt", "log2"],  # , None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4],
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


def main():
    # @TODO Look into implementing validation set
    # @TODO test out model using training data in order for certain years maybe 2020-2022 and testing data as 2023
    # Default years to use as data
    years_to_examine = ["2020", "2021", "2022", "2023"]

    # Ask users what years to use for NBA data
    print("Do you want to use current years or input new years? (Enter number of choice)")
    print("Current years: " + str(years_to_examine))
    print("1. Use current years")
    print("2. Input new years")
    user_answer = input("")
    # If they would like to set their own change to desired years
    if user_answer == "2":
        years_to_examine = input("What years would you like to examine? If multiple just type them with a space like "
                                 "\"2020 2021 2022\" ")
        years_to_examine = handle_year_input(years_to_examine)

    # Start setting up setings
    settings = "Done using data from " + str(years_to_examine)

    # Ask user what type of data from years they want to use
    options = {
        '1': lambda: dc.get_averaged_team_stats(years_to_examine),
        'q': exit,
    }
    print("\nWhat type of data would you like to feed models?")
    print("1. Averaged team stats")
    # Get users choice and lowercase it to make q/Q the same
    user_selection = input("Enter number associated with choice (Enter q to exit): ")
    user_selection = user_selection.lower()

    # Call menu option if valid if not let user know how to properly use menu
    if user_selection in options:
        x, y = options[user_selection]()
    else:
        dc.invalid_option(len(options))

    # Get data, target values and features of data

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
    print("\nIf you were to just bet on the home team over these seasons your accuracy would be " + bet_on_home_team(y))

    # Run models
    logistic_regression(x_train, x_test, y_train, y_test, sfs_settings, settings)
    ridge_classification(x_train, x_test, y_train, y_test, sfs_settings, settings)
    random_forest(x_train, x_test, y_train, y_test, sfs_settings, settings)
    gaussian_process_classifier(x_train, x_test, y_train, y_test, sfs_settings, settings)
    knn(x_train, x_test, y_train, y_test, sfs_settings, settings)
    gradient_boosting(x_train, x_test, y_train, y_test, sfs_settings, settings)
    svc(x_train, x_test, y_train, y_test, sfs_settings, settings)


if __name__ == "__main__":
    main()
