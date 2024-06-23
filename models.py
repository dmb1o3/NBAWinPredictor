import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from data_collector import gather_data_for_model


# https://www.youtube.com/watch?v=egTylm6C2is


def run_model(x_train, x_test, y_train, y_test, model):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    """
    # Example of printing predictions
    print("Predictions:")
    for i in range(len(x_test)):
        print(f"Features: {x_test[i]}, Actual: {y_test[i]}, Predicted: {y_pred[i]}")
    """


def logistic_regression(years):
    x, y = gather_data_for_model(years)
    # Scale data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    model = LogisticRegression(solver="saga", max_iter=10000)

    run_model(x_train, x_test, y_train, y_test, model)



def KNN(years):
    x, y = gather_data_for_model(years)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

    param_grid = {'n_neighbors': range(1, 30)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    best_k = grid_search.best_params_['n_neighbors']
    print("Best K " + str(best_k))

    model = KNeighborsClassifier(n_neighbors=best_k)
    run_model(x_train, x_test, y_train, y_test, model)


def main():
    logistic_regression(["2021", "2022", "2023"])
    KNN(["2020","2021", "2022", "2023"])


if __name__ == "__main__":
    main()