import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from data_collector import gather_data_for_model


# https://www.youtube.com/watch?v=egTylm6C2is


def logistic_regression(year):
    x, y = gather_data_for_model(year)
    print(x)
    print(y)


def main():
    logistic_regression("2023")

if __name__ == "__main__":
    main()