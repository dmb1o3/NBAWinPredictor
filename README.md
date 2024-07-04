# Description of NBA Win Predictor
This was a project built to explore machine learning models using something I'm interested in. 
I am a huge Clippers fan who has been following the NBA for the last couple of years and thought it would help me
learn more about data science and the NBA. This project is still a work in progress, however, it is at a usable point if you
would like to try it out or do you own projects. 

In this project, I used python, pandas and sci kit learn. Python and pandas were used heavily to gather, prepare and 
clean the data. While scikit learn was used to set up the models and hyperparameter tuning

My baseline to compare the models was 55% as that was the average win rate if you just bet on the home team. 
So far, with the models implemented, I am able to get 60–65% depending on the model and how much data is used. 
With my best results coming from the 2020 to 2023 season of 65% accuracy with logistic regression. 


## Overview

### data_collector.py
This file is used to both download and/or prepare the data. 

#### Downloading Data
![Leauge Schedule Diagram.svg](README%20Diagrams%2FDownload%20Diagram.svg)

#### Preparing Data
[ADD DIAGRAM SHOWING DATA PROCESSING AFTER DOWNLOAD]


### models.py
This file contains the scikit learn models implemented with hyperparameter tuning. To change what years of data the 
model looks at change the variable years_to_examine at the top of the main function.
#### 1. Logistic Regression

```
Highest Accuracy: 0.65
Mean squared error: 0.35
Cross-validation score: 0.60
Parameters: {'max_iter': 150, 'penalty': 'l1', 'solver': 'liblinear'}
Done using data from 2020 - 2023, 5 players per team, 5 games back, 2 game buffer 
```

#### 2. Random Forest

```
Highest Accuracy: 0.65
Mean squared error: 0.35
Cross-validation score: 0.63
Parameters: {'bootstrap': True, 'criterion': 'log_loss', 'max_depth': 20, 'max_features': 'sqrt', 
             'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 40}
Done using data from 2020 - 2023, 5 players per team, 5 games back, 2 game buffer 
```

#### 3. KNN

```
Highest Accuracy: 0.62
Mean squared error: 0.38
Cross-validation score: 0.59
Parameters: {'algorithm': 'auto', 'n_neighbors': 26, 'p': 2, 'weights': 'distance'}
Done using data from 2020 - 2023, 5 players per team, 5 games back, 2 game buffer 
```

#### 4. Gradient Boosting

```
Highest Accuracy: 0.63
Mean squared error: 0.37
Cross-validation score: 0.62
Parameters: {'criterion': 'friedman_mse', 'learning_rate': 0.3, 'loss': 'exponential', 
             'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 40}
Done using data from 2020 - 2023, 5 players per team, 5 games back, 2 game buffer 
```


### league_data.py
[Link to repo where the file was taken](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguegamelog.md)

This file is used to easily pull data from NBA API. Specifically, it is used to pull the league schedule for a 
given year. Allowing us to know what games were played, the game ids for those games, what teams played and other basic information.


### box_score_data.py
[Link to repo where the file was taken](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/boxscoretraditionalv2.md)

This file is used to easily pull data from NBA API. Specifically, it is used to pull the box score data containing
stats like minutes played, points, rebounds and more for all playes who played from both teams that played.


## Installation and Running Program
1. Download repository
2. Install requirements using "pip install -r requirements.txt" 
3. Run data_collector.py
4. Enter in desired seasons
5. After running can do whatever you would like with data but if you want to test the models go to models.py and change
   years_to_examine to contain years you have downloaded data for
6. Run models.py
