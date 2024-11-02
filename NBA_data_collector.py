from API import league_data as l_data, box_score_data as b_data
from concurrent.futures.thread import ThreadPoolExecutor
from SQL import db_manager as db, SQL_data_collector as dc
from threading import Lock
from retrying import retry
import pandas as pd
import numpy as np
import time


# How many threads to use when downloading individual game data from NBA API
NUM_THREADS = 4
# Max amount of times to retry download as sometimes they can timeout
MAX_DOWNLOAD_ATTEMPTS = 10
# Used to prevent duplicate request of data from NBA API
GAME_LOCK = Lock()
GAME_PROCESSED = set()


def dash_to_individual(years):
    """
    Function that when called will return a list of strings with years between two years defined by a dash.
    Input = "2013-2016", Output = ["2013", "2014", "2015", "2016']
    Input = "2016-2013", Output = ["2016", "2015", "2014", "2013']
    Input = "2016-2016", Output = ["2016"]

    :param years: String formatted as start_date-end_date i.e., 2020-2023, 2023-2020
    :return: Returns a list of strings with years between and including
    """
    # Split up string by dash
    years = years.split("-")
    # Set start to integer value, so we can easily get the next value
    start = int(years[0])
    # Set end equal to integer value, so we can easily compare
    end = int(years[1])
    # Set years equal to just beginning of an array as we will append stats eventually adding the end date back
    years = years[:1]

    # Check to see if we need to go up or down to get to the end. Also make sure that end and start are different
    if end > start:
        diff = 1
    elif start > end:
        diff = -1
    else:
        return [start]

    # Loop until we have added all years between start and end
    while True:
        start += diff
        years.append(str(start))
        if start == end:
            break

    return years

def handle_year_input(inputs):
    """
    Given input from years will break down into array including dashed entries i.e.,
    inputs = "2018, 2020-2023"
    return = ["2018", "2019", "2020", "2021", "2022", "2023"]

    :param inputs: String with input from user with spaces indicating separate years they want to look at
    :return: Array of strings breaking down the input into individual indexes
    """
    inputs = inputs.split(" ")
    years = []
    for inpt in inputs:
        if "-" in inpt:
            years += dash_to_individual(inpt)
        else:
            years.append(inpt)

    return years


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS)
def threaded_get_save_game_data(game_id, year):
    """
    Given a game id will return a dataframe containing data on what teams played, what players played and how many
    minutes. Specifically will save the team abbreviation, player id, player name and minutes. Minutes do
    require some cleaning as when given from the NBA api 0 minutes shows up as None and minutes are formatted oddly
    with seconds. What I assume is 31 minutes and 35 seconds shows up as 31.0000:35. Cleaning fills all nones with 0
    and splits minutes by periods keeping only the first part. Turning 31.0000:35 into 31 meaning we DO NOT round

    :param game_id: String with id of game we want data for. Ids are from the NBA api
    :param year:
    :return: Data frame containing the team id, team abbreviation, player id, player name and minutes for
             given game
    """
    try:
        with GAME_LOCK:
            if game_id in GAME_PROCESSED:
                return
            GAME_PROCESSED.add(game_id)

        b_score_data = b_data.BoxScoreTraditionalV2(game_id=game_id)
        game_data = b_score_data.get_data_frames()[0]

        # Minutes are stored with seconds. 35 minutes 30 seconds is 35.0000:30
        if int(year) > 1995:
            game_data["MIN"] = game_data["MIN"].str.split(".").str[0]
            game_data = game_data.fillna(0)  # Data uses none instead of 0

        game_data = game_data.rename(columns={"TO": "TOV"})
        # Upload game stat to database
        db.upload_df_to_postgres(game_data, "game_stats")
        return game_data
    except Exception as e:
        print(str(e) + " for " + str(game_id))


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS)
def get_league_schedule_team_stats(year):
    """

    """
    stats = ["GAME_ID", "MATCHUP", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
             "OREB", "DREB","REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]
    league_data = l_data.LeagueGameLog(season=year)
    league_data = league_data.get_data_frames()[0]
    # Add OPP_TEAM_ID
    df_at = league_data[league_data['MATCHUP'].str.contains('@')][['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME']].rename(
        columns={'TEAM_ID': 'OPP_TEAM_ID', 'TEAM_ABBREVIATION': 'OPP_TEAM_ABBREVIATION', 'TEAM_NAME': 'OPP_TEAM_NAME' })
    # Merge DataFrames on 'GAME_ID'
    league_data = pd.merge(league_data, df_at, on='GAME_ID', how='left')

    # Figure out who won
    league_data["WL"] = np.where(league_data["WL"] == "W", league_data["MATCHUP"].str.slice(start=0, stop=3),
                                 league_data["MATCHUP"].str.slice(start=-3))
    # Rename column to make it easier to understand
    league_data = league_data.rename(columns={"WL": "WINNER", "TEAM_ID": "HOME_TEAM_ID", "TEAM_NAME": "HOME_TEAM_NAME",
                                              "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION"})

    team_data = league_data[stats]
    team_data.loc[:, "MATCHUP"] = team_data["MATCHUP"].str.slice(start=0, stop=3)
    team_data = team_data.rename(columns={"MATCHUP": "TEAM"})

    # Data set contains two instances for a single game, one for the home team and one for the away team
    # here we only take matchups with vs. instead of @ meaning we take all home team copies game
    league_data = league_data[league_data["MATCHUP"].str.contains("vs.", na=False)]
    league_data["GAME_DATE"] = pd.to_datetime(league_data["GAME_DATE"])
    league_data.drop(stats[2:], axis=1, inplace=True)

    # Change order so it's more readable for humans
    desired_order = ["SEASON_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "HOME_TEAM_NAME", "HOME_TEAM_ABBREVIATION",
                     "HOME_TEAM_ID", "OPP_TEAM_NAME", "OPP_TEAM_ABBREVIATION", "OPP_TEAM_ID", "WINNER","VIDEO_AVAILABLE"]
    league_data = league_data.reindex(columns=desired_order)

    # Save Data
    return league_data, team_data


def check_save_missing_game_stats():
    """
    Will check schedule and game_stats tables for GAME_IDs. Any missing GAME_IDs in game_stats will be downloaded then
    saved to server.

    Since schedule is once API request it either is uploaded to server or not. Each games stat is a single API request.
    Sometimes server times out and games are not downloaded. Until we have a way to 100% guarantee that games get
    downloaded we can use this to download missing game stats

    :return: Nothing
    """
    games_no_game_stats = dc.get_missing_game_data()
    print("\nMissing game stats for " + str(len(games_no_game_stats)) + " games\n")
    # Reset games processed for multiple runs without reset
    # Hopefully doesn't happen but possible that it timeouts during redownload and user needs to run command again
    # They may not reset program so we can just wipe the variable
    global GAME_PROCESSED
    GAME_PROCESSED = set()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(lambda game:threaded_get_save_game_data(game[0], game[1]), games_no_game_stats)


def get_save_data_for_year(year):
    # Download schedule from NBA API
    schedule, team_data = get_league_schedule_team_stats(year)

    # Upload schedule and team data for season to database
    db.upload_df_to_postgres(schedule, "schedule")
    db.upload_df_to_postgres(team_data, "team_stats")

    # Use schedule to get games
    game_ids = list(schedule["GAME_ID"])

    # Download data for each game and save it to database
    print("Starting download of games for " + year)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(lambda game_id:threaded_get_save_game_data(game_id, year), game_ids)
    print("Finished took " + str((time.time() - start_time) / 60) + " minutes\n ")


def set_up_year_function():
    year_input = input("\nWhat year(s) would you like to download data for? If multiple can use - or , i.e 2022, 2023 or "
                       "2022-2023: ")
    years = handle_year_input(year_input)
    for year in years:
        get_save_data_for_year(year)


def invalid_option(options_length):
    print("\nInvalid option must be a number from 1 - " + str(options_length - 1) + " or q/Q to exit\n")


def menu_options():
    options = {
        '1': set_up_year_function,
        '2': check_save_missing_game_stats,
        '3': db.init_database,
        'q': exit,
    }

    while True:
        # Print out options
        print("1. Download data for an entire year")
        print("2. Check all downloaded schedules to make sure all game stats are downloaded")
        print("3. Setup database")
        print("")

        # Get users choice and lowercase it to make q/Q the same
        user_selection = input("Enter number associated with choice (Enter q to exit): ")
        user_selection = user_selection.lower()

        # Call menu option if valid if not let user know how to properly use menu
        if user_selection in options:
            options[user_selection]()
        else:
            invalid_option(len(options))




if __name__ == "__main__":
    menu_options()
