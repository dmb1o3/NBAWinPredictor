# This is a file to collect data using an NBA API for a given season
# Uses tools from this repo https://github.com/swar/nba_api

import os
import shutil
import time
import numpy as np
import pandas as pd
import leauge_data as l_data
import boxscore_data as b_data

from threading import Lock
from concurrent.futures import ThreadPoolExecutor

# Global Variables
NUM_CORES = os.cpu_count()  # Used to know how many threads to use
GAME_LOCK = Lock()  # Used to sync threads for saving data to gameProcessed
GAME_PROCESSED = set()  # Saves data about game_ids we processed so threads don't do redundant work
PLAYER_LOCK = Lock()  # Used to sync threads for saving data on playerProcessed
PLAYER_PROCESSED = set()  # Saves data about player_ids we processed so threads don't do redundant work
NUM_PLAYER_PER_TEAM = 5  # Number of players per team that we should save stats for
# TODO Mess around with number. Sometimes if to high will cause program to error
# Ex: 2022 SAS at GAMES_BACK = 10 and NUM_PLAYER_PER_TEAM = 6 they can only come up with 5 so index out of bounds
# Might be able to add a fake player but probably cause issues with comparisons as they will have to have 0 for stats
# Could also drop these games when found either by catching these errors and cleaning data after or by checking when
# we build player averages
GAMES_BACK = 5  # Number of games to go back. Must be greater than or equal to 1
# Some teams have rough start to season or trades causing it not to have enough players
# this adds a buffer to the games we look at so players can build up enough games
# to be usable
GAMES_BACK_BUFFER = 2


def folder_setup():
    """
    Makes sure that folders used in program are set up

    :return: Does not return anything
    """
    folder_check(os.getcwd() + "/data")
    folder_check(os.getcwd() + "/data/games/")


def folder_check(directory):
    """
    Given a directory will check to see if it exists. If it does not exist make the directory

    :param directory: String with path of directory we want to check and possibly make
    :return: Does not return anything
    """
    if not os.path.isdir(directory):
        # Make directory if needed
        os.mkdir(directory)


def clean_player_stats(player_stats):
    """
    Given player stats for a game will drop stats that we do not need.

    :param player_stats: Dataframe containing player stats for a game
    :return: Returns the dataframe without unneeded columns
    """
    # Drop player stats that we do not need
    # Can drop amount of field goals and free throws as we keep percentage made and amount attempted
    # Can drop rebounds as we keep offensive rebounds and defensive rebounds
    # TODO Figure out how we want to handle injuries. Read obsidian file Accounting for injury for more details
    player_stats.drop(columns=['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY',
                               'NICKNAME', 'START_POSITION', "COMMENT", 'FGM', 'FG3M', 'FTM', 'REB'], inplace=True)

    return player_stats


def make_column_names(prefix, stats):
    """
    Given a list of stats will make a dictionary of column names from those stats.
    prefix = "Player_", Stats = ["", "MIN", "PTS"], NUM_PLAYERS_PER_TEAM = 2
    output = {0:{[PLAYER_1, PLAYER_1_MIN, PLAYER_1_PTS]} 1:{[PLAYER_2, PLAYER_2_MIN, PLAYER_2_PTS]}}

    :param prefix: String with prefix for all column names
    :param stats: List of strings with stats to append to column names
    :return: Dictionary with keys being numbers from 0 to NUM_PLAYERS_PER_TEAM and values being list of
             column names made up by adding prefix to front of all strings in stats
    """
    d = {}
    for i in range(0, NUM_PLAYER_PER_TEAM):
        columns = []
        for stat in stats:
            columns.append(prefix + str(i + 1) + "_" + stat)
        d[i] = columns

    return d


def save_player_stats(schedule, year):
    # Get each team of that season
    teams = schedule.HOME_TEAM.unique()
    team_dataframes = {}
    team_schedule = {}
    # Set up column names so that we can more easily add data averaged data later on
    stats = ["PLAYER_ID", "MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT", "OREB", "DREB", "AST", "STL",
             "BLK", "TO", "PF", "PTS", "PLUS_MINUS"]
    column_prefix = "Player_"
    column_names_dict = make_column_names(column_prefix, stats)
    opp_column_names_dict = make_column_names("OPP_" + column_prefix, stats)
    # To make sure we get stats for a players whole season, we need to append to PLayers file in folder of player's id.
    # Because of this, we need to make sure we delete any old data that might be in this years /data/players/year folder
    # If we don't then we will have duplicate data
    try:
        shutil.rmtree(os.getcwd() + "/data/players/" + year)
    except:
        pass

    for team in teams:
        # For each team get their schedule
        team_schedule[team] = schedule[schedule['MATCHUP'].str.contains(team)][["GAME_ID", "GAME_DATE", "HOME_TEAM"]]
        # Average player_stats and save all player ids for players who played on team
        team_dataframes[team] = average_and_save_player_stats(team_schedule[team][["GAME_ID", "GAME_DATE"]], team, year)

    # Now that we have player stats, we can go through the season team by team again.
    # The reason this cannot be done in the same loop above is that we need to wait until all teams have run so a
    for team in teams:
        # Get the dataframes for all players who played on a given team in a season
        team_dataframes[team] = get_averaged_player_stats(team_dataframes[team], year, team)

        # Change team schedule to only have home games. Because model cannot predict future and averaged stats for a
        # game include that game what we will do later is move results back one. This means opponent data will need to
        # be from the next game. Right now we only want to add home data. Later we deal with opponet data
        team_sch = team_schedule[team][GAMES_BACK + GAMES_BACK_BUFFER:]
        # Split into home and away
        home_data = team_sch[team_sch["HOME_TEAM"] == team]
        away_data = team_sch[team_sch["HOME_TEAM"] != team]
        # Loop through home games
        for game in home_data["GAME_ID"]:
            # Get players who played in game
            players_in_game = team_dataframes[team][team_dataframes[team]["GAME_ID"] == game]
            # Get players who played the most minutes
            players_in_game = players_in_game.head(NUM_PLAYER_PER_TEAM)
            # Add players stats to schedule
            for i in range(0, NUM_PLAYER_PER_TEAM):
                column_names = column_names_dict[i]
                schedule.loc[schedule["GAME_ID"] == game, column_names] = players_in_game.iloc[i][stats].to_numpy()
        # Loop through away games
        for game in away_data["GAME_ID"]:
            # Get players who played in game
            players_in_game = team_dataframes[team][team_dataframes[team]["GAME_ID"] == game]
            # Get players who played the most minutes
            players_in_game = players_in_game.head(NUM_PLAYER_PER_TEAM)
            # Add players stats to schedule
            for i in range(0, NUM_PLAYER_PER_TEAM):
                column_names = opp_column_names_dict[i]
                schedule.loc[schedule["GAME_ID"] == game, column_names] = players_in_game.iloc[i][stats].to_numpy()

    # Get rid of any rows with no data
    schedule = schedule.dropna()
    # Convert dates to something that the machine can understand

    schedule.to_csv("data/games/" + year + "/Final Dataset " +
                    "(PLAYERS_PER_TEAM = " + str(NUM_PLAYER_PER_TEAM) + " GAMES_BACK = " + str(GAMES_BACK) +
                    " GAMES_BUFFER = " + str(GAMES_BACK_BUFFER) + ").csv", index=False)


# @TODO Issue where if already ran will combine stats with last run. Solution may be to delete /data/players/year folder
def average_and_save_player_stats(team_schedule, team_abbrev, year):
    """
    When called will look at all games a team played from the given schedule. It will then make dataframes for each
    player combining all of their stats for that schedule. It will average their stats over a number of games set by
    global GAMES_BACK. It will then combine all player data frames into one for the team which it will then sort
    primarily by GAME_ID and then by MIN so when we finish stats are clumped into each game instead of each player. Also
    means for each game player stats are sorted in descending order of minutes played. We also save the teams averages
    to data/year/team_abbrev/team_abbrev_player_averages.csv

    :param team_schedule: Pandas series containing GAME_IDs we want to look at
    :param team_abbrev: String with abbreviation of team
    :param year: String of year schedule is for. Used to know where to save data
    :return: Player ids of players who played on team at some point
    """
    directory = "data/games/" + year + "/" + team_abbrev + "/"
    player_dataframes = {}
    player_ids = []  # Gets rid of warning later on when we reference player_ids. Not necessary
    # For each game get stats
    for game in team_schedule["GAME_ID"]:
        # Collect player averages for game. Need to read GAME_ID as string as it has leading zeros
        game_stats = pd.read_csv(directory + str(game) + "_stats.csv", dtype={'GAME_ID': str})
        # Get players in game
        player_ids = game_stats.PLAYER_ID.unique()
        for player_id in player_ids:
            # Get stats of player we are looking at
            player_stats = game_stats.loc[game_stats["PLAYER_ID"] == player_id].copy()
            player_stats = clean_player_stats(player_stats)
            player_stats = player_stats.merge(team_schedule[['GAME_ID', 'GAME_DATE']], on='GAME_ID',
                                              how='left')  # print(player_stats)
            # Check if we have dataframe already if so add to their original data frame
            if player_id in player_dataframes:
                player_dataframes[player_id] = pd.concat([player_dataframes[player_id], player_stats])
            # Else create a dataframe for them
            else:
                player_dataframes[player_id] = player_stats

    folder_check(os.getcwd() + "/data/players")
    folder_check(os.getcwd() + "/data/players/" + year)
    # For all player dataframes averages players games
    for player_id in player_dataframes:
        directory = os.getcwd() + "/data/players/" + year + "/" + str(player_id)
        folder_check(directory)
        try:
            player_data = pd.read_csv(directory + "/Player_Stats.csv", dtype={'GAME_ID': str})
            # Combine players games on current team with previous team
            combo_player_data = pd.concat([player_data, player_dataframes[player_id]])
            # Sort by game date to make sure things are in order
            combo_player_data = combo_player_data.sort_values(by=['GAME_DATE'], ascending=[True])
            # Save data
            combo_player_data.to_csv(directory + "/Player_Stats.csv", index=False)
        except Exception as e:
            player_dataframes[player_id].to_csv(directory + "/Player_Stats.csv", index=False)

    return list(player_dataframes.keys())


def get_averaged_player_stats(player_ids, year, team):
    directory = os.getcwd() + "/data/players/" + year + "/"
    invalid_cols = ["GAME_ID", "GAME_DATE", "PLAYER_ID"]
    player_dataframes = {}
    for player_id in player_ids:
        try:
            # Try to open averaged stats as if a previous team already did work we can just open it
            player_dataframes[player_id] = pd.read_csv(directory + player_id + "/Player_Averages.csv",
                                                       dtype={'GAME_ID': str})
        except Exception as e:
            # If an exception is thrown then we know file does not exist
            # Read the player stats
            player_dataframes[player_id] = pd.read_csv(directory + str(player_id) + "/Player_Stats.csv",
                                                       dtype={'GAME_ID': str})
            # Average them
            # Set up dataframe for rolling averages
            valid_cols = player_dataframes[player_id].select_dtypes(include=[float, int])
            valid_cols.drop(columns=["PLAYER_ID"], inplace=True)
            valid_cols = valid_cols.columns
            # Shift player stats back so that we can
            player_dataframes[player_id] = pd.concat([player_dataframes[player_id][invalid_cols],
                                                      player_dataframes[player_id][valid_cols].shift()], axis=1)
            # Average Player stats
            player_dataframes[player_id] = pd.concat([player_dataframes[player_id][invalid_cols],
                                                      player_dataframes[player_id][valid_cols].rolling(
                                                          GAMES_BACK).mean()],
                                                     axis=1)
            # Drop rows with any cols with None
            player_dataframes[player_id] = player_dataframes[player_id].dropna()

            # Save to dataframe and also to csv incase future team wants to use
            player_dataframes[player_id].to_csv(directory + str(player_id) + "/Player_Averages.csv", index=False)

    team_df = pd.concat(player_dataframes.values(), ignore_index=True)
    team_df = team_df.sort_values(by=['GAME_DATE', 'GAME_ID', 'MIN'], ascending=[True, True, False])
    team_df.to_csv(os.getcwd() + "/data/games/" + year + "/" + team + "/Team_Player_Averages.csv", index=False)
    return team_df


def get_game_data(game_id):
    """
    Given a game id will return a dataframe containing data on what teams played, what players played and how many
    minutes. Specifically will save the team abbreviation, player id, player name and minutes. Minutes do
    require some cleaning as when given from the NBA api 0 minutes shows up as None and minutes are formatted oddly
    with seconds. What I assume is 31 minutes and 35 seconds shows up as 31.0000:35. Cleaning fills all nones with 0
    and splits minutes by periods keeping only the first part. Turning 31.0000:35 into 31 meaning we DO NOT round

    :param game_id: String with id of game we want data for. ids are from the NBA api
    :return: Data frame containing the team id, team abbreviation, player id, player name and minutes for
             given game
    """
    box_score_data = b_data.BoxScoreTraditionalV2(game_id=game_id)
    box_score_data = box_score_data.get_data_frames()[0]  # [["TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN"]]
    # Minutes are stored with seconds. 35 minutes 30 seconds is 35.0000:30
    # The line below removes everything after the period
    box_score_data["MIN"] = box_score_data["MIN"].str.split(".").str[0]
    box_score_data = box_score_data.fillna(0)  # Data uses none instead of 0
    return box_score_data


def thread_save_game_data(year, row):
    """
    Expected to be called by multiple threads but not necessary. Will save data for a given GAME_ID (from row) and also
    save career stats of all players who played in game. This functions uses get_game_data() to get data on players in
    the game and save_player_data() to get and save career stats of player
    get data. It saves game data it gets to
    .../data/games/YEAR/GAME_ID/TEAM_ABBREVIATION/minutes.csv
    and saves player data to
    .../data/careerStats/PLAYER_ID/careerRegularSeasonStats.csv

    :param year: String with year game was played. Used to know what folder to save data to
    :param row: Tuple expected to contain a GAME_ID value and MATCHUP value.
                    GAME_ID is from game played and used by NBA API to get data
                    MATCHUP is string with team abbreviations playing. Ex: "LAC vs. LAL"
    :return: Does not return anything
    """
    # Get data from row tuple
    game_id = row.GAME_ID
    matchup = row.MATCHUP
    # Check to make sure thread is not saving game we already saved
    with GAME_LOCK:
        if game_id in GAME_PROCESSED:
            return
        game_data = get_game_data(game_id)
        GAME_PROCESSED.add(game_id)
    # Get game data
    home_team = matchup[0:3]
    away_team = matchup[-3:]
    home_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(home_team)]
    away_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(away_team)]
    # Make sure we have folder to save to
    directory = os.getcwd() + "/data/games/" + year
    home_directory = directory + "/" + home_team + "/"
    away_directory = directory + "/" + away_team + "/"
    folder_check(directory)
    folder_check(home_directory)
    folder_check(away_directory)
    # Save game data
    home_data.to_csv(home_directory + game_id + "_stats.csv", index=False)
    away_data.to_csv(away_directory + game_id + "_stats.csv", index=False)


def save_league_schedule(year):
    """
    Will save the league schedule for the given year. Will save the csv to
    .../data/games/year/games.csv.
    Saves the GAME_ID and MATCHUP for each game in schedule. We need GAME_ID but could get rid of MATCHUP
    keeping it for now for readability

    :param year: String containing the year we want schedule of
    :return: Dataframe with gameIDs and matchups
    """
    folder_check(os.getcwd() + "/data/games/" + year)  # Check we have a /games/year folder
    league_data = l_data.LeagueGameLog(season=year)
    league_data = league_data.get_data_frames()[0][["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "WL"]]
    # Add column for home team
    league_data["HOME_TEAM"] = league_data["MATCHUP"].str[:3]
    # Rename TEAM_ID to HOME_TEAM_ID for more accurate name
    league_data = league_data.rename(columns={})
    # Add OPP_TEAM_ID
    df_at = league_data[league_data['MATCHUP'].str.contains('@')][['GAME_ID', 'TEAM_ID']].rename(
        columns={'TEAM_ID': 'OPP_TEAM_ID'})
    # Merge DataFrames on 'GAME_ID'
    league_data = pd.merge(league_data, df_at, on='GAME_ID', how='left')

    # Figure out who won
    league_data["WL"] = np.where(league_data["WL"] == "W", league_data["MATCHUP"].str.slice(start=0, stop=3),
                                 league_data["MATCHUP"].str.slice(start=-3))
    # Rename column to make it easier to understand
    league_data = league_data.rename(columns={"WL": "WINNER", "TEAM_ID": "HOME_TEAM_ID"})

    league_data["HOME_TEAM_WON"] = (league_data['HOME_TEAM'] == league_data['WINNER']).astype(int)
    # Data set contains two instances for a single game one for the home team and one for away team
    # here we only take matchups with vs. instead of @ meaning we take all home team copies game
    league_data = league_data[league_data["MATCHUP"].str.contains("vs.", na=False)]
    # Change order so it's more readable for humans
    desired_order = ['GAME_ID', 'GAME_DATE', 'MATCHUP', 'HOME_TEAM_ID', 'OPP_TEAM_ID', 'HOME_TEAM', 'WINNER',
                     'HOME_TEAM_WON']
    league_data = league_data.reindex(columns=desired_order)
    league_data.to_csv(os.getcwd() + "/data/games/" + year + "/schedule.csv", index=False)

    return league_data


def save_league_data(year):
    """
    Saves all data needed for model for given year. Uses save_league_schedule() to get league schedule for given year
    and then feeds the given dataframe to thread_save_game_data() to process and save. After that runs
    save_player_data() which will go through the game data and get averages going b

    :param year: String with year we want data for
    :return: Does not return anything
    """
    # Save and get game id for all games played for given year
    schedule = save_league_schedule(year)
    # Reset global variable used for gamesProcessed to save space in case of multiple years saved per run.
    global GAME_PROCESSED
    GAME_PROCESSED = set()
    # Might want to consider another threading option
    # For debugging does not seem to raise any errors even when there are some that stop function from working
    # Using schedule save data about players that played and their minutes
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        executor.map(lambda row: thread_save_game_data(year, row), schedule.itertuples(index=False))

    # Save players stats
    save_player_stats(schedule, year)

    # Using data from the stats we need to append to our schedule data frame
    game_ids = schedule["GAME_ID"].tolist()


def get_all_data(years, data_is_downloaded):
    """
    Will get all data needed for a model for years given. The function also outputs the time it
    takes to save data

    Functions Called
    Uses save_league_data()

    :param data_is_downloaded: Boolean that is true if data is downloaded for given years and false if not
    :param years: List of strings. Each index being a year we want NBA data for
    :return: Does not return anything
    """
    # Save league schedule
    for year in years:
        start_time = time.time()
        print("Starting for NBA season " + year)
        if data_is_downloaded:
            save_player_stats(pd.read_csv("data/games/" + year + "/schedule.csv", dtype={'GAME_ID': str}), year)
        else:
            save_league_data(year)
        print("Finished took " + str((time.time() - start_time) / 60) + " minutes")

    # Combine all years into one dataframe


def gather_data_for_model(years):
    """
    The function assumes that the data has already been downloaded and prepared for the years provided.
    It also assumes that settings (GAMES_BACK, GAMES_BACK_BUFFER, NUM_PLAYER_PER_TEAM) used for all years is not only
    the same but also the current settings of the progam.

    :param year: Years the model would like data for
    :return: Returns two things
             1. Numpy array of all the parameters
             2. Numpy array with results
    """
    data = []
    for year in years:
        # Get data frame for given year
        df = pd.read_csv("data/games/" + year + "/Final Dataset " +
                           "(PLAYERS_PER_TEAM = " + str(NUM_PLAYER_PER_TEAM) + " GAMES_BACK = " + str(GAMES_BACK) +
                           " GAMES_BUFFER = " + str(GAMES_BACK_BUFFER) + ").csv", dtype={'GAME_ID': str})
        data.append(df)

    data = pd.concat(data, ignore_index=True)

    # Convert parameters to numpy array
    invalid_cols = ["GAME_DATE", "MATCHUP", "WINNER", "HOME_TEAM", "HOME_TEAM_WON"]
    x = data.drop(columns=invalid_cols).to_numpy()

    # Convert results to numpy array
    y = data["HOME_TEAM_WON"].to_numpy()
    data.to_csv("data/data.csv", index=False)

    return x, y


def main():
    # Get information from user, so we know what seasons to download and/or prepare data for
    # Also asks user if they already have data downloaded, so we can skip download and skip to preparing that data
    years = input("What years would you like to download/prepare? If multiple just type them with a space like \"2020 "
                  "2021 2022\" ")
    years = years.split()
    print("Do you have the data downloaded already? (Type number associated with choice)")
    print("1. Yes")
    print("2. No")
    downloaded_data = input("")
    if downloaded_data == "1":
        downloaded_data = True
    else:
        downloaded_data = False

    # Make sure we have basic folders needed for program setup
    folder_setup()
    get_all_data(years, downloaded_data)


if __name__ == "__main__":
    main()
