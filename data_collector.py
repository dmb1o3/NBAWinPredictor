# This is a file to collect data using an NBA API for a given season

import os
import shutil
import time
import numpy as np
import pandas as pd
import league_data as l_data
import box_score_data as b_data
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from retrying import retry

# Global Variables
NUM_THREADS = 4  # Used to know how many threads to use when downloading individual data for games
MAX_DOWNLOAD_ATTEMPTS = 4  # Set to -1 for infinite. Controls number of times to try to download from NBA api
GAME_LOCK = Lock()  # Used to sync threads for saving data to gameProcessed
GAME_PROCESSED = set()  # Saves data about game_ids we processed so threads don't do redundant work
PLAYER_LOCK = Lock()  # Used to sync threads for saving data on playerProcessed
PLAYER_PROCESSED = set()  # Saves data about player_ids we processed so threads don't do redundant work

# Global Settings for data collection
NUM_PLAYER_PER_TEAM = 5  # Number of players per team that we should save stats for
GAMES_BACK = 5  # Number of games to go back for rolling average. Must be greater than or equal to 1
GAMES_BACK_TEAM_V_TEAM = 4  # Number of games to go back for rolling average. Must be greater than or equal to 1
GAMES_BACK_BUFFER = 2  # Buffer to help some teams as in some season they struggle in beginning


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





def folder_setup():
    """
    Makes sure that folders used in program are set up

    :return: Does not return anything
    """
    folder_check(os.getcwd() + "/data")
    folder_check(os.getcwd() + "/data/games/")
    folder_check(os.getcwd() + "/data/players/")
    folder_check(os.getcwd() + "/data/models/")


def folder_check(directory):
    """
    Given a directory will check to see if it exists. If it does not exist, make the directory

    :param directory: String with a path for directory we want to check and possibly make
    :return: Does not return anything
    """
    if not os.path.isdir(directory):
        # Make directory if needed
        os.mkdir(directory)


def clean_player_stats(player_stats):
    """
    Given player stats for a game will drop stats that we do not need.
    Drops columns that have unuseful information, redundant data or data we can calculate with other data in
    player box score.

    :param player_stats: Dataframe containing player stats for a single game
    :return: Returns the dataframe without unnecessary columns
    """
    # Can drop number of field goals and free throws made as we keep percentage made and amount attempted
    # Can drop total rebounds as we keep offensive rebounds and defensive rebounds
    player_stats.drop(columns=['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY',
                               'NICKNAME', 'START_POSITION', "COMMENT", 'FGM', 'FG3M', 'FTM', 'REB'], inplace=True)

    return player_stats


def make_column_names(prefix, stats):
    """
    Given a list of stats will return a list of column names for those stats with the prefixed attached.
    Prefix = "Player_", Stats = ["", "MIN", "PTS"], NUM_PLAYERS_PER_TEAM = 2
    output = [[PLAYER_1, PLAYER_1_MIN, PLAYER_1_PTS], [PLAYER_2, PLAYER_2_MIN, PLAYER_2_PTS]]

    :param prefix: String with prefix for all column names
    :param stats: List of strings with stats to use as column names after attaching a prefix
    :return: 2d array with each index being a list of strings containing column names created for one player
    """
    d = []
    for i in range(0, NUM_PLAYER_PER_TEAM):
        column = []
        for stat in stats:
            column.append(prefix + str(i + 1) + "_" + stat)
        d.append(column)

    return d


def set_up_columns(schedule, home_columns):
    """
    Given a schedule and the columns of stats we want for, the home team will make columns for said stats.
    This is done so all dataframes for all years are made the same. Otherwise, if we process an away game first,
    some years would have game info, away stats, home stats instead of the intended game stats, home stats, away stats

    :param schedule: Dataframe with an NBA schedule
    :param home_columns: 2d array with each array containing column names we want added
    :return: Returns schedule with home_columns added
    """
    # Set up home columns first so they appear first
    for player_column in home_columns:
        for stat_colum in player_column:
            schedule[stat_colum] = None

    return schedule


def is_thirty_one_month(month):
    """
    Will return True if it is a month with 31 days in it and False if not

    :param month: Integer value representing a month i.e., Jan = 1, Feb = 2, Mar = 3 ...
    :return: Boolean that is True if it is a month with 31 days in it and False if not
    """
    thirty_one_months = [1, 3, 5, 7, 8, 10, 12]
    if month in thirty_one_months:
        return True

    return False


def is_back_to_back(previous_date, current_date):
    """
    Given two dates will return 1 if dates are back to back (sequential) and 0 if not

    :param previous_date: Series with date (yyyy-mm-dd) of previous game
    :param current_date: Series with date (yyyy-mm-dd) of current game
    :return: Return 1 if dates are back to back and 0 if not
    """

    # Get strings from series
    previous_date = previous_date.iloc[0]
    current_date = current_date.iloc[0]
    # Extract previous and current day
    p_day = int(previous_date[8:])
    c_day = int(current_date[8:])
    # Extract previous and current month
    p_month = int(previous_date[5:7])
    c_month = int(current_date[5:7])
    # Extract previous and current year
    p_year = int(previous_date[0:4])
    c_year = int(current_date[0:4])

    # Handle december year skip
    if p_month == 12 and p_day == 31 and p_year + 1 == c_year and c_month == 1 and c_day == 1:
        return 1

    # Should never happen since we go through data sequentially but just to be safe and in case we end up using
    # preparing multiple years at some point
    if p_year != c_year:
        return 0

    # All months have at least 28 days in them if the previous day + 1 is the current day then its b2b
    if p_month == c_month and p_day < 28 and p_day + 1 == c_day:
        return 1

    # Handle february
    if p_month == 2:
        # See if leap year and sequential
        if p_year % 4 == 0 and p_day + 1 == c_day:
            return 1
        # Catch rollover to next month
        elif p_day == 29 or p_day == 28:
            # For leap year
            if p_year % 4 == 0 and p_day == 29 and c_day == 1:
                return 1
            elif p_day == 28 and c_day == 1:
                return 1
            else:
                return 0
        else:
            return 0

    # Handle months with 31 days
    if is_thirty_one_month(p_month):
        if p_month == c_month and p_day < 31 and p_day + 1 == c_day:
            return 1
        # Handle roll over
        if p_day == 31 and c_day == 1 and p_month + 1 == c_month:
            return 1
    # Handle months with 30 days
    else:
        if p_month == c_month and p_day < 30 and p_day + 1 == c_day:
            return 1
        # Handle roll over
        if p_day == 30 and c_day == 1 and p_month + 1 == c_month:
            return 1

    return 0


def save_player_stats(team_schedule, team_abbrev, year):
    """
    When called will look at all games a team played from the given schedule. It will then make dataframes for each
    player combining all of their stats for that schedule. It will average their stats over a number of games set by
    global GAMES_BACK. It will then combine all player data frames into one for the team which it will then sort
    primarily by GAME_ID and then by MIN, so when we finish, stats are clumped into each game instead of each player.
    Also means for each game player stats are sorted in descending order of minutes played. We also save the team
    averages to data/year/team_abbrev/team_abbrev_player_averages.csv

    :param team_schedule: Pandas series containing GAME_IDs we want to look at
    :param team_abbrev: String with abbreviation of team
    :param year: String of year schedule is for. Used to know where to save data
    :return: Player ids of players who played on the team at some point
    """
    directory = "data/games/" + year + "/" + team_abbrev + "/"
    player_dataframes = {}
    player_ids = []  # Gets rid of warning later on when we reference player_ids. Not necessary
    # For each game get stats
    for game in team_schedule["GAME_ID"]:
        # Collect player averages for game. Need to read GAME_ID as a string as it has leading zeros
        game_stats = pd.read_csv(directory + str(game) + "_stats.csv", dtype={'GAME_ID': str})
        # Get players in the game
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
                player_dataframes[player_id] = player_stats

    folder_check(os.getcwd() + "/data/players/" + year)
    # For all player dataframes averages players games
    for player_id in player_dataframes:
        directory = os.getcwd() + "/data/players/" + year + "/" + str(player_id)
        folder_check(directory)
        try:
            player_data = pd.read_csv(directory + "/Player_Stats.csv", dtype={'GAME_ID': str})
            # Combine player's games on the current team with the previous team
            combo_player_data = pd.concat([player_data, player_dataframes[player_id]])
            # Sort by game date to make sure things are in order
            combo_player_data = combo_player_data.sort_values(by=['GAME_DATE'], ascending=[True])
            # Save data
            combo_player_data.to_csv(directory + "/Player_Stats.csv", index=False)
        except Exception as e:
            player_dataframes[player_id].to_csv(directory + "/Player_Stats.csv", index=False)

    return list(player_dataframes.keys())


def get_averaged_player_stats(player_ids, year, team):
    """
    Will get stats for given player_ids from given year that have played on a given team. These stats are from a players
    entire season not just with team. These stats will then have a rolling average applied to them decided by global
    variable GAMES_BACK. These stats will then be saved to /data/players/ + year + / + player_id + /Player_Averages.csv
    so that future teams can use them if needed. After doing this with all players stats will then combine them into one
    dataframe, sort them in order of GAME_DATE, GAME_ID and then MIN and then return and save the dataframe to
    "/data/games/" + year + "/" + team + "/Team_Player_Averages.csv"

    :param player_ids: List of player_ids that played on given team at some point in given year
    :param year: Year of nba season we are looking at
    :param team: String with abbreviation of an NBA team we are looking at
    :return: Returns data frame with entire teams averaged player stats. In order of the date the game was played
    """
    # Set up directory for finding/saving player stats
    directory = os.getcwd() + "/data/players/" + year + "/"
    # Columns we do not want averaged
    invalid_cols = ["GAME_ID", "GAME_DATE", "PLAYER_ID"]
    # Data frame where keys will be player_id and values will be dataframe of player stats
    player_dataframes = {}
    for player_id in player_ids:
        try:
            # Try to open averaged stats as if a previous team already did work, we can just open it
            player_dataframes[player_id] = pd.read_csv(directory + player_id + "/Player_Averages.csv",
                                                       dtype={'GAME_ID': str})
        except Exception as e:
            # If an exception is thrown then we know file does not exist

            # Read the player stats
            player_dataframes[player_id] = pd.read_csv(directory + str(player_id) + "/Player_Stats.csv",
                                                       dtype={'GAME_ID': str})

            # Set up dataframe for rolling averages
            valid_cols = player_dataframes[player_id].select_dtypes(include=[float, int])
            valid_cols.drop(columns=["PLAYER_ID"], inplace=True)
            valid_cols = valid_cols.columns

            # Shift player stats back so that when we look at a game, we only have stats from previous games
            # preventing us from "seeing the future"
            player_dataframes[player_id] = pd.concat([player_dataframes[player_id][invalid_cols],
                                                      player_dataframes[player_id][valid_cols].shift()], axis=1)
            # Average Player stats
            player_dataframes[player_id] = pd.concat([player_dataframes[player_id][invalid_cols],
                                                      player_dataframes[player_id][valid_cols].rolling(
                                                          GAMES_BACK).mean()],
                                                     axis=1)
            # Drop rows with any cols with None
            player_dataframes[player_id] = player_dataframes[player_id].dropna()

            # Save dataframe to csv incase future team wants to use
            player_dataframes[player_id].to_csv(directory + str(player_id) + "/Player_Averages.csv", index=False)

    # Create team dataframe by combining all player dataframes
    team_df = pd.concat(player_dataframes.values(), ignore_index=True)
    # Sort by GAME_DATE so that all games are in chronological order, then by GAME_ID in case some games have the same
    # GAME_DATE and then by MIN so that each game has the player with the highest minutes at the top of that game_id
    team_df = team_df.sort_values(by=['GAME_DATE', 'GAME_ID', 'MIN'], ascending=[True, True, False])
    # Save dataframe so we can look at if we want
    team_df.to_csv(os.getcwd() + "/data/games/" + year + "/" + team + "/Team_Player_Averages.csv", index=False)
    return team_df


def prepare_data(schedule, year):
    """
    Given a schedule for a given nba season and the year of that schedule will go through the season team by team,
    collect player stats, average them based on GAMES_BACK, and then add NUM_PLAYERS_PER_TEAM of stats for home and
    away teams. Players are added in order of minutes played, so if NUM_PLAYERS_PER_TEAM = 4 then 4 players from each
    team who played in game are added based on the highest minutes played in last GAMES_BACK number of games. We do
    not include the game played in averages as that would be future data when evaluating that particular game.

    :param schedule: A data frame with the following columns GAME_ID, GAME_DATE, MATCHUP, HOME_TEAM_ID, OPP_TEAM_ID,
                     HOME_TEAM, WINNER, HOME_TEAM_WON
    :param year: The year the nba schedule is from
    :return: Does not return anything but saves dataframe of data to "data/games/" + year + "/Final Dataset + MODIFIERS"
    """
    print("\nPreparing data from NBA season " + year)
    start_time = time.time()
    # Set up variables
    team_dataframes = {}  # Used to keep track of player_ids of players on team then used to store players dataframes
    team_schedule = {}  # Used to store schedule dataframe for individual teams

    # Set up column names so that we can more easily add averaged data later on
    stats = ["PLAYER_ID", "MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT", "OREB", "DREB", "AST", "STL",
             "BLK", "TO", "PF", "PTS", "PLUS_MINUS"]
    column_prefix = "PLAYER_"
    home_column_names = make_column_names("HOME_" + column_prefix, stats)
    away_column_names = make_column_names("AWAY_" + column_prefix, stats)
    schedule = set_up_columns(schedule, [["HOME_WIN_STREAK", "HOME_B2B"]] + home_column_names)

    # To make sure we get stats for a players whole season, we need to append to Players file in folder of player's
    # id. Because of this, we need to make sure we delete any old data that might be in this year's
    # /data/players/year folder If we don't, then we will have duplicate data
    try:
        shutil.rmtree(os.getcwd() + "/data/players/" + year)
    except:
        pass

    # Get each team of that season
    teams = schedule.HOME_TEAM.unique()

    for team in teams:
        # For each team get their schedule
        team_schedule[team] = schedule[schedule['MATCHUP'].str.contains(team)][
            ["GAME_ID", "GAME_DATE", "MATCHUP", "HOME_TEAM", "WINNER"]]
        # Average player_stats and save all player ids for players who played on the team
        team_dataframes[team] = save_player_stats(team_schedule[team][["GAME_ID", "GAME_DATE"]], team, year)

    # Now that we have player stats, we can go through the season team by team again.
    # The reason this cannot be done in the same loop above is that we need to wait until all teams have run so a
    # player's stats are for the whole season
    for team in teams:
        # Get the averaged dataframes for all players who played on the team in season
        team_dataframes[team] = get_averaged_player_stats(team_dataframes[team], year, team)
        # Right now we just set to 0 but for some teams since not start of season will be on win streak
        games_not_processed = team_schedule[team][:GAMES_BACK + GAMES_BACK_BUFFER]
        # Loop through games not processed for a model so that we have accurate win streak for start of games we process
        current_win_loss_streak = 0
        for winner in games_not_processed["WINNER"][::-1]:
            if winner != team:
                if current_win_loss_streak > 0:
                    break
                current_win_loss_streak -= 1
            else:
                if current_win_loss_streak < 0:
                    break
                current_win_loss_streak += 1

        # Get the previous game date so we know if back to back games
        previous_game_date = games_not_processed["GAME_DATE"][-1:]
        team_schedule[team] = team_schedule[team][GAMES_BACK + GAMES_BACK_BUFFER:]
        team_sch = team_schedule[team]

        # Loop through team schedule
        for game in team_sch["GAME_ID"]:
            # Get players who played in game
            players_in_game = team_dataframes[team][team_dataframes[team]["GAME_ID"] == game]
            # Get players who played the most minutes
            players_in_game = players_in_game.head(NUM_PLAYER_PER_TEAM)
            if team_sch.loc[team_sch["GAME_ID"] == game]["HOME_TEAM"].values[0] == team:
                columns = home_column_names
                prefix = "HOME_"
            else:
                columns = away_column_names
                prefix = "AWAY_"

            # Add win streak stat to schedule
            schedule.loc[schedule["GAME_ID"] == game, prefix + "WIN_STREAK"] = current_win_loss_streak

            # Add b2b
            game_date = schedule.loc[schedule["GAME_ID"] == game]["GAME_DATE"]
            schedule.loc[schedule["GAME_ID"] == game, prefix + "B2B"] = is_back_to_back(previous_game_date, game_date)
            previous_game_date = game_date

            # Update win/loss streak
            if schedule.loc[schedule["GAME_ID"] == game]["WINNER"].values[0] == team:
                if current_win_loss_streak > 0:
                    current_win_loss_streak += 1
                else:
                    current_win_loss_streak = 1
            else:
                if current_win_loss_streak < 0:
                    current_win_loss_streak -= 1
                else:
                    current_win_loss_streak = -1

            # Add player's stats to schedule
            for i in range(0, NUM_PLAYER_PER_TEAM):
                column_names = columns[i]
                schedule.loc[schedule["GAME_ID"] == game, column_names] = (players_in_game.iloc[i][stats].to_numpy())

    # Get rid of any rows with no data
    schedule = schedule.dropna()
    # Convert the GAME_DATE column to 3 columns of integers
    year_month_day = schedule["GAME_DATE"].str.split('-', expand=True)
    schedule.loc[:, "GAME_DATE"] = year_month_day[0]
    schedule = pd.concat([schedule.iloc[:, :2], year_month_day[1], schedule.iloc[:, 2:]], axis=1)
    schedule = pd.concat([schedule.iloc[:, :3], year_month_day[2], schedule.iloc[:, 3:]], axis=1)
    schedule = schedule.rename(columns={"GAME_DATE": "YEAR", 1: "MONTH", 2: "DAY"})

    print("Finished took " + str((time.time() - start_time) / 60) + " minutes to prepare data")

    # Save data frame so we can reuse
    schedule.to_csv("data/games/" + year + "/Final Dataset " +
                    "(PLAYERS_PER_TEAM = " + str(NUM_PLAYER_PER_TEAM) + " GAMES_BACK = " + str(GAMES_BACK) +
                    " GAMES_BUFFER = " + str(GAMES_BACK_BUFFER) + ").csv", index=False)

    return teams, schedule


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS)
def get_game_data(game_id, year):
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
    box_score_data = b_data.BoxScoreTraditionalV2(game_id=game_id)
    box_score_data = box_score_data.get_data_frames()[0]  # [["TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN"]]
    # Minutes are stored with seconds. 35 minutes 30 seconds is 35.0000:30

    if int(year) > 1995:
        box_score_data["MIN"] = box_score_data["MIN"].str.split(".").str[0]
        box_score_data = box_score_data.fillna(0)  # Data uses none instead of 0
    return box_score_data


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS)
def save_league_schedule(year):
    """
    Will save the league schedule for the given year. Will save the csv to
    .../data/games/year/games.csv.
    Save the GAME_ID and MATCHUP for each game in schedule. We need GAME_ID but could get rid of MATCHUP
    keeping it for now for readability

    :param year: String containing the year we want schedule of
    :return: Dataframe with gameIDs and matchups
    """
    folder_check(os.getcwd() + "/data/games/" + year)  # Check we have a /games/year folder
    league_data = l_data.LeagueGameLog(season=year)
    league_data = league_data.get_data_frames()[0][["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "WL"]]
    # Add column for the home team
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
    # Data set contains two instances for a single game, one for the home team and one for the away team
    # here we only take matchups with vs. instead of @ meaning we take all home team copies game
    league_data = league_data[league_data["MATCHUP"].str.contains("vs.", na=False)]
    # Change order so it's more readable for humans
    desired_order = ['GAME_ID', 'GAME_DATE', 'MATCHUP', 'HOME_TEAM_ID', 'OPP_TEAM_ID', 'HOME_TEAM', 'WINNER',
                     'HOME_TEAM_WON']
    league_data = league_data.reindex(columns=desired_order)
    league_data.to_csv(os.getcwd() + "/data/games/" + year + "/schedule.csv", index=False)

    print("League schedule for " + year + " has been saved")
    return league_data


def thread_save_game_data(year, row):
    """
    Expected to be called by multiple threads but not necessary.
    Will save data for a given GAME_ID (from row) and
    also save career stats of all players who played in the game.
    This functions uses get_game_data() to get data on
    players in the game It saves game data it gets to .../data/games/YEAR/GAME_ID/TEAM_ABBREVIATION/minutes.csv

    :param year: String containing year game was played.
    Used to know what folder to save data to
    :param row: Tuple expected to contain a GAME_ID value and MATCHUP value.
                    GAME_ID is from game played and used by NBA API to get data
                    MATCHUP is string with team abbreviations playing.
                    Ex: "LAC vs. LAL"
    :return: Does not return anything
    """
    try:
        # Get data from row tuple
        game_id = row.GAME_ID
        matchup = row.MATCHUP

        # Check to make sure thread is not saving game we already saved
        with GAME_LOCK:
            if game_id in GAME_PROCESSED:
                return
            GAME_PROCESSED.add(game_id)
        game_data = get_game_data(game_id, year)

        # Get game data
        home_team = matchup[0:3]
        away_team = matchup[-3:]
        home_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(home_team)]
        away_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(away_team)]

        # Make sure we have folders to save to
        directory = os.getcwd() + "/data/games/" + year
        home_directory = directory + "/" + home_team + "/"
        away_directory = directory + "/" + away_team + "/"
        folder_check(home_directory)
        folder_check(away_directory)
        # Save game data
        home_data.to_csv(home_directory + game_id + "_stats.csv", index=False)
        away_data.to_csv(away_directory + game_id + "_stats.csv", index=False)

    except Exception as e:
        print(str(e) + " for " + str(row))


def save_download_data(year):
    """
    Saves all data needed for a model for a given year.Uses save_league_schedule() to get league schedule for a given
    year and then feeds the given dataframe to thread_save_game_data() to process and save.

    :param year: String with year we want data for
    :return: Does not return anything
    """

    print("\nStarting download for NBA season " + year)
    start_time = time.time()
    # Save and get game id for all games played for a given year
    schedule = save_league_schedule(year)
    # Make sure we have a folder to save games to
    folder_check(os.getcwd() + "/data/games/" + year)
    # Reset global variable used for gamesProcessed to save space in case of multiple years saved per run.
    global GAME_PROCESSED
    GAME_PROCESSED = set()

    # Using schedule save data about players that played and their minutes
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(lambda row: thread_save_game_data(year, row), schedule.itertuples(index=False))

    print("Finished took " + str((time.time() - start_time) / 60) + " minutes to download")
    return schedule


def get_all_data(years, data_is_downloaded):
    """
    Will get all data needed for a model for years given. The function also outputs the time it
    takes to save and prepare data.

    :param data_is_downloaded: Boolean that is true if data is downloaded for all given years and false if not
    :param years: List of strings. Each index being a year we want NBA data for
    :return: Does not return anything
    """
    # Save league schedule
    for year in years:
        if data_is_downloaded:
            schedule = pd.read_csv("data/games/" + year + "/schedule.csv", dtype={'GAME_ID': str})
        else:
            schedule = save_download_data(year)

        # Prepare data for nba season
        teams = prepare_data(schedule, year)
        # Get team v team stats
        # get_team_stats(teams, schedule)
        #


def main():
    # TODO add ability top apply rolling average over multiple seasons. Won't help much for rolling average but will
    # help for team v team stats as some teams only play twice a year
    # TODO add column to track how many players from team are on team from game to game. So if 5 players and all 5 play
    # in next game its 5 if only 3 then 3. If in game after
    # get_team_stats(['GSW', 'DEN', 'LAC', 'ORL', 'SAS', 'MEM', 'CHI', 'TOR', 'BKN', 'MIA', 'NYK', 'UTA', 'CHA',
    # 'IND', 'MIL', 'LAL', 'ATL', 'SAC', 'POR', 'DAL', 'CLE', 'BOS', 'WAS', 'DET',
    # 'PHX', 'MIN', 'NOP', 'HOU', 'PHI', 'OKC'],
    # pd.read_csv("data/games/" + "2023" + "/schedule.csv", dtype={'GAME_ID': str}))
    # exit()
    # @TODO Maybe add in start_position along with stats so we know what position best players play
    # Get information from user, so we know what seasons to download and/or prepare data for
    # Also asks user if they already have data downloaded, so we can skip download and skip to preparing that data
    years = input("What years would you like to download/prepare? If multiple just type them with a space like \"2020 "
                  "2021 2022\" ")
    years = handle_year_input(years)
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

    # Get data for all years
    get_all_data(years, downloaded_data)


def get_team_stats(teams, schedule):
    """


    In the future may change to pull stats against team v team purely instead of relying on prepare_data()

    :param teams:
    :param schedule:
    :return:
    """
    # Loop through each team
    for i in range(0, len(teams)):
        current_team = teams[i]
        print("\n Start " + current_team)

        # For each team loop through season for all others team adding in team v team stats
        for versus_team in teams[i + 1:]:
            # Get games where teams played against each other
            team_v_team_schedule = schedule.loc[((schedule["MATCHUP"] == current_team + " vs. " + versus_team) |
                                                 (schedule["MATCHUP"] == versus_team + " vs. " + current_team))]
            print(team_v_team_schedule)
            #


def get_data_for_model(years):
    """
    The function assumes that the data has already been downloaded and prepared for the years provided.
    It also assumes that settings (GAMES_BACK, GAMES_BACK_BUFFER, NUM_PLAYER_PER_TEAM) used for all years are not only
    the same but also the current settings of the program.

    :param years: Array of strings with each index being a year we would like nba data for
    :return: Returns two things
             1. Numpy array of all parameters
             2. Numpy array with results
    """
    print("\nCollecting data for years: " + str(years) + "\nGAMES_BACK = " + str(GAMES_BACK) +
          "\nNUM_PLAYER_PER_TEAM = " + str(NUM_PLAYER_PER_TEAM) + "\nGAMES_BACK_BUFFER = " + str(GAMES_BACK_BUFFER))
    data = []
    for year in years:
        # Get data frame for given year
        df = pd.read_csv("data/games/" + year + "/Final Dataset " +
                         "(PLAYERS_PER_TEAM = " + str(NUM_PLAYER_PER_TEAM) + " GAMES_BACK = " + str(GAMES_BACK) +
                         " GAMES_BUFFER = " + str(GAMES_BACK_BUFFER) + ").csv", dtype={'GAME_ID': str})
        data.append(df)

    data = pd.concat(data, ignore_index=True)

    # Convert parameters to a numpy array
    extra_columns = ["GAME_ID"]
    invalid_cols = ["MATCHUP", "WINNER", "HOME_TEAM", "HOME_TEAM_WON"]
    x = data.drop(columns=invalid_cols + extra_columns)
    x.to_csv("data/models/Features.csv", index=False)
    column_names = x.columns.to_list()
    x = x.to_numpy()

    # Convert results to a numpy array
    y = data["HOME_TEAM_WON"].to_numpy()

    # Make sure a folder is set up then save data
    folder_check("data/models")

    return x, y, column_names


if __name__ == "__main__":
    main()
