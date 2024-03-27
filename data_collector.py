# This is a file to collect data using an NBA API for a given season
# Uses tools from this repo https://github.com/swar/nba_api

import os
import time
import numpy as np
import player_data as p_data
import leauge_data as l_data
import boxscore_data as b_data

from threading import Lock
from concurrent.futures import ThreadPoolExecutor

# Global Variables
num_cores = os.cpu_count()  # Used to know how many threads to use
gameLock = Lock()  # Used to sync threads for saving data to gameProcessed
gameProcessed = set()  # Saves data about game_ids we processed so threads don't do redundant work
playerLock = Lock()  # Used to sync threads for saving data on playerProcessed
playerProcessed = set()  # Saves data about player_ids we processed so threads don't do redundant work


def folder_setup():
    """
    Makes sure that folders used in program are set up

    :return: Does not return anything
    """
    folder_check(os.getcwd() + "/data")
    folder_check(os.getcwd() + "/data/games/")
    folder_check(os.getcwd() + "/data/careerStats/")


def folder_check(directory):
    """
    Given a directory will check to see if it exists. If it does not exist make the directory

    :param directory: Path of directory we want to check and possibly make
    :return: Does not return anything
    """
    if not os.path.isdir(directory):
        # Make directory if needed
        os.mkdir(directory)


def thread_save_player_stats(players):
    """
    Expected to be called with threads but not necessary. Will download the career stats of all players given to
    .../data/careerStats/PLAYER_ID/careerRegularSeasonStats.csv

    :param players: List of players we want career stats for
    :return: Does not return anything
    """
    directory = os.getcwd() + "/data/careerStats/"
    for player in players:
        p = str(player)
        with playerLock:
            # If we already have data for player
            # We can skip to the next loop
            if p in playerProcessed:
                continue
            playerProcessed.add(p)

        player_data = p_data.PlayerCareerStats(p)
        player_data = player_data.get_data_frames()[0]
        # Choose which stats to keep. Choose not to keep makes for FG, 3P or FT as we should not need and
        # can calculate using the attempts and pct if we do end up needing it
        player_data = player_data[["SEASON_ID", "PLAYER_AGE", "GP", "GS", "MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT",
                                  "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]]
        folder_check(directory + p)
        player_data.to_csv(directory + p + "/careerRegularSeasonStats.csv")


def get_game_data(game_id):
    """
    Given a game id will return a dataframe containing data on what teams played, what players played and how many
    minutes. Specifically will save the team abbreviation, player id, player name and minutes. Minutes do
    require some cleaning as when given from the NBA api 0 minutes shows up as None and minutes are formatted oddly
    with seconds. What I assume is 31 minutes and 35 seconds shows up as 31.0000:35. Cleaning fills all nones with 0
    and splits minutes by periods keeping only the first part. Turning 31.0000:35 into 31 meaning we DO NOT round

    :param game_id: id of game we want data for. ids are from the NBA api
    :return: returns a data frame containing the team id, team abbreviation, player id, player name and minutes for
             given game
    """
    box_score_data = b_data.BoxScoreTraditionalV2(game_id=game_id)
    box_score_data = box_score_data.get_data_frames()[0] #[["TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN"]]
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

    :param year: Year game was played. Used to know what folder to save data to
    :param row: Tuple expected to contain a GAME_ID value and MATCHUP value.
                    GAME_ID is from game played and used by NBA API to get data
                    MATCHUP is string with team abbreviations playing. Ex: "LAC vs. LAL"
    :return: Does not return anything
    """
    # Get data from row tuple
    game_id = row.GAME_ID
    matchup = row.MATCHUP
    # Check to make sure thread is not saving game we already saved
    with gameLock:
        if game_id in gameProcessed:
            return
        gameProcessed.add(game_id)
    # Get game data
    game_data = get_game_data(game_id)
    # Split data into home team and away team
    """
    home_team = matchup[0:3]
    away_team = matchup[-3:]
    home_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(home_team)]
    away_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(away_team)]
    # Check player data to see if we need to save any player stats
    both_team_players = game_data["PLAYER_ID"]
    thread_save_player_stats(both_team_players)
    """
    # Make sure we have folder to save to
    directory = os.getcwd() + "/data/games/" + year
    folder_check(directory)
    # Save game data
    game_data.to_csv(directory + "/" + game_id + "_stats.csv")
    """
    directory = os.getcwd() + "/data/games/" + year + "/" + game_id
    folder_check(directory)
    folder_check(directory + "/" + home_team)
    folder_check(directory + "/" + away_team)
    # Save game data
    home_data.to_csv(directory + "/" + home_team + "/minutes.csv")
    away_data.to_csv(directory + "/" + away_team + "/minutes.csv")
    """


def save_league_schedule(year):
    """
    Will save the league schedule for the given year. Will save the csv to
    .../data/games/year/games.csv.
    Saves the GAME_ID and MATCHUP for each game in schedule. We need GAME_ID but could get rid of MATCHUP
    keeping it for now for readability

    :param year: String containing the year we want schedule of
    :return: dataframe with gameIDs and matchups
    """
    folder_check(os.getcwd() + "/data/games/" + year)  # Check we have a /games/year folder
    league_data = l_data.LeagueGameLog(season=year)
    league_data = league_data.get_data_frames()[0] [["GAME_ID", "GAME_DATE", "MATCHUP", "WL"]]
    # Add column for home team
    league_data["HOME_TEAM"] = league_data["MATCHUP"].str[:3]

    # Figure out who won
    league_data["WL"] = np.where(league_data["WL"] == "W", league_data["MATCHUP"].str.slice(start=0, stop=3),
                                 league_data["MATCHUP"].str.slice(start=-3))
    # Rename column to make it easier to understand
    league_data = league_data.rename(columns={"WL": "WINNER"})
    # Data set contains two instances for a single game one for the home team and one for away team
    # here we only take matchups with vs. instead of @ meaning we take all home team copies game
    league_data = league_data[league_data["MATCHUP"].str.contains("vs.", na=False)]
    league_data.to_csv(os.getcwd() + "/data/games/" + year + "/schedule.csv")
    return league_data


def save_league_data(year):
    """
    Saves all data needed for model for given year. Uses save_league_schedule() to get league schedule for given year
    and then feeds the given dataframe to thread_save_game_data() to process and save.

    :param year: Year we want data for
    :return: Does not return anything
    """
    # Save and get game id for all games played for given year
    schedule = save_league_schedule(year)
    # Reset global variable used for gamesProcessed to save RAM in case of multiple years saved per run.
    # DO NOT need to reset player as we are very likely to save time by keeping playerProcessed for multiple years
    # and do not need to save a players career stats twice
    global gameProcessed
    gameProcessed = set()
    # Might want to consider another threading option
    # For debugging does not seem to raise any errors even when there are some that stop function from working
    # Using schedule save data about players that played and their minutes
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        executor.map(lambda row: thread_save_game_data(year, row), schedule.itertuples(index=False))


def get_all_data(years):
    """
    Will get all data need for model for years given. Uses save_league_data(). Function also outputs time it takes to
    save data

    :param years: Expected to be a list of strings. Each with a year we want NBA data for
    :return: Does not return anything
    """
    # Save league schedule
    for year in years:
        start_time = time.time()
        print("Saving data for NBA season " + year)
        save_league_data(year)
        print("Finished saving data took " + str(time.time() - start_time) + " seconds")


def main():
    folder_setup()
    get_all_data(["2020", "2021"])


if __name__ == "__main__":
    main()
