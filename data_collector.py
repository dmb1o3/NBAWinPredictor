# This is a file to collect data using an NBA API for a given season
# Uses tools from this repo https://github.com/swar/nba_api
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import team_data as tData
import leauge_data as lData
import boxscore_data as bData
import os

num_cores = os.cpu_count()
gameLock = Lock()  # Used to sync threads for saving game data
gameProcessed = set()  # Used to make sure thread don't process same game multiple time
teamLock = Lock()  # Used to sync threads for saving data on team_ID
teamProcessed = set()  #
# @TODO Think about where we should store who won game for validation later on


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
    Given a directory will check to see if it exists. If it does not exist make the directory requested.

    :param directory: Path of directory
    :return:
    """
    if not os.path.isdir(directory):
        # Make directory
        os.mkdir(directory)


def get_game_data(game_id):
    """

    :param game_id:
    :return:
    """
    box_score_data = bData.BoxScoreTraditionalV2(game_id=game_id)
    box_score_data = box_score_data.get_data_frames()[0][["TEAM_ID",
                                                          "TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN"]]
    # Minutes are stored with seconds. 35 minutes 30 seconds is 35.0000:30
    # The line below removes everything after the period
    box_score_data["MIN"] = box_score_data["MIN"].str.split(".").str[0]
    box_score_data = box_score_data.fillna(0)  # Data uses none instead of 0
    # Edit min to remove extra data
    return box_score_data


def thread_save_game_data(year, row):
    """
    Given a schedule will save the data for that game into .../data/games/year/

    :param year:
    :param row:
    :return:
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
    home_team = matchup[0:3]
    away_team = matchup[-3:]
    home_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(home_team)]
    away_data = game_data.loc[game_data['TEAM_ABBREVIATION'].str.contains(away_team)]
    # Make sure we have folder to save to
    directory = os.getcwd() + "/data/games/" + year + "/" + game_id
    folder_check(directory)
    folder_check(directory + "/" + home_team)
    folder_check(directory + "/" + away_team)
    # Save data
    home_data.to_csv(directory + "/" + home_team + "/minutes.csv")
    away_data.to_csv(directory + "/" + away_team + "/minutes.csv")


def save_league_schedule(year):
    """
    Will save the league schedule for all teams for the given year. Will save the csv to .../data/games/year/games.csv.
    Saves the GAME_ID and MATCHUP. We need GAME_ID but could get rid of MATCHUP keeping it for now for readability

    :param: year: String containing the year we want schedule of
    :return: dataframe with gameIDs and matchups
    """

    folder_check(os.getcwd() + "/data/games/" + year)  # Check we have a /games/year folder
    league_data = lData.LeagueGameLog(season=year)
    league_data = league_data.get_data_frames()[0][["GAME_ID", "MATCHUP"]]
    # Data set contains two instances for a single game one for the home team and one for away team
    # here we only take matchups with vs. instead of @ meaning we take all home team copies game
    league_data = league_data[league_data["MATCHUP"].str.contains("vs.", na=False)]
    league_data.to_csv(os.getcwd() + "/data/games/" + year + "/schedule.csv")
    return league_data


def save_league_data(year):
    # Save and get game id for all games played for given year
    schedule = save_league_schedule(year)
    print("Finished saving schedule")
    # Using schedule save data about players that played and their minutes
    # for threads use num_cores * 2 as api request and i/o should have some waiting time. Feel free to tweak it
    global gameProcessed
    gameProcessed = set()  # When doing multiple years want to make sure we clear this, so it doesn't get to big
    with ThreadPoolExecutor(max_workers=num_cores * 2) as executor:
        executor.map(lambda row: thread_save_game_data(year, row), schedule.itertuples(index=False))


def get_all_data(years):
    """
    Will get all data need for model to run for the years given.

    :param: years: Expected to be a list of strings. Each with a year we want NBA data for
    :return: Does not return anything
    """
    # Save league schedule
    for year in years:
        save_league_data(year)
    # Save career stats for players added


def main():
    folder_setup()
    get_all_data(["2020"])


if __name__ == "__main__":
    main()
