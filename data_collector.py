# This is a file to collect data using an NBA API for a given season
# Uses tools from this repo https://github.com/swar/nba_api
import team_data as tData
import leauge_data as lData
import boxscore_data as bData
import os


def folder_setup():
    """
    Makes sure that folders used in program are set up

    :return: Does not return anything
    """
    folder_check(os.getcwd() + "/data")  # Check we have a /games/year folder
    folder_check(os.getcwd() + "/data/games/")  # Check we have a /games/year folder


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
    box_score_data = bData.BoxScoreTraditionalV2(game_id=game_id)
    box_score_data = box_score_data.get_data_frames()[0][["TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "MIN"]]
    # Minutes are stored with seconds. 35 minutes 30 seconds is 35.0000:30
    # The line below removes everything after the period
    box_score_data["MIN"] = box_score_data["MIN"].str.split(".").str[0]
    box_score_data = box_score_data.fillna(0)  # Data uses none instead of 0
    # Edit min to remove extra data
    return box_score_data


def save_all_game_data(year, schedule):
    """
    Given a schedule will save the data for that game into .../data/games/year/

    :param schedule: Schedule of NBA games. Expecting a dataframe with two rows. One of game ids and another of matchups
    :param year: year of schedule
    :return:
    """
    for index, row in schedule.iterrows():
        print(f"Name: {row['GAME_ID']}, Age: {row['MATCHUP']}")
        # Get game data
        game_data = get_game_data(row['GAME_ID'])
        folder_check(os.getcwd() + "/data/games/" + year + "/" + row['GAME_ID'])
        game_data.to_csv(os.getcwd() + "/data/games/" + year + "/" + row['GAME_ID'] + "/minutes.csv")
        break


def save_league_schedule(year):
    """
    Will save the league schedule for all teams for the given year. Will save the csv to
    .../data/games/year/games.csv. Saves the GAME_ID and MATCHUP. We need GAME_ID but could get rid of MATCHUP
    keeping it for now for readability

    :param year: String containing the year we want schedule of
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
    # Using game ids save data for each game
    save_all_game_data(year, schedule)


def main():
    folder_setup()
    save_league_data("2022")


if __name__ == "__main__":
    main()
