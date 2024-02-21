# This is a file to collect data using an NBA API for a given season
# Uses tools from this repo https://github.com/swar/nba_api
import team_data as tData
import team_schedule_collector as schedule
import leauge_data as lData
import os


# Nikola Jokić
#career = playercareerstats.PlayerCareerStats(player_id='203999')
# pandas data frames (optional: pip install pandas)
#career.get_data_frames()[0].to_csv("data/careerStats/Nikola Jokic.csv")
#print(pData.find_player_by_id("203999"))
#print(tData.find_teams_by_nickname("Suns"))
#teamData = schedule.TeamGameLog("1610612756", "2022")
#teamData.get_data_frames()[0].to_csv(os.getcwd() + "/data/Schedule/2023/Suns/Schedule.csv")

#print(teamData.get_data_frames()[0].to_csv("data/Schedule/2023/Clippers/Schedule.csv"))


def folder_check(directory):
    """
    Given a directory will check to see if it exists. If it does not exist make the directory requested.

    :param directory: Path of directory
    :return:
    """
    if not os.path.isdir(directory):
        # Make directory
        os.mkdir(directory)


def save_team_schedule(team_nickname, year):
    """
    Will save the schedule of a team based on a given team_id for a given year. Will save the data to
    ...\\data\\Schedule\\year\\teamName. The data that we will save is game_ID,

    :param team_nickname: ID of team you want schedule for. ID is from nba site and can be found by using
    :param year: String containing the year we want schedule of. If we want 2023/2024 season use string "2023"
    :return: Does not return anything
     """
    # @TODO MAKE CHECK TO SEE IF WE ALREADY HAVE DATA
    folder_check(os.getcwd() + "/data/Schedule/" + year)  # Check to see if we have year as a folder
    folder_check(os.getcwd() + "/data/Schedule/" + year + "/" + team_nickname)  # Check to see if we have team in year
    team_id = tData.find_teams_by_nickname(team_nickname)[0]['id']  # Has info on team. We just take ID
    team_data = schedule.TeamGameLog(team_id, year)
    team_data.get_data_frames()[0].to_csv(os.getcwd() + "/data/Schedule/" + year + "/" + team_nickname + "/Schedule.csv")
    return 0


def save_game_data(game_id):
    return




def save_all_team_schedule(year):
    """
    Will save the league schedule for all teams for the given year. Will save the csv to
    .../data/schedule/year/Schedule.csv.

    :param year: String containing the year we want schedule of
    :return: Does not return anything
    """
    folder_check(os.getcwd() + "/data/schedule/" + year)  # Check we have a /Schedule/year folder
    league_data = lData.LeagueGameLog(season=year)
    league_data.get_data_frames()[0].to_csv(os.getcwd() + "/data/schedule/" + year + "/Schedule.csv")

    # Using list of team names loop and save schedule


def main():
    save_all_team_schedule("2020")


if __name__ == "__main__":
    main()
