from retrying import retry
from API import league_data as l_data
from SQL import db_manager as db
import pandas as pd
import numpy as np


MAX_DOWNLOAD_ATTEMPTS = 4  # Set to -1 for infinite. Controls number of times to try to download from NBA api



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
def get_league_schedule(year):
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


def main():
    # Get information from user, so we know what seasons to download and/or prepare data for
    # Also asks user if they already have data downloaded, so we can skip download and skip to preparing that data
    years = input("What years would you like to download/prepare? If multiple just type them with a space like \"2020 "
                  "2021 2022\" ")
    years = handle_year_input(years)
    for year in years:
        # Download schedule from NBA API
        schedule, team_data = get_league_schedule(year)
        # Upload schedule and team data for season to database
        db.upload_df_to_postgres(schedule, "schedule")
        db.upload_df_to_postgres(team_data, "team_stats")
        print("League schedule for " + year + " has been saved")
        # Use schedule to get games




if __name__ == "__main__":
    main()