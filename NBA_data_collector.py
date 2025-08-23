from API import league_data as l_data, box_score_data as b_data, advanced_box_score_data as adv_b_data, box_score_summary_v2 as bss_data
from concurrent.futures import ThreadPoolExecutor, as_completed
from SQL import db_manager as db, SQL_data_collector as dc
from threading import Lock
from retrying import retry
import pandas as pd
import numpy as np
import time


# How many threads to use when downloading individual game data from NBA API
NUM_THREADS = 4 # If put above 4 way more likely for threads to timeout
# Max amount of times to retry download
MAX_DOWNLOAD_ATTEMPTS = 5
# Max amount of time a thread will wait
WAIT_MAX = 30000 # In milliseconds
# If at 1000 1s, 2s, 4s, 8s ...
WAIT_MULTIPLIER = 1000 # In milliseconds
DELAY = 5000 # In milliseconds
# Max jitter
MAX_JITTER = 2000 # In milliseconds
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


def minute_sec_decompress(time_str):
    time_str = str(time_str)
    if time_str == "0":
        return time_str

    minutes, seconds = time_str.split(':')
    # Get rid of excess ex 36 min is 36.00000
    minutes = minutes.split('.')[0]

    # Check to see if at 60 seconds and if so increase minutes by one and set seconds to 00
    if seconds == "60":
        seconds = "00"
        minutes = int(minutes) + 1

    # Recombine and return
    return f"{minutes} minutes {seconds} seconds"


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS, wait_exponential_multiplier=WAIT_MULTIPLIER,
       wait_exponential_max=WAIT_MAX, wait_jitter_max=MAX_JITTER)
def get_save_advanced_box_score_data(game_id):
    try:
        b_score_data = adv_b_data.BoxScoreAdvancedV2(game_id=game_id)
        player_data = b_score_data.get_data_frames()[0]
        team_data = b_score_data.get_data_frames()[1]

        # Change minutes from 36.00000:35 to 36:35
        player_data = player_data.fillna(0)
        player_data["MIN"] = player_data["MIN"].apply(minute_sec_decompress)
        team_data = team_data.fillna(0)
        team_data["MIN"] = team_data["MIN"].apply(minute_sec_decompress)

        # Upload player and team data to database
        db.upload_df_to_postgres(player_data, "adv_player_stats", False)
        db.upload_df_to_postgres(team_data, "adv_team_stats", False)
    except Exception as e:
        print(str(e) + " for " + str(game_id) + " in get_save_advanced_box_score_data()")
        raise


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS, wait_exponential_multiplier=WAIT_MULTIPLIER,
       wait_exponential_max=WAIT_MAX, wait_jitter_max=MAX_JITTER)
def get_save_box_score_data(game_id):
    try:
        b_score_data = b_data.BoxScoreTraditionalV2(game_id=game_id)
        game_data = b_score_data.get_data_frames()[0]

        # Minutes are stored with seconds. 35 minutes 30 seconds is 35.0000:30
        game_data = game_data.fillna(0)  # Data uses none instead of 0
        game_data["MIN"] = game_data["MIN"].apply(minute_sec_decompress)
        game_data = game_data.rename(columns={"TO": "TOV"})
        print(game_data)
        # Upload game stat to database
        db.upload_df_to_postgres(game_data, "player_stats", False)
    except Exception as e:
        print(str(e) + " for " + str(game_id) + " in get_save_box_score_data()")
        raise

@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS, wait_exponential_multiplier=WAIT_MULTIPLIER,
       wait_exponential_max=WAIT_MAX, wait_jitter_max=MAX_JITTER)
def get_save_attendance_official_misc_team_data(game_id):
    """
    This will get and save attendance, refree and some miscelanous team stats data to database

    """
    try:
        # Get data
        bss_score_data = bss_data.BoxScoreSummaryV2(game_id=game_id)
        bss_score_data = bss_score_data.get_data_frames()
        # Combine team stats
        hustle_stats = bss_score_data[1]
        pts_per_qtr = bss_score_data[5]
        misc_stats = pd.merge(hustle_stats, pts_per_qtr, on=["TEAM_ID"])
        misc_stats["GAME_ID"] = game_id
        misc_stats = misc_stats.drop(["GAME_DATE_EST", "GAME_SEQUENCE", "LEAGUE_ID", "PTS", "TEAM_NICKNAME",
                                      "TEAM_CITY_NAME", "TEAM_ABBREVIATION_y", "TEAM_ABBREVIATION_x", "TEAM_CITY"], axis=1)
        # Handle stats for referees
        officials = bss_score_data[2]
        officials["GAME_ID"] = game_id
        # Handle Attendance
        attendance = bss_score_data[4][["ATTENDANCE"]].copy() # Copy keeps as dataframe instead of slice
        attendance["GAME_ID"] = game_id
        # @TODO come back and look at think suppose to be series stats but weird
        #print(bss_score_data[7].to_string())
        # Upload data to database
        print(officials.to_string())
        print(attendance.to_string())
        print(misc_stats.to_string())
        db.upload_df_to_postgres(officials, "officials", False)
        db.upload_df_to_postgres(attendance, "attendance", False)
        db.upload_df_to_postgres(misc_stats, "misc_team_stats", False)

    except Exception as e:
        print(str(e) + " for " + str(game_id) + " in get_save_attendance_official_misc_team_data()")
        raise



def threaded_get_save_all_game_data(game_id, year):
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

        # Get Box Score data
        get_save_box_score_data(game_id)
        get_save_attendance_official_misc_team_data(game_id)
        # Get advanced stats in testing prior to 1995 returns no data
        if int(year) > 1995:
            get_save_advanced_box_score_data(game_id)

    except Exception as e:
        print(str(e) + " for " + str(game_id) + " in threaded_get_save_all_game_data")


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS, wait_exponential_multiplier=WAIT_MULTIPLIER,
       wait_exponential_max=WAIT_MAX, wait_jitter_max=MAX_JITTER)
def get_save_league_schedule_team_stats(year):
    team_stats_cols = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION", "MIN",
             "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
             "FT_PCT", "OREB", "DREB","REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"]


    league_data = l_data.LeagueGameLog(season=year)
    league_data = league_data.get_data_frames()[0]

    team_data = league_data[team_stats_cols]

    # Add away team data
    df_at = league_data[league_data['MATCHUP'].str.contains('@')][['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME']].rename(
        columns={'TEAM_ID': 'AWAY_TEAM_ID', 'TEAM_ABBREVIATION': 'AWAY_TEAM_ABBREVIATION', 'TEAM_NAME': 'AWAY_TEAM_NAME' })
    # Merge DataFrames on 'GAME_ID'
    league_data = pd.merge(league_data, df_at, on='GAME_ID', how='left')

    # Figure out who won
    league_data["WL"] = np.where(league_data["WL"] == "W", league_data["MATCHUP"].str.slice(start=0, stop=3),
                                 league_data["MATCHUP"].str.slice(start=-3))
    # Rename column to make it easier to understand
    league_data = league_data.rename(columns={"WL": "WINNER", "TEAM_ID": "HOME_TEAM_ID", "TEAM_NAME": "HOME_TEAM_NAME",
                                              "TEAM_ABBREVIATION": "HOME_TEAM_ABBREVIATION"})


    # Data set contains two instances for a single game, one for the home team and one for the away team
    # here we only take matchups with vs. instead of @ meaning we take all home team copies game
    league_data = league_data[league_data["MATCHUP"].str.contains("vs.", na=False)]
    league_data["GAME_DATE"] = pd.to_datetime(league_data["GAME_DATE"])
    league_data.drop(team_stats_cols[4:], axis=1, inplace=True)

    # Change order so it's more readable for humans
    desired_order = ["SEASON_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "HOME_TEAM_NAME", "HOME_TEAM_ABBREVIATION",
                     "HOME_TEAM_ID", "AWAY_TEAM_NAME", "AWAY_TEAM_ABBREVIATION", "AWAY_TEAM_ID", "WINNER","VIDEO_AVAILABLE"]
    league_data = league_data.reindex(columns=desired_order)


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
    games_no_game_stats = dc.get_missing_game_data() # Don't need col names so take first return value only
    print("\nMissing player stats for " + str(len(games_no_game_stats["player_stats"])) + " games")
    print("\nMissing advanced stats for " + str(len(games_no_game_stats["adv_stats"])) + " games")
    print("\nMissing official attendance_misc_stats stats for " +
          str(len(games_no_game_stats["official_attendance_misc_stats"])) + " games\n")


    functions_and_data = [
        (get_save_box_score_data, games_no_game_stats["player_stats"]),
        (get_save_advanced_box_score_data, games_no_game_stats["adv_stats"]),
        (get_save_attendance_official_misc_team_data, games_no_game_stats["official_attendance_misc_stats"])
    ]

    print(games_no_game_stats["player_stats"])

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for func, lst in functions_and_data:
            for item in lst:
                futures.append(executor.submit(func, item[0]))

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Task failed: {e}")


def get_save_data_for_year(year):
    # Download schedule from NBA API
    schedule, team_data = get_save_league_schedule_team_stats(year)
    # Upload schedule and team data for season to database
    db.upload_df_to_postgres(schedule, "schedule", True)
    db.upload_df_to_postgres(team_data, "team_stats", False)

    # Use schedule to get games
    game_ids = list(schedule["GAME_ID"])

    # Download data for each game and save it to database
    print("Starting download of games for " + year)
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(lambda game_id:threaded_get_save_all_game_data(game_id, year), game_ids)
    print("Finished took " + str((time.time() - start_time) / 60) + " minutes\n")


def set_up_year_function():
    year_input = input("\nWhat year(s) would you like to download data for? If multiple can use - or , i.e 2022, 2023 or "
                       "2022-2023: ")
    years = handle_year_input(year_input)
    for year in years:
        if dc.does_schedule_for_year_exist(year):
            print(f"\nSchedule already exists for {year} checking if any new games for {year}")
            #db.upload_new_part_df_to_postgres(team_stats, "team_stats", False)
            check_save_missing_game_stats()
        else:
            get_save_data_for_year(year)


def invalid_option(options_length):
    print("\nInvalid option must be a number from 1 - " + str(options_length - 1) + " or q/Q to exit\n")


def menu_options():
    # @TODO fix how we check for missing game data. If current year and more games are played since last saved. Will not save those games played
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

        print("")



if __name__ == "__main__":
    #get_save_box_score_data("0022401184")
    #get_save_attendance_official_misc_team_data("0022401161")
    menu_options()
