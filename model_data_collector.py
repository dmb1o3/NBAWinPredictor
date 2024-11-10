import pandas
import pandas as pd
from SQL import SQL_data_collector
from SQL.SQL_data_collector import get_data_from_table, get_team_stats_by_year, get_team_in_year, get_adv_team_stats_by_year
from SQL.db_manager import run_sql_query, run_sql_query_params

#@TODO add a function to make a B2B column give game ids and columns so it can be reused for team, player or other stats
GAMES_BACK = 5



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



def handle_year_input():
    """
    Given input from years will break down into array including dashed entries i.e.,
    inputs = "2018, 2020-2023"
    return = ["2018", "2019", "2020", "2021", "2022", "2023"]

    :param inputs: String with input from user with spaces indicating separate years they want to look at
    :return: Array of strings breaking down the input into individual indexes
    """
    inputs = input("What years did you want to use for models? If multiple can use - or , i.e 2022, 2023 or "
                       "2022-2023:")
    inputs = inputs.split(" ")
    years = []
    for inpt in inputs:
        if "-" in inpt:
            years += dash_to_individual(inpt)
        else:
            years.append(inpt)

    return years



def make_opp_column_names(stats):
    """
    Given a list of stats will return a list of column names for those stats with the prefixed attached.
    Prefix = "Player_", Stats = ["", "MIN", "PTS"], NUM_PLAYERS_PER_TEAM = 2
    output = [[PLAYER_1, PLAYER_1_MIN, PLAYER_1_PTS], [PLAYER_2, PLAYER_2_MIN, PLAYER_2_PTS]]

    :param prefix: String with prefix for all column names
    :param stats: List of strings with stats to use as column names after attaching a prefix
    :return: 2d array with each index being a list of strings containing column names created for one player
    """
    d = []
    for stat in stats:
        d.append("OPP_" + stat)

    return d



def average_stats(years):
    team_stats = {} # Key = Team Abbrev i.e LAC, Value = dataframe of stats
    stats = [""]
    # For each year get data for team and apply rolling average
    for year in years:
        # Get teams in a year
        teams = list(get_team_in_year(year)[0]) # Other value in tuple is column name but don't need
        for team_tuple in teams:
            team = team_tuple[0]
            # Get team stats for the year
            stats, column_names = get_team_stats_by_year(year, team)
            stats_df = pd.DataFrame(stats, columns=column_names)
            # Drop columns that cannot be translated to int or float. Drop GAME_ID since does not help predict
            dropped = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]
            dropped_df = stats_df[dropped]
            averaged_team_stats = stats_df.drop(dropped, axis=1)

            # Average stats
            averaged_team_stats = averaged_team_stats.rolling(GAMES_BACK).mean()
            # Shift values down 1 so that way we are not using stats from game played to predict game
            averaged_team_stats = pd.concat([dropped_df, averaged_team_stats.shift()], axis=1)
            # if team already has dataframe in team_stats dictionary append, if not add new key
            if team in team_stats:
                team_stats[team] = pd.concat([team_stats[team], averaged_team_stats], ignore_index=True)
            else:
                team_stats[team] = averaged_team_stats

    # Combine all team_data_frames
    all_averaged_stats = pd.concat(team_stats.values(), ignore_index=True)
    # Get rid of NA rows caused by rolling average
    all_averaged_stats = all_averaged_stats.dropna()
    # Combine rows in data frame so they contain opponent stats as well
    all_averaged_stats = all_averaged_stats.merge(all_averaged_stats, on='GAME_ID', suffixes=('', '_OPP'))
    # Clean duplicates
    all_averaged_stats = all_averaged_stats[all_averaged_stats['TEAM_ABBREVIATION'] != all_averaged_stats['TEAM_ABBREVIATION_OPP']]
    # Get unique game ids
    game_ids = list(all_averaged_stats["GAME_ID"].unique())
    # Get home team and winner for each game
    # Rows from away teams perspective. If LAL vs LAC tonight 2 rows for game one from LAL and LAC perspective
    # This will make it so we only have the home team perspective
    query = f"""
    SELECT "GAME_ID", "HOME_TEAM_ABBREVIATION", "WINNER"
    FROM schedule
    WHERE "GAME_ID" = ANY(%(game_ids)s)
    """
    # @TODO Implement way to replace team abbreviation with team id so model can learn teams
    home_team_win, cols = run_sql_query_params(query, {"game_ids":game_ids})
    home_team_win = pandas.DataFrame(home_team_win, columns=cols)

    all_averaged_stats = all_averaged_stats.merge(home_team_win, on='GAME_ID')
    all_averaged_stats = all_averaged_stats[all_averaged_stats['TEAM_ABBREVIATION'] == all_averaged_stats['HOME_TEAM_ABBREVIATION']]

    # Convert winner to binary for if home team won
    all_averaged_stats = all_averaged_stats.rename(columns={'WINNER': 'HOME_TEAM_WON'})
    all_averaged_stats['HOME_TEAM_WON'] = (all_averaged_stats['TEAM_ABBREVIATION'] == all_averaged_stats['HOME_TEAM_WON']).astype(int)

    # Drop rows we no longer need
    all_averaged_stats = all_averaged_stats.drop(['HOME_TEAM_ABBREVIATION', 'GAME_ID', "TEAM_NAME", "TEAM_NAME_OPP",
                                                  "TEAM_ABBREVIATION", "TEAM_ABBREVIATION_OPP"], axis=1)

    print(all_averaged_stats.to_string())

    # Reset indexes
    all_averaged_stats = all_averaged_stats.reset_index(drop=True)
    return all_averaged_stats.drop(["HOME_TEAM_WON"], axis=1),



def get_averaged_team_stats(years, keep_game_id=False):
    team_stats = {} # Key = Team Abbrev i.e LAC, Value = dataframe of stats
    stats = [""]
    # For each year get data for team and apply rolling average
    for year in years:
        # Get teams in a year
        teams = list(get_team_in_year(year)[0]) # Other value in tuple is column name but don't need
        for team_tuple in teams:
            team = team_tuple[0]
            # Get team stats for the year
            stats, column_names = get_team_stats_by_year(year, team)
            stats_df = pd.DataFrame(stats, columns=column_names)
            # Drop columns that cannot be translated to int or float. Drop GAME_ID since does not help predict
            dropped = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION"]
            dropped_df = stats_df[dropped]
            averaged_team_stats = stats_df.drop(dropped, axis=1)

            # Average stats
            averaged_team_stats = averaged_team_stats.rolling(GAMES_BACK).mean()
            # Shift values down 1 so that way we are not using stats from game played to predict game
            averaged_team_stats = pd.concat([dropped_df, averaged_team_stats.shift()], axis=1)
            # if team already has dataframe in team_stats dictionary append, if not add new key
            if team in team_stats:
                team_stats[team] = pd.concat([team_stats[team], averaged_team_stats], ignore_index=True)
            else:
                team_stats[team] = averaged_team_stats

    # Combine all team_data_frames
    all_averaged_stats = pd.concat(team_stats.values(), ignore_index=True)
    # Get rid of NA rows caused by rolling average
    all_averaged_stats = all_averaged_stats.dropna()
    # Combine rows in data frame so they contain opponent stats as well
    all_averaged_stats = all_averaged_stats.merge(all_averaged_stats, on='GAME_ID', suffixes=('', '_OPP'))
    # Clean duplicates
    all_averaged_stats = all_averaged_stats[all_averaged_stats['TEAM_ABBREVIATION'] != all_averaged_stats['TEAM_ABBREVIATION_OPP']]
    # Get unique game ids
    game_ids = list(all_averaged_stats["GAME_ID"].unique())
    # Get home team and winner for each game
    # Rows from away teams perspective. If LAL vs LAC tonight 2 rows for game one from LAL and LAC perspective
    # This will make it so we only have the home team perspective
    query = f"""
    SELECT "GAME_ID", "HOME_TEAM_ABBREVIATION", "WINNER"
    FROM schedule
    WHERE "GAME_ID" = ANY(%(game_ids)s)
    """
    home_team_win, cols = run_sql_query_params(query, {"game_ids":game_ids})
    home_team_win = pandas.DataFrame(home_team_win, columns=cols)

    all_averaged_stats = all_averaged_stats.merge(home_team_win, on='GAME_ID')
    all_averaged_stats = all_averaged_stats[all_averaged_stats['TEAM_ABBREVIATION'] == all_averaged_stats['HOME_TEAM_ABBREVIATION']]

    # Convert winner to binary for if home team won
    all_averaged_stats = all_averaged_stats.rename(columns={'WINNER': 'HOME_TEAM_WON'})
    all_averaged_stats['HOME_TEAM_WON'] = (all_averaged_stats['TEAM_ABBREVIATION'] == all_averaged_stats['HOME_TEAM_WON']).astype(int)

    drop_cols = ['HOME_TEAM_ABBREVIATION', "TEAM_NAME", "TEAM_NAME_OPP", "TEAM_ABBREVIATION",
                 "TEAM_ABBREVIATION_OPP"]

    if not keep_game_id:
        drop_cols.append('GAME_ID')

    # Drop rows we no longer need
    all_averaged_stats = all_averaged_stats.drop(drop_cols, axis=1)

    # Reset indexes
    all_averaged_stats = all_averaged_stats.reset_index(drop=True)
    return all_averaged_stats.drop(["HOME_TEAM_WON"], axis=1), all_averaged_stats["HOME_TEAM_WON"]


def get_averaged_adv_team_stats(years, keep_game_id=False):
    #@TODO fix to add minutes to final df when timestamp issue is fixed
    team_stats = {} # Key = Team Abbrev i.e LAC, Value = dataframe of stats
    stats = [""]
    # For each year get data for team and apply rolling average
    for year in years:
        # Get teams in a year
        teams = list(get_team_in_year(year)[0]) # Other value in tuple is column name but don't need
        for team_tuple in teams:
            team = team_tuple[0]
            # Get team stats for the year
            stats, column_names = get_adv_team_stats_by_year(year, team)
            stats_df = pd.DataFrame(stats, columns=column_names)
            # Drop columns that cannot be translated to int or float. Drop GAME_ID since does not help predict
            dropped = ["GAME_ID", "TEAM_ID", "TEAM_NAME", "TEAM_ABBREVIATION",  "TEAM_CITY", "MIN"]
            dropped_df = stats_df[dropped]
            averaged_adv_team_stats = stats_df.drop(dropped, axis=1)

            # Average stats
            averaged_adv_team_stats = averaged_adv_team_stats.rolling(GAMES_BACK).mean()
            # Shift values down 1 so that way we are not using stats from game played to predict game
            averaged_adv_team_stats = pd.concat([dropped_df, averaged_adv_team_stats.shift()], axis=1)
            # if team already has dataframe in team_stats dictionary append, if not add new key
            if team in team_stats:
                team_stats[team] = pd.concat([team_stats[team], averaged_adv_team_stats], ignore_index=True)
            else:
                team_stats[team] = averaged_adv_team_stats

    # Combine all team_data_frames
    all_averaged_stats = pd.concat(team_stats.values(), ignore_index=True)
    # Get rid of NA rows caused by rolling average
    all_averaged_stats = all_averaged_stats.dropna()
    # Combine rows in data frame so they contain opponent stats as well
    all_averaged_stats = all_averaged_stats.merge(all_averaged_stats, on='GAME_ID', suffixes=('', '_OPP'))
    # Clean duplicates
    all_averaged_stats = all_averaged_stats[all_averaged_stats['TEAM_ABBREVIATION'] != all_averaged_stats['TEAM_ABBREVIATION_OPP']]
    # Get unique game ids
    game_ids = list(all_averaged_stats["GAME_ID"].unique())
    # Get home team and winner for each game
    # Rows from away teams perspective. If LAL vs LAC tonight 2 rows for game one from LAL and LAC perspective
    # This will make it so we only have the home team perspective
    query = f"""
    SELECT "GAME_ID", "HOME_TEAM_ABBREVIATION", "WINNER"
    FROM schedule
    WHERE "GAME_ID" = ANY(%(game_ids)s)
    """
    home_team_win, cols = run_sql_query_params(query, {"game_ids":game_ids})
    home_team_win = pandas.DataFrame(home_team_win, columns=cols)

    all_averaged_stats = all_averaged_stats.merge(home_team_win, on='GAME_ID')
    all_averaged_stats = all_averaged_stats[all_averaged_stats['TEAM_ABBREVIATION'] == all_averaged_stats['HOME_TEAM_ABBREVIATION']]

    # Convert winner to binary for if home team won
    all_averaged_stats = all_averaged_stats.rename(columns={'WINNER': 'HOME_TEAM_WON'})
    all_averaged_stats['HOME_TEAM_WON'] = (all_averaged_stats['TEAM_ABBREVIATION'] == all_averaged_stats['HOME_TEAM_WON']).astype(int)

    drop_cols = ['HOME_TEAM_ABBREVIATION', "TEAM_NAME", "TEAM_NAME_OPP",
                 "TEAM_ABBREVIATION", "TEAM_ABBREVIATION_OPP", "TEAM_CITY", "TEAM_CITY_OPP", "MIN", "MIN_OPP"]

    if not keep_game_id:
        drop_cols.append('GAME_ID')


    # Drop rows we no longer need
    all_averaged_stats = all_averaged_stats.drop(drop_cols, axis=1)

    all_averaged_stats = all_averaged_stats.reset_index(drop=True)
    return all_averaged_stats.drop(["HOME_TEAM_WON"], axis=1), all_averaged_stats["HOME_TEAM_WON"]


def get_averaged_team_and_adv_team_stats(years):
    # @TODO when timestamp issue is fixed make sure we don't double up on min from the two team tables
    # @TODO look into why adv_team_stats and team_stats are not same length
    # Get advanced team stats
    adv_team_stats, winner = get_averaged_adv_team_stats(years, True)
    # Get team stats
    average_team_stats = get_averaged_team_stats(years, True)[0]
    # Join winner back temporally to make sure it lines up
    merged_team_stats = adv_team_stats.join(winner)
    # Combine
    merged_team_stats = merged_team_stats.merge(average_team_stats, on=["GAME_ID", "TEAM_ID", "TEAM_ID_OPP"])

    return merged_team_stats.drop(["HOME_TEAM_WON", "GAME_ID"], axis=1), merged_team_stats["HOME_TEAM_WON"]








if __name__ == "__main__":
    print(get_team_stats_by_year("2023", "WAS"))