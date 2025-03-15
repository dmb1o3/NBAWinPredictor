import pandas
import pandas as pd
from backend.SQL.config import PLAYERS_PER_TEAM
from backend.SQL import SQL_data_collector
import backend.SQL.SQL_data_collector as sdc
from backend.SQL.db_manager import run_sql_query, run_sql_query_params

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


def get_averaged_team_stats(years, keep_game_id=False):
    team_stats = {} # Key = Team Abbrev i.e LAC, Value = dataframe of stats
    stats = [""]
    # For each year get data for team and apply rolling average
    for year in years:
        # Get teams in a year
        teams = list(sdc.get_team_in_year(year)[0]) # Other value in tuple is column name but don't need
        for team_tuple in teams:
            team = team_tuple[0]
            # Get team stats for the year
            stats, column_names = sdc.get_team_stats_by_year(year, team)
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
        teams = list(sdc.get_team_in_year(year)[0]) # Other value in tuple is column name but don't need
        for team_tuple in teams:
            team = team_tuple[0]
            # Get team stats for the year
            stats, column_names = sdc.get_adv_team_stats_by_year(year, team)
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


def get_averaged_team_and_adv_team_stats(years, keep_game_id=False):
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

    # Get game ids
    game_ids = list(merged_team_stats["GAME_ID"])
    b2b_df = get_back_2_back(game_ids)
    # combine
    merged_team_stats = merged_team_stats.merge(b2b_df, on=["GAME_ID"])
    merged_team_stats.to_csv('test.csv', index=False)

    if keep_game_id:
        return merged_team_stats.drop(["HOME_TEAM_WON"], axis=1), merged_team_stats["HOME_TEAM_WON"]

    return merged_team_stats.drop(["HOME_TEAM_WON", "GAME_ID"], axis=1), merged_team_stats["HOME_TEAM_WON"]

def clean_player_stats_apply_roll_avg_shift(player_stats, games_back):
    """
    Given a dictionary containing dataframes of player stats will go through each data frame, clean and prepare it for
    model. Will also apply a rolling average and shift the stats back a game. This way if we use the stats for
    predictive modeling we are using rolling average from previous games with the current game not included.
    """
    player_ids = list(player_stats.keys())
    for player_id in player_ids:
        # Make started column which when rolling average is applied will show percentage player started
        player_stats[player_id]["STARTED"] = player_stats[player_id]['START_POSITION'].apply(lambda x: 1 if x != ' ' else 0)

        # Convert minutes from time stamp to float
        player_stats[player_id]['MIN'] = player_stats[player_id]['MIN'].dt.total_seconds() / 60

        # Drop rows that are not useful for models
        drop_cols = ["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_CITY", "PLAYER_NAME", "NICKNAME", "COMMENT", "START_POSITION"]
        player_stats[player_id] = player_stats[player_id].drop(drop_cols, axis=1)

        # Apply rolling average to columns that can be averaged
        non_rolling_cols = ["GAME_ID", "PLAYER_ID"]
        rolling_cols = player_stats[player_id].select_dtypes(include=[float, int])
        # GAME_ID is not float or int it's a string
        rolling_cols.drop(columns=["PLAYER_ID"], inplace=True)
        rolling_cols = rolling_cols.columns
        # Apply shift for stats so for each GAME_ID contains data from games before
        player_stats[player_id] = pd.concat([player_stats[player_id][non_rolling_cols],
                                                  player_stats[player_id][rolling_cols].shift()], axis=1)
        # Apply rolling average
        player_stats[player_id] = pd.concat([player_stats[player_id][non_rolling_cols],
                                                  player_stats[player_id][rolling_cols].rolling(games_back).mean()],
                                                  axis=1)

        # Drop columns with NA
        player_stats[player_id] = player_stats[player_id].dropna()


    return player_stats

def get_averaged_player_stats(years, rolling_average=5, players_per_team=PLAYERS_PER_TEAM, keep_game_id=False):
    team_dataframes = {} # Key = Team Abbrev i.e LAC value = data frame of averaged player stats for whole team
    player_stats = {} # Key = Player id, Value = dataframe of average stats for one player
    # We loop on years as a team is not guaranteed to exist next season i.e seattle supersonics
    for year in years:
        # Get teams that played in the year
        teams = list(sdc.get_team_in_year(year)[0]) # Other value in tuple is column name but don't need
        # For each team collect average stats of players that year
        for team_tuple in teams:
            team = team_tuple[0]
            # Get player stats for that year of all players on team
            player_stats = sdc.get_player_stats_year_team(year, team)
            pd.set_option('display.max_columns', None)
            player_stats = clean_player_stats_apply_roll_avg_shift(player_stats, rolling_average)
            exit()
            stats, column_names = sdc.get_adv_team_stats_by_year(year, team)
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
            if team in player_stats:
                player_stats[team] = pd.concat([player_stats[team], averaged_adv_team_stats], ignore_index=True)
            else:
                player_stats[team] = averaged_adv_team_stats

    # Combine all team_data_frames
    all_averaged_stats = pd.concat(player_stats.values(), ignore_index=True)
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

    print(all_averaged_stats.to_string())
    exit()

    # Drop rows we no longer need
    all_averaged_stats = all_averaged_stats.drop(drop_cols, axis=1)

    all_averaged_stats = all_averaged_stats.reset_index(drop=True)
    return all_averaged_stats.drop(["HOME_TEAM_WON"], axis=1), all_averaged_stats["HOME_TEAM_WON"]


def get_back_2_back(game_ids):
    """
    Given a list of game_ids will return three columns in this order GAME_ID, HOME_TEAM_B2B, OPP_TEAM_B2B
    """
    query = """
    WITH game_details AS (
        SELECT 
            s."GAME_ID",
            s."HOME_TEAM_ID",
            s."AWAY_TEAM_ID",
            s."GAME_DATE",
            -- Check if home team played the previous day
            CASE WHEN EXISTS (
                SELECT 1 
                FROM schedule prev 
                WHERE (prev."HOME_TEAM_ID" = s."HOME_TEAM_ID" OR prev."AWAY_TEAM_ID" = s."HOME_TEAM_ID")
                  AND prev."GAME_DATE" = s."GAME_DATE" - INTERVAL '1 day'
            ) THEN 1 ELSE 0 END AS "HOME_TEAM_B2B",
            -- Check if away team played the previous day
            CASE WHEN EXISTS (
                SELECT 1 
                FROM schedule prev 
                WHERE (prev."HOME_TEAM_ID" = s."AWAY_TEAM_ID" OR prev."AWAY_TEAM_ID" = s."AWAY_TEAM_ID")
                  AND prev."GAME_DATE" = s."GAME_DATE" - INTERVAL '1 day'
            ) THEN 1 ELSE 0 END AS "AWAY_TEAM_B2B"
        FROM schedule s
        WHERE s."GAME_ID" = ANY(%(game_ids)s)
    )
    SELECT "GAME_ID", "HOME_TEAM_B2B", "AWAY_TEAM_B2B"
    FROM game_details 
    ORDER BY "GAME_DATE", "GAME_ID"; 
    """

    b2b_data, cols = run_sql_query_params(query, {"game_ids":game_ids})
    b2b_data = pandas.DataFrame(b2b_data, columns=cols)
    return b2b_data


def get_win_streak(game_ids):
    """
    Given a list of game_ids will return three columns in this order GAME_ID, HOME_TEAM_WIN_STREAK, OPP_TEAM_B2B
    """
    query = """
    WITH game_details AS (
        SELECT 
            s."GAME_ID",
            s."HOME_TEAM_ID",
            s."AWAY_TEAM_ID",
            s."GAME_DATE",
            -- Check if home team played the previous day
            CASE WHEN EXISTS (
                SELECT 1 
                FROM schedule prev 
                WHERE (prev."HOME_TEAM_ID" = s."HOME_TEAM_ID" OR prev."AWAY_TEAM_ID" = s."HOME_TEAM_ID")
                  AND prev."GAME_DATE" = s."GAME_DATE" - INTERVAL '1 day'
            ) THEN 1 ELSE 0 END AS "HOME_TEAM_B2B",
            -- Check if away team played the previous day
            CASE WHEN EXISTS (
                SELECT 1 
                FROM schedule prev 
                WHERE (prev."HOME_TEAM_ID" = s."AWAY_TEAM_ID" OR prev."AWAY_TEAM_ID" = s."AWAY_TEAM_ID")
                  AND prev."GAME_DATE" = s."GAME_DATE" - INTERVAL '1 day'
            ) THEN 1 ELSE 0 END AS "AWAY_TEAM_B2B"
        FROM schedule s
        WHERE s."GAME_ID" = ANY(%(game_ids)s)
    )
    SELECT "GAME_ID", "HOME_TEAM_B2B", "AWAY_TEAM_B2B"
    FROM game_details 
    ORDER BY "GAME_DATE", "GAME_ID"; 
    """

    b2b_data, cols = run_sql_query_params(query, {"game_ids":game_ids})
    b2b_data = pandas.DataFrame(b2b_data, columns=cols)
    return b2b_data


def get_home_away_team_pts(game_ids):
    """
    Given a list of game_ids will return the home team points and away team points for all game_ids provided

    """

    query = """
    SELECT
        s."GAME_ID",
        home."PTS" AS "HOME_TEAM_PTS",
        away."PTS" as "AWAY_TEAM_PTS"
    FROM schedule s
    JOIN team_stats home
        ON s."GAME_ID" = home."GAME_ID"
        AND s."HOME_TEAM_ABBREVIATION" = home."TEAM_ABBREVIATION"
    JOIN team_stats away
        ON s."GAME_ID" = away."GAME_ID"
        AND s."AWAY_TEAM_ABBREVIATION" = away."TEAM_ABBREVIATION"
    WHERE s."GAME_ID" = ANY(%(game_ids)s)
    ORDER BY s."GAME_DATE", s."GAME_ID"
    """
    points_data, cols = run_sql_query_params(query, {"game_ids":game_ids})
    points_data = pandas.DataFrame(points_data, columns=cols)
    return points_data


def get_player_stats(year, player_id):
    query = """
    SELECT ps.* 
    FROM player_stats ps
    join schedule s on ps."GAME_ID" = s."GAME_ID"
    WHERE ps."PLAYER_ID" = %(player_id)s
    AND RIGHT(s."SEASON_ID", 4) = %(year)s
    ORDER BY s."GAME_DATE", s."GAME_ID"
    """
    return run_sql_query_params(query, {"player_id":player_id, "year":year})

if __name__ == "__main__":
    print(get_home_away_team_pts(["0021800001","0021800002", "0021800003"]))