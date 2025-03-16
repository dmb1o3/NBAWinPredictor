from backend.SQL import db_manager as db
import pandas as pd
# @TODO Think about combining with db manager. Might make sense logically

def does_schedule_for_year_exist(year):
    """
    Given a string representing a year will return true if database has schedule for that year and false if not
    """
    query = """
    SELECT EXISTS(
        SELECT 1
        FROM schedule s
        WHERE RIGHT("SEASON_ID", 4) = %(year)s
    );
    """
    return db.run_sql_query_params(query, {"year":year})[0][0][0]


def get_missing_game_data():
    # Query to get game ids and season ids for all game stats missing from schedules in schedule table
    query = """
    SELECT DISTINCT "GAME_ID", RIGHT("SEASON_ID",4)
    FROM schedule s
    WHERE NOT EXISTS (
        SELECT 1
        FROM player_stats ps
        WHERE s."GAME_ID" = ps."GAME_ID"
    )
    OR NOT EXISTS (
    SELECT 1
    FROM adv_player_stats aps
    WHERE s."GAME_ID" = aps."GAME_ID"
    )
    OR NOT EXISTS (
    SELECT 1
    FROM adv_team_stats ats
    WHERE s."GAME_ID" = ats."GAME_ID"
    );
    """
    return db.run_sql_query(query)


def where_builder(table_abbrev, col_conditions):
    'tn."PLAYER_NAME" = %(PLAYER_NAME)s'
    first_time = True
    where_str = ""
    for column_name, condition_val in col_conditions.items():
        condition, val = condition_val.split(" ", 1)
        col_conditions[column_name] = val
        if not first_time:
            where_str += "\n\tAND "
        else:
            first_time = False
        where_str += f'{table_abbrev}."{column_name}" {condition} %({column_name})s'
    print(where_str)

    return where_str, col_conditions



def select_builder(column_names):
    if column_names[0] == "*":
        select_str = f'{column_names[0]}'
        return select_str

    select_str = f'"{column_names[0]}"'
    # No error if out of bounds just doesn't loop
    for column in column_names[1:]:
        select_str += f', "{column}"'

    return select_str

def get_team_in_year(year):
    query = f"""
    SELECT DISTINCT "HOME_TEAM_ABBREVIATION"
    FROM schedule
    WHERE RIGHT(schedule."SEASON_ID", 4) = '{year}'
    """

    return db.run_sql_query(query)


def get_adv_team_stats_by_year(year, team_abbrev):
    query = f"""
    SELECT adv_team_stats.*
    FROM adv_team_stats
    join schedule on schedule."GAME_ID" = adv_team_stats."GAME_ID"
    WHERE RIGHT(schedule."SEASON_ID", 4) = '{year}'
    AND adv_team_stats."TEAM_ABBREVIATION" = '{team_abbrev}'
    ORDER BY schedule."GAME_DATE" ASC
    """

    return db.run_sql_query(query)


def get_team_stats_by_year(year, team_abbrev):
    query = f"""
    SELECT team_stats.*
    FROM team_stats
    join schedule on schedule."GAME_ID" = team_stats."GAME_ID"
    WHERE RIGHT(schedule."SEASON_ID", 4) = '{year}'
    AND team_stats."TEAM_ABBREVIATION" = '{team_abbrev}'
    ORDER BY schedule."GAME_DATE" ASC
    """

    return db.run_sql_query(query)


def get_game_ids_home_away_team_ids(year):
    year = "2" + year
    query = f"""
    SELECT "GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"
    FROM public.schedule
    WHERE "SEASON_ID" = '{year}'
    ORDER BY "GAME_DATE" ASC
    """

    x, column_names = db.run_sql_query(query)
    x = pd.DataFrame(x, columns=column_names)

    return x


def get_data_from_table(return_column_names, table_name, col_conditions):
    """
    Will return rows from a single table that match conditions set in col_conditions

    return_column_names: List containing columns names. If all wanted just have ["*"]
    table_name: String containing table name
    col_conditions: Dictionary set up
                    {
                        "PLAYER_NAME": "= Chris Paul",
                        "PTS":"< 30",
                        "FG_PCT": "> 0.44"
                    }

    """
    table_abbreviation = "tn"
    return_column_names = select_builder(return_column_names)
    where_str, col_conditions = where_builder(table_abbreviation, col_conditions)
    query = f"""
    SELECT {return_column_names} 
    FROM {table_name} {table_abbreviation}
    WHERE {where_str}
    """

    #     WHERE tn."PLAYER_NAME" = %(PLAYER_NAME)s
    print(query)
    print(col_conditions)

    return db.run_sql_query_params(query, col_conditions)


def get_player_stats_year_team(year, team):
    """
    Given a year and team will return a dictionary of dataframes with all player stats for players that
    played on team that year. Will include stats from players on another team if they happened to get traded mid-season.

    year: Year to get player stats for
    team: Abbreviation of team i.e LAC for
    return: Dictionary Key = player id and value = dataframe of playerstats
    """
    year = "2" + year # Years are stored with a 2 in front for season id so 2023 is 22023
    query = f"""
    SELECT ps.*
    FROM public.player_stats ps
    -- JOIN so we have season id to know what year 
    JOIN schedule s on ps."GAME_ID" = s."GAME_ID"
    WHERE s."SEASON_ID" = '{year}'
    AND ps."PLAYER_ID" in (
	    SELECT DISTINCT ps2."PLAYER_ID"
	    FROM public.player_stats ps2
	    JOIN schedule s2 on s2."GAME_ID" = ps2."GAME_ID"
	    WHERE s2."SEASON_ID" = '{year}'
	    AND ps2."TEAM_ABBREVIATION" = '{team}'
    )
    ORDER BY "GAME_ID" ASC, "PLAYER_ID" ASC 
    """

    player_stats, column_names = db.run_sql_query(query)
    player_stats = pd.DataFrame(player_stats, columns=column_names)
    # Return dictionary of players tats
    return {player_id: df_group for player_id, df_group in player_stats.groupby("PLAYER_ID")}

