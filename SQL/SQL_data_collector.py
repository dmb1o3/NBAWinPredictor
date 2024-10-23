from certifi import where
from numpy.lib.function_base import select
from sqlalchemy import false

from SQL import db_manager as db

def get_missing_game_data():
    # Query to get game ids and season ids for all game stats missing from schedules in schedule table
    query = """
    SELECT "GAME_ID", "SEASON_ID"
    FROM schedule s
    WHERE NOT EXISTS (
        SELECT 1 
        FROM game_stats gs
        WHERE s."GAME_ID" = gs."GAME_ID"
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



if __name__ == "__main__":
    print(get_data_from_table(["*"],"game_stats",{"PLAYER_NAME" :"= Kevin Garnett", "PTS":"> 20"}))