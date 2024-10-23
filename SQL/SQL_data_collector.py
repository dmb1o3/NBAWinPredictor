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


def get_data_for_player_by_name(player_name):
    """
    Given a string with a player name will return the rows containing

    """
    query = f"""
    SELECT * 
    FROM game_stats gs
    WHERE gs."PLAYER_NAME" = %(player_name)s
    """
    return db.run_sql_query_params(query, player_name)



if __name__ == "__main__":
    print(get_data_for_player_by_name({"player_name" :"Shaquille O'Neal"}))