from backend.SQL.config import config_params, conn_string, config_params_no_db
from backend.SQL.tables import team_stats_table, adv_team_stats_table, player_stats_table, adv_player_stats_table,  schedule_table
from sqlalchemy import create_engine
import psycopg2

def close_cursor_conn(cursor, conn):
    if cursor:
        cursor.close()
    if conn:
        conn.close()


def connect_to_server(autocommit):
    """
    Will create and return a connection to a database. It will use the config_params from config.py to know what
    database to connect to
    """
    try:
        conn = psycopg2.connect(**config_params) # ** Unpacks dict {"a":1, "b":2} => connect(a=1, b=2)
        # Need autocommit on for certain commands to work like database creation
        conn.autocommit = autocommit
        return conn

    except Exception as e:
        print(f"Error connecting to Postgres SQL server: {e}")


def quite_upload_df_to_postgres(df, table_name):
    db = psycopg2.connect(**config_params)
    conn = db.connect()

    try:
        df.to_sql(table_name, con=conn, if_exists="append", index=False)
    except Exception as e:
        print(f"Error uploading DataFrame: {e}")
    finally:
        conn.close()


def run_sql_query_params(query, params):
    """

    query: Query with named parameters ex. "WHERE gs."PLAYER_NAME" = %(player_name)s"
    params: Dictionary of params for query ex. {"PLAYER_NAME": "Shaquille O'Neal"}
    returns: Rows that are true for query
    """
    conn = connect_to_server(False)
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    close_cursor_conn(cursor, conn)
    columns = [desc[0] for desc in cursor.description]
    return rows, columns


def run_sql_query(query):
    conn = connect_to_server(False)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    close_cursor_conn(cursor, conn)
    columns = [desc[0] for desc in cursor.description]
    return rows, columns


def upload_df_to_postgres(df, table_name, prnt):
    db = create_engine(conn_string)
    conn = db.connect()

    try:
        df.to_sql(table_name, con=conn, if_exists="append", index=False)
        if prnt:
            print(f"\nDataframe uploaded successfully to {table_name} table")
    except Exception as e:
        print(f"Error uploading DataFrame: {e}")
    finally:
        conn.close()



def upload_csv_to_postgres(csv_file_path):
    conn = connect_to_server(True)
    cursor = conn.cursor()
    try:
        with open(csv_file_path, 'r') as f:
            next(f)  # Skip the header row if CSV has one
            cursor.copy_from(f, 'schedule', sep=',', columns=("GAME_ID", "GAME_DATE", "MATCHUP", "HOME_TEAM_ID",
                                                              "OPP_TEAM_ID", "HOME_TEAM", "WINNER", "HOME_TEAM_WON"))

        conn.commit()
        print("CSV file uploaded successfully.")

    except Exception as e:
        print(f"Error uploading CSV file: {e}")
    finally:
        close_cursor_conn(cursor, conn)


def create_table(table_schema, table_name):
    conn = connect_to_server(True)
    cursor = conn.cursor()
    try:
        cursor.execute(table_schema)
        print(f"Successfully created table {table_name}")

    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
    finally:
        close_cursor_conn(cursor, conn)


def create_database(dbname):
    try:
        conn = psycopg2.connect(**config_params_no_db) # ** Unpacks dict {"a":1, "b":2} => connect(a=1, b=2)
        # Need autocommit on for certain commands to work like database creation
        conn.autocommit = True
        cursor = conn.cursor()

    except Exception as e:
        print(f"\nError connecting to Postgres SQL server: {e}")

    try:
        cursor.execute(f"CREATE DATABASE {dbname};")
        print(f"\nSuccessfully created database {dbname}")

    except Exception as e:
        print(f"\nError creating database {dbname}: {e}")

    finally:
        close_cursor_conn(conn, cursor)


def init_database():
    # @TODO Change this to be a list so we can just loop
    create_database(config_params["dbname"])
    create_table(schedule_table, "schedule")
    create_table(team_stats_table, "team_stats")
    create_table(adv_team_stats_table, "adv_team_stats_table")
    create_table(player_stats_table, "player_stats")
    create_table(adv_player_stats_table, "adv_player_stats_table")