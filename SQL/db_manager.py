from psycopg2 import connect
import pandas as pd
from SQL.config import team_stats_table, game_stats_table, config_params, conn_string, schedule_table
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
    return rows


def run_sql_query(query):
    conn = connect_to_server(False)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    close_cursor_conn(cursor, conn)
    return rows


def upload_df_to_postgres(df, table_name):
    db = create_engine(conn_string)
    conn = db.connect()

    try:
        df.to_sql(table_name, con=conn, if_exists="append", index=False)
        print(f"Dataframe uploaded successfully to {table_name} table")
    except Exception as e:
        print(f"Error uploading DataFrame: {e}")
    finally:
        conn.close()



def upload_csv_to_postgres(csv_file_path):
    conn = connect_to_server(True)
    cursor = conn.cursor()
    try:
        # Use the COPY command to load the CSV data
        with open(csv_file_path, 'r') as f:
            next(f)  # Skip the header row if your CSV has one
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
    conn = connect_to_server(True)
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE DATABASE {dbname};")
        print(f"Successfully created database {dbname}")

    except Exception as e:
        print(f"Error creating database {dbname}: {e}")

    finally:
        close_cursor_conn(conn, cursor)


if __name__ == "__main__":
    create_table(schedule_table, "schedule")
    create_table(team_stats_table, "team_stats")
    create_table(game_stats_table, "game_stats")



   # upload_csv_to_postgres(r"C:\Users\mrjoy\PycharmProjects\NBAWinPredictor\data\games\2022\schedule.csv")