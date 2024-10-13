from psycopg2 import connect

from config import  config_params

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


def upload_csv_to_postgres(csv_file_path):
    conn = connect_to_server(True)
    cursor = conn.cursor()
    try:
        # Use the COPY command to load the CSV data
        with open(csv_file_path, 'r') as f:
            next(f)  # Skip the header row if your CSV has one
            cursor.copy_from(f, 'schedule', sep=',', columns=('game_id', 'game_date', 'matchup',
                                                              'home_team_id', 'opp_team_id',
                                                              'home_team', 'winner', 'home_team_won'))

        conn.commit()
        print("CSV file uploaded successfully.")

    except Exception as e:
        print(f"Error uploading CSV file: {e}")
    finally:
        close_cursor_conn(cursor, conn)


def create_table(table_name):
    conn = connect_to_server(True)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE schedule (
            game_id char(10) PRIMARY KEY,
            game_date date,
            matchup char(11),
            home_team_id char(10),
            opp_team_id char(10),
            home_team char(3),
            winner char(3),
            home_team_won int
        );
        """)
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
    create_table("schedule")
    upload_csv_to_postgres(r"C:\Users\mrjoy\PycharmProjects\NBAWinPredictor\data\games\2023\schedule.csv")