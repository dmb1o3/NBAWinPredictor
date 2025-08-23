import requests
import pandas as pd
from SQL import db_manager as db

# Box score url without game_id attached at end
NBA_BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022401194.json"

HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
}


def fetch_json(url, timeout=10):
    """
    Given a url with a JSON will request the JSON from url and then return it


    """
    resp = requests.get(url, headers=HEADER, timeout=timeout)
    resp.raise_for_status()  # raises exception if request failed
    return resp.json()


def process_box_score_json(json):
    """
    Given a JSON will make dataframes equivalent to that of get_save_box_score_data(), get_save_advanced_box_score_data()
    get_save_attendance_official_misc_team_data() in NBA_data_collector.py and will upload it to the proper tables
    """
    game = json["game"]
    game_id = game["gameId"]

    # Handle Officials table
    officials = pd.DataFrame(game["officials"])
    officials["GAME_ID"] = game_id
    # Drop excess columns not from API endpoint
    officials = officials.drop(["assignment", "name", "nameI"], axis=1)
    # Rename to match in database
    officials = officials.rename(columns={"personId": "OFFICIAL_ID", "firstName": "FIRST_NAME",
                                          "familyName": "LAST_NAME", "jerseyNum": "JERSEY_NUM"})
    # Upload to DB
    #db.upload_df_to_postgres(officials, "officials", False)

    # Handle Attendance. We overwrite here since sometimes API endpoint has null value. JSON from url will never
    attendance = pd.DataFrame([{"GAME_ID":game_id, "ATTENDANCE":game["attendance"]}])
    db.upload_df_to_postgres(attendance, "attendance", True, True)






    home_team = game["homeTeam"]
    away_team = game["awayTeam"]



    print(officials.to_string())
    exit()
    # Create lists to eventual turn into df
    players = []
    attendance = []
    misc_team_stats = []
    for team in [home_team, away_team]:
        for player in team["players"]:
            row = {}
            row["GAME_ID"] = game_id
            row["TEAM_ID"] = team["teamId"]
            row["TEAM_ABBREVIATION"] = team["teamId"]
            row["TEAM_CITY"] = team["teamId"]

            row["PLAYER_ID"] = p["personId"]
            row["PLAYER_NAME"] = f"{p['firstName']} {p['lastName']}"
            players.append(row)

    df = pd.DataFrame(players)
    print(df.head())

    return df




data = fetch_json(NBA_BOXSCORE_URL)
process_box_score_json(data)
