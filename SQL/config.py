config_params = {
    "dbname":"nba",
    "user":"postgres",
    "password":"YOUR_PASSWORD",
    "host":"localhost",
    "port":"5432",
}

conn_string = ("postgresql://" + config_params["user"] + ":" + config_params["password"] + "@" +
               config_params["host"] + ":" + config_params["port"] + "/" + config_params["dbname"])


# Table Definitions
schedule_table = """
CREATE TABLE schedule (
    "SEASON_ID" char(5),
    "GAME_ID" char(10) PRIMARY KEY,
    "GAME_DATE" date,
    "MATCHUP" char(11),
    "HOME_TEAM_NAME" varchar(50),
    "HOME_TEAM_ABBREVIATION" char(3),
    "HOME_TEAM_ID" char(10),
    "OPP_TEAM_NAME" varchar(50),
    "OPP_TEAM_ABBREVIATION" char(3),
    "OPP_TEAM_ID" char(10),
    "WINNER" char(3),
    "VIDEO_AVAILABLE" int,

    UNIQUE ("GAME_ID")
);
"""

team_stats_table = """
CREATE TABLE team_stats (
    "GAME_ID" char(10),
    "TEAM" char(3),
    "MIN" int,
    "FGM" int,
    "FGA" int,
    "FG_PCT" float,
    "FG3M" int,
    "FG3A" int,
    "FG3_PCT" float,
    "FTM" int,
    "FTA" int,
    "FT_PCT" float,
    "OREB" int,
    "DREB" int,
    "REB" int,
    "AST" int,
    "STL" int,
    "BLK" int,
    "TOV" int,
    "PF" int,
    "PTS" int,
    "PLUS_MINUS" int,
    
    PRIMARY KEY ("GAME_ID", "TEAM"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""

game_stats_table = """
CREATE TABLE game_stats (
    "GAME_ID" char(10),
    "TEAM_ID" char(10),
    "TEAM_ABBREVIATION" char(3),
    "TEAM_CITY" varchar(50),
    "PLAYER_ID" int,
    "PLAYER_NAME" varchar(50),
    "NICKNAME" varchar(50),
    "START_POSITION" char(1),
    "COMMENT" varchar(50),
    "MIN" int,
    "FGM" int,
    "FGA" int,
    "FG_PCT" float,
    "FG3M" int,
    "FG3A" int,
    "FG3_PCT" float,
    "FTM" int,
    "FTA" int,
    "FT_PCT" float,
    "OREB" int,
    "DREB" int,
    "REB" int,
    "AST" int,
    "STL" int,
    "BLK" int,
    "TOV" int,
    "PF" int,
    "PTS" int,
    "PLUS_MINUS" int,

    PRIMARY KEY ("GAME_ID", "PLAYER_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""