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
    "AWAY_TEAM_NAME" varchar(50),
    "AWAY_TEAM_ABBREVIATION" char(3),
    "AWAY_TEAM_ID" char(10),
    "WINNER" char(3),
    "VIDEO_AVAILABLE" int,

    UNIQUE ("GAME_ID")
);
"""

team_stats_table = """
CREATE TABLE team_stats (
    "GAME_ID" char(10),
    "TEAM_ID" char(10),
    "TEAM_NAME"  varchar(50),
    "TEAM_ABBREVIATION" char(3),
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

    PRIMARY KEY ("GAME_ID", "TEAM_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""

adv_team_stats_table = """
CREATE TABLE adv_team_stats (
    "GAME_ID" char(10),
    "TEAM_ID" char(10),
    "TEAM_NAME"  varchar(50),
    "TEAM_ABBREVIATION" char(3),
    "TEAM_CITY" varchar(50),
    "MIN" INTERVAL,
    "E_OFF_RATING" float,
    "OFF_RATING" float,  
    "E_DEF_RATING" float,
    "DEF_RATING" float,
    "E_NET_RATING" float,
    "NET_RATING" float,
    "AST_PCT" float,
    "AST_TOV" float,
    "AST_RATIO" float,
    "OREB_PCT" float,
    "DREB_PCT" float,
    "REB_PCT" float,
    "E_TM_TOV_PCT" float,
    "TM_TOV_PCT" float,
    "EFG_PCT" float,  
    "TS_PCT" float,  
    "USG_PCT" float,  
    "E_USG_PCT" float,  
    "E_PACE" float,   
    "PACE" float,  
    "PACE_PER40" float,  
    "POSS" int,
    "PIE"  float,

    PRIMARY KEY ("GAME_ID", "TEAM_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""

player_stats_table = """
CREATE TABLE player_stats (
    "GAME_ID" char(10),
    "TEAM_ID" char(10),
    "TEAM_ABBREVIATION" char(3),
    "TEAM_CITY" varchar(50),
    "PLAYER_ID" int,
    "PLAYER_NAME" varchar(50),
    "NICKNAME" varchar(50),
    "START_POSITION" char(1),
    "COMMENT" varchar(50),
    "MIN" INTERVAL,
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

adv_player_stats_table = """
CREATE TABLE adv_player_stats (
    "GAME_ID" char(10),
    "TEAM_ID" char(10),
    "TEAM_ABBREVIATION" char(3),
    "TEAM_CITY" varchar(50),
    "PLAYER_ID" int,
    "PLAYER_NAME" varchar(50),
    "NICKNAME" varchar(50),
    "START_POSITION" char(1),
    "COMMENT" varchar(50),
    "MIN" INTERVAL,
    "E_OFF_RATING" float,
    "OFF_RATING" float,  
    "E_DEF_RATING" float,
    "DEF_RATING" float,
    "E_NET_RATING" float,
    "NET_RATING" float,
    "AST_PCT" float,
    "AST_TOV" float,
    "AST_RATIO" float,
    "OREB_PCT" float,
    "DREB_PCT" float,
    "REB_PCT" float,
    "TM_TOV_PCT" float,
    "EFG_PCT" float,  
    "TS_PCT" float,  
    "USG_PCT" float,  
    "E_USG_PCT" float,  
    "E_PACE" float,   
    "PACE" float,  
    "PACE_PER40" float,  
    "POSS" int,
    "PIE"  float,

    PRIMARY KEY ("GAME_ID", "PLAYER_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""

officials_table = """
CREATE TABLE officials (
    "GAME_ID" char(10),
    "OFFICIAL_ID" varchar(50),
    "FIRST_NAME" varchar(50),
    "LAST_NAME" varchar(50),
    "JERSEY_NUM" varchar(50),
    
    PRIMARY KEY ("GAME_ID", "OFFICIAL_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""

attendance_table = """
CREATE TABLE attendance (
    "GAME_ID" char(10),
    "ATTENDANCE" int,

    PRIMARY KEY ("GAME_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""

misc_team_stats_table = """
CREATE TABLE misc_team_stats (
    "GAME_ID" char(10),
    "TEAM_ID" char(10),
    "PTS_PAINT" int,
    "PTS_2ND_CHANCE" int,
    "PTS_FB" int,
    "LARGEST_LEAD" int,
    "LEAD_CHANGES" int,
    "TIMES_TIED" int,
    "TEAM_TURNOVERS" int,
    "TOTAL_TURNOVERS" int,
    "TEAM_REBOUNDS" int,
    "PTS_OFF_TO" int,
    "TEAM_WINS_LOSSES" varchar(50),
    "PTS_QTR1" int,
    "PTS_QTR2" int,
    "PTS_QTR3" int,
    "PTS_QTR4" int,
    "PTS_OT1" int,
    "PTS_OT2" int,
    "PTS_OT3" int,
    "PTS_OT4" int,
    "PTS_OT5" int,
    "PTS_OT6" int,
    "PTS_OT7" int,
    "PTS_OT8" int,
    "PTS_OT9" int,
    "PTS_OT10" int,    
        
    PRIMARY KEY ("GAME_ID", "TEAM_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("GAME_ID")
);
"""