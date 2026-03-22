# Table Definitions
schedule_table = """
CREATE TABLE schedule (
    "season_id" char(5),
    "game_id" char(10) PRIMARY KEY,
    "game_date" date,
    "home_team_id" char(10),
    "away_team_id" char(10),
    "video_available" int,

    UNIQUE ("game_id")
);
"""

team_stats_table = """
CREATE TABLE team_stats (
    "game_id" char(10),
    "team_id" char(10),
    "minutes" INT,
    "field_goals_made" INT,
    "field_goals_attempted" INT,
    "three_pointers_made" INT,
    "three_pointers_attempted" INT,
    "free_throws_made" INT,
    "free_throws_attempted" INT,
    "offensive_rebounds" INT,
    "defensive_rebounds" INT,
    "assists" INT,
    "steals" INT,
    "blocks" INT,
    "turnovers" INT,
    "personal_fouls" INT,
    "points" INT,                                   
    "plus_minus" int,                   

    PRIMARY KEY ("game_id", "team_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

teams_starter_vs_bench_stats_table = """
CREATE TABLE team_starter_vs_bench_stats (
    "game_id" char(10),
    "team_id" char(10),
    "minutes" INTERVAL,
    "field_goals_made" INT,
    "field_goals_attempted" INT,
    "three_pointers_made" INT,
    "three_pointers_attempted" INT,
    "free_throws_made" INT,
    "free_throws_attempted" INT,
    "offensive_rebounds" INT,
    "defensive_rebounds" INT,
    "assists" INT,
    "steals" INT,
    "blocks" INT,
    "turnovers" INT,
    "personal_fouls" INT,
    "points" INT,
    "starter_bench" VARCHAR(8),
    
    PRIMARY KEY ("game_id", "team_id", "starter_bench"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

adv_team_stats_table = """
CREATE TABLE adv_team_stats (
    "game_id" char(10),
    "team_id" char(10),
    "minutes" INTERVAL,
    "estimated_offensive_rating" float,
    "offensive_rating" float,  
    "estimated_defensive_rating" float,
    "defensive_rating" float,
    "estimated_net_rating" float,
    "net_rating" float,
    "assist_percentage" float,
    "assist_to_turnover" float,
    "assist_ratio" float,
    "offensive_rebound_percentage" float,
    "defensive_rebound_percentage" float,
    "rebound_percentage" float,
    "estimated_team_turnover_percentage" float,
    "turnover_ratio" float,
    "effective_field_goal_percentage" float,  
    "true_shooting_percentage" float,  
    "usage_percentage" float,  
    "estimated_usage_percentage" float,  
    "estimated_pace" float,   
    "pace" float,  
    "pace_per_40" float,  
    "possessions" int,
    "PIE"  float,

    PRIMARY KEY ("game_id", "team_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

player_stats_table = """
CREATE TABLE player_stats (
    "game_id" char(10),
    "team_id" char(10),
    "player_id" int,
    "starting_position" char(1),
    "comment" varchar(50),
    "minutes" INTERVAL,
    "field_goals_made" int,
    "field_goals_attempted" int,
    "three_pointers_made" int,
    "three_pointers_attempted" int,
    "free_throws_made" int,
    "free_throws_attempted" int,
    "offensive_rebounds" INT,
    "defensive_rebounds" INT,
    "assists" INT,
    "steals" INT,
    "blocks" INT,
    "turnovers" INT,
    "personal_fouls" INT,
    "points" INT,
    "plus_minus" int,

    PRIMARY KEY ("game_id", "player_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

adv_player_stats_table = """
CREATE TABLE adv_player_stats (
    "game_id" char(10),
    "team_id" char(10),
    "player_id" int,
    "starting_position" char(1),
    "minutes" INTERVAL,
    "estimated_offensive_rating" float,
    "offensive_rating" float,  
    "estimated_defensive_rating" float,
    "defensive_rating" float,
    "estimated_net_rating" float,
    "net_rating" float,
    "assist_percentage" float,
    "assist_to_turnover" float,
    "assist_ratio" float,
    "offensive_rebound_percentage" float,
    "defensive_rebound_percentage" float,
    "rebound_percentage" float,
    "turnover_ratio" float,
    "effective_field_goal_percentage" float,  
    "true_shooting_percentage" float,  
    "usage_percentage" float,  
    "estimated_usage_percentage" float,  
    "estimated_pace" float,   
    "pace" float,  
    "pace_per_40" float,  
    "possessions" int,
    "PIE"  float,

    PRIMARY KEY ("game_id", "player_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
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
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("game_id")
);
"""

attendance_table = """
CREATE TABLE attendance (
    "GAME_ID" char(10),
    "ATTENDANCE" int,

    PRIMARY KEY ("GAME_ID"),
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("game_id")
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
    FOREIGN KEY ("GAME_ID") REFERENCES schedule("game_id")
);
"""