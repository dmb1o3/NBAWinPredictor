# Table Definitions
schedule_table = """
CREATE TABLE schedule (
    "season_id" char(5),
    "game_id" char(10) PRIMARY KEY,
    "game_date" date,
    "home_team_id" char(10),
    "away_team_id" char(10),
    "video_available" int,
    "neutral_site" boolean
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
    "pie"  float,

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
    "pie"  float,

    PRIMARY KEY ("game_id", "player_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

officials_table = """
CREATE TABLE officials (
    "game_id" char(10),
    "referee_id" INT,
    "name" varchar(50),
    "jersey_num" INT,
    
    PRIMARY KEY ("game_id", "referee_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

attendance_table = """
CREATE TABLE attendance (
    "game_id" char(10),
    "arena_id" INT,
    "arena_city" VARCHAR(50),
    "arena_state" CHAR(2),
    "arena_timezone" VARCHAR(50),
    "attendance" int,
    "sellout" int,

    PRIMARY KEY ("game_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""

misc_team_stats_table = """
CREATE TABLE misc_team_stats (
    "game_id" char(10),
    "team_id" char(10),
    "points_in_the_paint" int,
    "points_second_chance" int,
    "points_fast_break" int,
    "biggest_lead" int,
    "lead_changes" int,
    "times_tied" int,
    "biggest_scoring_run" int,
    "points_from_turnovers" int,
    "period1_score" int,
    "period2_score" int,
    "period3_score" int,
    "period4_score" int,
        
    PRIMARY KEY ("game_id", "team_id"),
    FOREIGN KEY ("game_id") REFERENCES schedule("game_id")
);
"""