# This is a file to collect data using an NBA API for a given season
# Uses tools from this repo https://github.com/swar/nba_api
import pandas
from nba_api.stats.endpoints import playercareerstats
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import commonplayerinfo
import player_data as pData
import team_schedule_collector as schedule
import json


# Nikola Jokić
career = playercareerstats.PlayerCareerStats(player_id='203999')
# pandas data frames (optional: pip install pandas)
career.get_data_frames()[0].to_csv("data/careerStats/Nikola Jokic.csv")
# TODO: Figure out how to translate from player_id to
print(pData.find_player_by_id("203999"))