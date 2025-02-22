"use client";
import React, { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Navbar from "../../components/Navbar";
import PlayerStatsTable from "../../components/PlayerStatsTable"

export default function BoxScore() {
  const searchParams = useSearchParams();
  const game_ID = searchParams.get("game_ID");
  const [gameInfo, setGameInfo] = useState(null);
  const [homeTeamStats, setHomeTeamStats] = useState(null);
  const [awayTeamStats, setAwayTeamStats] = useState(null);
  const [homePlayerStats, setHomePlayerStats] = useState(null);
  const [awayPlayerStats, setAwayPlayerStats] = useState(null);

  // Fetch box score from API
  useEffect(() => {
    const fetchBoxScore = async () => {
      try {
        const response = await fetch(
          `http://localhost:5000/box-score?game_ID=${game_ID}`
        );
        const data = await response.json();
        console.log(data);
        // Set the info of the game who played
        setGameInfo(data[0][0]);

        // Figure out and set home and away team stats
        const team_stats = data[1];
        const homeTeam = team_stats.find(
          (team) => team.TEAM_ABBREVIATION === data[0][0].HOME_TEAM_ABBREVIATION
        );
        const awayTeam = team_stats.find(
          (team) => team.TEAM_ABBREVIATION === data[0][0].AWAY_TEAM_ABBREVIATION
        );

        setHomeTeamStats(homeTeam);
        setAwayTeamStats(awayTeam);

        // Figure out and set home and away player stats
        const player_stats = data[2];
        const homePlayer = player_stats.filter(
          (player) =>
            player.TEAM_ABBREVIATION === data[0][0].HOME_TEAM_ABBREVIATION
        );
        const awayPlayer = player_stats.filter(
          (player) =>
            player.TEAM_ABBREVIATION === data[0][0].AWAY_TEAM_ABBREVIATION
        );

        setHomePlayerStats(homePlayer);
        setAwayPlayerStats(awayPlayer);
      } catch (error) {
        console.error("Error fetching games:", error);
      }
    };

    fetchBoxScore();
  }, []);

  return (
    <>
      {awayTeamStats && (
        <div className="bg-dark-bg text-white">
          <Navbar />
          <div className="text-3xl font-bold p-4 mx-auto w-max pt-[100px]">
            <h1>
              {homeTeamStats.TEAM_NAME} vs {awayTeamStats.TEAM_NAME}
            </h1>
          </div>
          <div className="px-4 py-3 max-w-[90%] mx-auto">
            <div className="flex overflow-hidden rounded-xl border border-border-color bg-dark-bg">
              <table className="flex-1">
                <thead>
                  <tr className="bg-grid-top">
                    <th className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 px-4 py-3 text-left text-white w-[400px] text-sm font-medium leading-normal"></th>
                    <th className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 px-4 py-3 text-left text-white w-[400px] text-sm font-medium leading-normal">
                      {homeTeamStats.TEAM_ABBREVIATION}
                    </th>
                    <th className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 px-4 py-3 text-left text-white w-[400px] text-sm font-medium leading-normal">
                      {awayTeamStats.TEAM_ABBREVIATION}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Points
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.PTS}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.PTS}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Field Goals Made-Attempted
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px] text-sm font-normal leading-normal">
                      {homeTeamStats.FGM}-{homeTeamStats.FGA}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.FGM}-{awayTeamStats.FGA}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Field Goal Percentage
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {(homeTeamStats.FG_PCT * 100).toFixed(1)}%
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {(awayTeamStats.FG_PCT * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      3-Point Field Goals Made-Attempted
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.FG3M}-{homeTeamStats.FG3A}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.FG3M}-{awayTeamStats.FG3A}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      3-Point Field Goal Percentage
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px] text-sm font-normal leading-normal">
                      {(homeTeamStats.FG3_PCT * 100).toFixed(1)}%
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {(awayTeamStats.FG3_PCT * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Free Throws Made-Attempted
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.FTM}-{homeTeamStats.FTA}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.FTM}-{awayTeamStats.FTA}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Free Throw Percentage
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {(homeTeamStats.FT_PCT * 100).toFixed(1)}%
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {(awayTeamStats.FT_PCT * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Total Rebounds
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.REB}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.REB}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Offensive Rebounds
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.OREB}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.OREB}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Defensive Rebounds
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.DREB}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.DREB}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Assists
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.AST}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.AST}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Steals
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.STL}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.STL}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Blocks
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.BLK}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.BLK}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Turnovers
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.TOV}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.TOV}
                    </td>
                  </tr>
                  <tr className="border-t border-t-border-color">
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-120 h-[72px] px-4 py-2 w-[400px] text-white text-sm font-normal leading-normal">
                      Personal Fouls
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-240 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {homeTeamStats.PF}
                    </td>
                    <td className="table-bef4c38b-f732-421a-b902-642bbea91d55-column-360 h-[72px] px-4 py-2 w-[400px]  text-sm font-normal leading-normal">
                      {awayTeamStats.PF}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <PlayerStatsTable teamName={homeTeamStats.TEAM_NAME} playerStats={homePlayerStats} />
          <PlayerStatsTable teamName={awayTeamStats.TEAM_NAME} playerStats={awayPlayerStats} />

        </div>
      )}
    </>
  );
}
