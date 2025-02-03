"use client";
import React, { useEffect, useState } from 'react';
import { useSearchParams } from "next/navigation";

export default function BoxScore() {

  const searchParams = useSearchParams();
  const game_ID = searchParams.get("game_ID");
  const [gameInfo, setGameInfo] = useState(null);
  const [homeTeamStats, setHomeTeamStats] = useState(null);
  const [awayTeamStats, setAwayTeamStats] = useState(null);


  // Fetch box score from API
  useEffect(() => {
    const fetchBoxScore = async () => {
      try {
        const response = await fetch(`http://localhost:5000/box-score?game_ID=${game_ID}`);
        const data = await response.json();
        // Set the info of the game who played
        setGameInfo(data[0][0]);

        // Figure out and set home and away team states
        const stats = data[1]
        const home = stats.find(
            (team) => team.TEAM_ABBREVIATION === data[0][0].HOME_TEAM_ABBREVIATION
        );
        const away = stats.find(
            (team) => team.TEAM_ABBREVIATION === data[0][0].AWAY_TEAM_ABBREVIATION
        )

        setHomeTeamStats(home)
        setAwayTeamStats(away)
        console.log(data[0][0])
        console.log(away)
        console.log(home)
      } catch (error) {
        console.error('Error fetching games:', error);
      }
    };

    fetchBoxScore();
  }, []);

return (
  <>
    {awayTeamStats && (
      <div>
        <h1 className="text-2xl font-bold p-4 mx-auto">NBA Games</h1>
        <div className="text-2xl font-bold p-4 mx-auto w-max">
          <h1>{homeTeamStats.TEAM_NAME} vs {awayTeamStats.TEAM_NAME}</h1>
        </div>
      </div>
    )}
  </>
);
};

