'use client';
import React, { useEffect, useState } from 'react';
import { Game } from '../types/types.ts';
import Link from "next/link";

const GameGrid: React.FC = () => {
  const [games, setGames] = useState<Game[]>([]);

  // Fetch recent games from API
  useEffect(() => {
    const fetchGames = async () => {
      try {
        const response = await fetch('http://localhost:5000/recent-games?limit=21');
        const data = await response.json();
        setGames(data);
      } catch (error) {
        console.error('Error fetching games:', error);
      }
    };

    fetchGames();
  }, []);

  return (
    <div className="game-grid text-white">
      {games.length === 0 ? (
        <p>Loading games...</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {games.map((game) => (
              <Link
                href={{
                    pathname: '/box-score',
                    query: { game_ID: game.GAME_ID },
                }}
                key={game.GAME_ID}
              >
            <div className="border p-4 rounded shadow-md">
              <h2 className="text-xl font-bold">
                {game.HOME_TEAM_NAME} vs. {game.AWAY_TEAM_NAME}
              </h2>
              <p>{game.GAME_DATE}</p>
            </div>
            </Link>

          ))}
        </div>
      )}
    </div>
  );
};

export default GameGrid;
