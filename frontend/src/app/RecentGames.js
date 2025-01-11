import React, { useState, useEffect } from 'react';

const RecentGames = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch the most recent games from the backend
  useEffect(() => {
    fetch('http://localhost:5000/api/recent_games')  // Flask API endpoint
      .then(response => response.json())
      .then(data => {
        setGames(data);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data: ", error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <div>
      <h2>Most Recent NBA Games</h2>
      <ul>
        {games.map(game => (
          <li key={game.game_id}>
            <strong>{game.home_team}</strong> vs <strong>{game.away_team}</strong><br />
            Date: {game.game_date}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default RecentGames;
