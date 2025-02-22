import React from 'react';

const PlayerStatsTable = ({ teamName = '', playerStats = [] }) => {
  const sortPlayersByMinutes = (players = []) => {
    if (!Array.isArray(players)) return [];

    return players
      .slice()
      .sort((a, b) => {
        const getSeconds = (timeStr) => {
          if (!timeStr) return 0;
          const [minutes, seconds] = timeStr.split(":").map(Number);
          return minutes * 60 + seconds;
        };

        const aSeconds = getSeconds(a.MIN);
        const bSeconds = getSeconds(b.MIN);
        return bSeconds - aSeconds;
      });
  };

  const sortedPlayers = sortPlayersByMinutes(playerStats);

  if (!sortedPlayers.length) {
    return (
      <div className="text-center p-4">
        <h2 className="text-2xl font-bold">{teamName || 'Team'} - No player data available</h2>
      </div>
    );
  }

  return (
    <div>
      <div className="text-2xl font-bold p-4 mx-auto w-max mt-8">
        <h2>{teamName} Player Stats</h2>
      </div>
      <div className="px-4 py-3 max-w-[90%] mx-auto">
        <div className="overflow-x-auto rounded-xl border border-border-color bg-dark-bg">
          <table className="w-full">
            <thead>
              <tr className="bg-grid-top">
                <th className="px-4 py-3 text-left text-white text-sm font-medium">Player</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">MIN</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FGM</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FGA</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FG_PCT</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FG3M</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FG3A</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FG3_PCT</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FTM</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FTA</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">FT_PCT</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">PTS</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">OREB</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">DREB</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">REB</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">AST</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">STL</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">BLK</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">TO</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">PF</th>
                <th className="px-4 py-3 text-left text-white text-sm font-medium">+/-</th>
              </tr>
            </thead>
            <tbody>
              {sortedPlayers.map((player, index) => (
                <tr key={index} className="border-t border-t-border-color">
                  <td className="px-4 py-2 text-white text-sm">{player.PLAYER_NAME}</td>
                  <td className="px-4 py-2 text-sm">{player.MIN}</td>
                  <td className="px-4 py-2 text-sm">{player.FGM}</td>
                  <td className="px-4 py-2 text-sm">{player.FGA}</td>
                  <td className="px-4 py-2 text-sm">{(player.FG_PCT * 100).toFixed(1)}</td>
                  <td className="px-4 py-2 text-sm">{player.FG3M}</td>
                  <td className="px-4 py-2 text-sm">{player.FG3A}</td>
                  <td className="px-4 py-2 text-sm">{(player.FG3_PCT * 100).toFixed(1)}</td>
                  <td className="px-4 py-2 text-sm">{player.FTM}</td>
                  <td className="px-4 py-2 text-sm">{player.FTA}</td>
                  <td className="px-4 py-2 text-sm">{(player.FT_PCT * 100).toFixed(1)}</td>
                  <td className="px-4 py-2 text-sm">{player.PTS}</td>
                  <td className="px-4 py-2 text-sm">{player.OREB}</td>
                  <td className="px-4 py-2 text-sm">{player.DREB}</td>
                  <td className="px-4 py-2 text-sm">{player.REB}</td>
                  <td className="px-4 py-2 text-sm">{player.AST}</td>
                  <td className="px-4 py-2 text-sm">{player.STL}</td>
                  <td className="px-4 py-2 text-sm">{player.BLK}</td>
                  <td className="px-4 py-2 text-sm">{player.TOV}</td>
                  <td className="px-4 py-2 text-sm">{player.PF}</td>
                  <td className="px-4 py-2 text-sm">{player.PLUS_MINUS}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PlayerStatsTable;