// src/App.tsx
import React from 'react';
import GameGrid from '../../components/GameGrid';

export default function SearchGames() {
  return (
    <div className="App mx-auto">
      <h1 className="text-2xl font-bold p-4 mx-auto">NBA Games</h1>
      <GameGrid />
    </div>
  );
}
