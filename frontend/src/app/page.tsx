// src/App.tsx
import React from 'react';
import GameGrid from '../components/GameGrid';

const App: React.FC = () => {
  return (
    <div className="App">
      <h1 className="text-2xl font-bold p-4">NBA Games</h1>
      <GameGrid />
    </div>
  );
};

export default App;
