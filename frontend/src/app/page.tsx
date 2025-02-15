// src/App.tsx
import React from 'react';
import GameGrid from '../components/GameGrid';
import Navbar from '../components/Navbar'

const App: React.FC = () => {
  return (
    <div className="App mx-auto flex flex-col min-h-screen bg-dark-bg">
      <Navbar/>
      <div className="mx-auto max-w-[90%] pt-[100px]">
        <h1 className="text-2xl font-bold text-white p-4 mx-auto">Recent Games</h1>
        <GameGrid />
      </div>
    </div>
  );
};

export default App;
