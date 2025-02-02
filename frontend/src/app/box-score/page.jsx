"use client";
import React, { useEffect, useState } from 'react';
import { useSearchParams } from "next/navigation";

export default function BoxScore() {

  const searchParams = useSearchParams();
  const gameID = searchParams.get("gameID");
  const [boxScore, setBoxScore] = useState(null);

  // Fetch box score from API
  useEffect(() => {
    const fetchBoxScore = async () => {
      try {
        const response = await fetch('http://localhost:5000/box-score?gameID=${gameId}');
        const data = await response.json();
        setBoxScore(data);
        console.log(data)
      } catch (error) {
        console.error('Error fetching games:', error);
      }
    };

    fetchBoxScore();
  }, []);

  return (
    <div className="App mx-auto">
      <h1 className="text-2xl font-bold p-4 mx-auto">NBA Games</h1>

    </div>
  );
}
