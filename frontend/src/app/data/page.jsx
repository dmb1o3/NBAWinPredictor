"use client"
import React, { useState } from 'react';
import Navbar from '../../components/Navbar'

const NBADataManager = () => {
  const [startYear, setStartYear] = useState('');
  const [endYear, setEndYear] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleDownload = async () => {
    if (!startYear || !endYear) {
      setMessage('Please enter both start and end years');
      return;
    }
    setIsLoading(true);
    setMessage('Downloading data...');
    try {
      const response = await fetch('http://localhost:5000/download-nba-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ startYear, endYear }),
      });
      const data = await response.json();
      setMessage(data.message || 'Download completed!');
    } catch (error) {
      setMessage('Error downloading data: ' + error.message);
    }
    setIsLoading(false);
  };

  const handleCheckMissing = async () => {
    if (!startYear || !endYear) {
      setMessage('Please enter both start and end years');
      return;
    }
    setIsLoading(true);
    setMessage('Checking for missing data...');
    try {
      const response = await fetch('http://localhost:5000/check-missing-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ startYear, endYear }),
      });
      const data = await response.json();
      setMessage(data.message || 'Check completed!');
    } catch (error) {
      setMessage('Error checking data: ' + error.message);
    }
    setIsLoading(false);
  };

  const handleSetupDatabase = async () => {
    setIsLoading(true);
    setMessage('Setting up database...');
    try {
      const response = await fetch('http://localhost:5000/setup-database', {
        method: 'POST',
      });
      const data = await response.json();
      setMessage(data.message || 'Database setup completed!');
    } catch (error) {
      setMessage('Error setting up database: ' + error.message);
    }
    setIsLoading(false);
  };

  return (
      <div className="bg-dark-bg text-white min-h-screen">
          <Navbar/>
    <div className="max-w-2xl mx-auto p-6 space-y-6 pt-[125px]">
      <h1 className="text-3xl font-bold text-center mb-8">NBA Data Manager</h1>

      <div className="flex gap-4 mb-6">
        <div className="flex-1">
          <label className="block text-sm font-medium mb-2" htmlFor="startYear">
            Start Year
          </label>
          <input
            id="startYear"
            type="number"
            min="1946"
            max={new Date().getFullYear()}
            value={startYear}
            onChange={(e) => setStartYear(e.target.value)}
            className="w-full p-2 border rounded bg-white/10 border-gray-600"
            placeholder="e.g., 2020"
          />
        </div>
        <div className="flex-1">
          <label className="block text-sm font-medium mb-2" htmlFor="endYear">
            End Year
          </label>
          <input
            id="endYear"
            type="number"
            min="1946"
            max={new Date().getFullYear()}
            value={endYear}
            onChange={(e) => setEndYear(e.target.value)}
            className="w-full p-2 border rounded bg-white/10 border-gray-600"
            placeholder="e.g., 2023"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={handleDownload}
          disabled={isLoading}
          className="p-3 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          Download Data
        </button>

        <button
          onClick={handleCheckMissing}
          disabled={isLoading}
          className="p-3 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
        >
          Check Missing Data
        </button>

        <button
          onClick={handleSetupDatabase}
          disabled={isLoading}
          className="p-3 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
        >
          Setup Database
        </button>
      </div>

      {message && (
        <div className="mt-6 p-4 rounded bg-gray-800 text-white">
          {message}
        </div>
      )}

      {isLoading && (
        <div className="text-center mt-6">
          Processing...
        </div>
      )}
    </div>
   </div>
  );
};

export default NBADataManager;