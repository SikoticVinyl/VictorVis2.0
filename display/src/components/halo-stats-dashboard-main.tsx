// src/components/HaloStatsDashboard.jsx
'use client';

import React from 'react';
import StatCard from './stat-card';
import PerformanceGraph from './performance-graph';
import MatchHistoryTable from './match-history-table';

const HaloStatsDashboard = () => {
  // Mock data structure based on API response format
  const playerStats = {
    matchHistory: [
      { date: '2024-01', kdr: 2.1, dps: 150, wins: 8, losses: 2 },
      { date: '2024-02', kdr: 1.9, dps: 145, wins: 7, losses: 3 },
      { date: '2024-03', kdr: 2.3, dps: 160, wins: 9, losses: 1 },
    ],
    teamStats: {
      totalMatches: 30,
      winRate: 0.75,
      avgKDR: 2.1,
      avgDPS: 152,
    }
  };

  return (
    <div className="p-4 space-y-4">
      {/* Stats Overview */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-bold mb-4">Team Performance Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard 
            value={playerStats.teamStats.totalMatches}
            label="Total Matches"
            bgColor="bg-blue-50"
          />
          <StatCard 
            value={`${(playerStats.teamStats.winRate * 100).toFixed(1)}%`}
            label="Win Rate"
            bgColor="bg-green-50"
          />
          <StatCard 
            value={playerStats.teamStats.avgKDR.toFixed(2)}
            label="Average KDR"
            bgColor="bg-yellow-50"
          />
          <StatCard 
            value={playerStats.teamStats.avgDPS}
            label="Average DPS"
            bgColor="bg-purple-50"
          />
        </div>
      </div>

      {/* Performance Graph */}
      <PerformanceGraph data={playerStats.matchHistory} />

      {/* Match History Table */}
      <MatchHistoryTable matches={playerStats.matchHistory} />
    </div>
  );
};

export default HaloStatsDashboard;