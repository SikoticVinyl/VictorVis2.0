// src/components/match-history-table.tsx
import React from 'react';

// Define the type for a single match
interface Match {
  date: string;
  kdr: number;
  dps: number;
  wins: number;
  losses: number;
}

// Define the props interface
interface MatchHistoryTableProps {
  matches: Match[];
}

const MatchHistoryTable = ({ matches }: MatchHistoryTableProps) => {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-4">Match History</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">KDR</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">DPS</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Wins</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Losses</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {matches.map((match: Match, index: number) => (
              <tr key={index}>
                <td className="px-6 py-4 whitespace-nowrap">{match.date}</td>
                <td className="px-6 py-4 whitespace-nowrap">{match.kdr}</td>
                <td className="px-6 py-4 whitespace-nowrap">{match.dps}</td>
                <td className="px-6 py-4 whitespace-nowrap">{match.wins}</td>
                <td className="px-6 py-4 whitespace-nowrap">{match.losses}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MatchHistoryTable;