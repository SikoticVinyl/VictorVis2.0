import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { AlertCircle, TrendingUp, Trophy, Users } from 'lucide-react';

const HaloAnalyticsDashboard = () => {
  // Sample data - would be replaced with real API data
  const matchData = [
    { date: '2024-01', kills: 120, deaths: 80, kd: 1.5 },
    { date: '2024-02', kills: 150, deaths: 90, kd: 1.67 },
    { date: '2024-03', kills: 180, deaths: 85, kd: 2.12 },
    { date: '2024-04', kills: 165, deaths: 95, kd: 1.74 }
  ];

  return (
    <div className="w-full max-w-6xl mx-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Halo Performance Analytics</h1>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-6 w-6 text-blue-500" />
              <div>
                <p className="text-sm text-gray-500">Average K/D Ratio</p>
                <p className="text-2xl font-bold">1.75</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Trophy className="h-6 w-6 text-yellow-500" />
              <div>
                <p className="text-sm text-gray-500">Win Rate</p>
                <p className="text-2xl font-bold">62%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <Users className="h-6 w-6 text-green-500" />
              <div>
                <p className="text-sm text-gray-500">Team Play Score</p>
                <p className="text-2xl font-bold">85</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-6 w-6 text-red-500" />
              <div>
                <p className="text-sm text-gray-500">Areas to Improve</p>
                <p className="text-2xl font-bold">3</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Trend */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] w-full">
            <LineChart data={matchData} width={800} height={300}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="kd" stroke="#8884d8" name="K/D Ratio" />
              <Line type="monotone" dataKey="kills" stroke="#82ca9d" name="Kills" />
              <Line type="monotone" dataKey="deaths" stroke="#ff7300" name="Deaths" />
            </LineChart>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Strengths</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-4">
              <li>Strong performance in Team Slayer modes</li>
              <li>Excellent power weapon control</li>
              <li>Above average assist rate</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Improvement Areas</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="list-disc pl-4">
              <li>Objective game modes need focus</li>
              <li>Vehicle usage could be optimized</li>
              <li>Map position awareness in critical moments</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default HaloAnalyticsDashboard;
