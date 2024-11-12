// src/components/PerformanceGraph.jsx
'use client';

import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PerformanceData {
  date: string;
  kdr: number;
  dps: number;
}

interface PerformanceGraphProps {
  data: PerformanceData[];
}

const PerformanceGraph: React.FC<PerformanceGraphProps> = ({ data }) => {
  return (
    <div className="bg-white rounded-lg shadow p-6 mb-6">
      <h2 className="text-xl font-bold mb-4">Performance Trends</h2>
      <div style={{ width: '100%', height: '400px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line 
              yAxisId="left" 
              type="monotone" 
              dataKey="kdr" 
              stroke="#8884d8" 
              name="KDR" 
            />
            <Line 
              yAxisId="right" 
              type="monotone" 
              dataKey="dps" 
              stroke="#82ca9d" 
              name="DPS" 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PerformanceGraph;