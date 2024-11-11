// src/components/StatCard.jsx
import React from 'react';

const StatCard = ({ value, label, bgColor = 'bg-blue-50' }) => {
  return (
    <div className={`${bgColor} p-4 rounded-lg`}>
      <div className="text-lg font-bold">{value}</div>
      <div className="text-sm text-gray-600">{label}</div>
    </div>
  );
};

export default StatCard;
