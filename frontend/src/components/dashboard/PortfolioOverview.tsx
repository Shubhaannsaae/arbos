import React from 'react';
import { Portfolio } from '../../types/portfolio';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

interface PortfolioOverviewProps {
  portfolio: Portfolio | null;
}

export const PortfolioOverview: React.FC<PortfolioOverviewProps> = ({ portfolio }) => {
  if (!portfolio) {
    return <div className="text-gray-500">No portfolio data available</div>;
  }

  const stats = [
    {
      name: 'Total Value',
      value: formatCurrency(portfolio.totalValue),
      change: formatPercentage(portfolio.dayChange),
      changeType: portfolio.dayChange >= 0 ? 'increase' : 'decrease'
    },
    {
      name: 'Today\'s P&L',
      value: formatCurrency(portfolio.dayPnL),
      change: formatPercentage(portfolio.dayChangePercent),
      changeType: portfolio.dayPnL >= 0 ? 'increase' : 'decrease'
    },
    {
      name: 'Active Positions',
      value: portfolio.activePositions.toString(),
      change: null,
      changeType: null
    },
    {
      name: 'Available Balance',
      value: formatCurrency(portfolio.availableBalance),
      change: null,
      changeType: null
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <div key={stat.name} className="bg-gray-50 p-4 rounded-lg">
          <dt className="text-sm font-medium text-gray-500">{stat.name}</dt>
          <dd className="mt-1 flex items-baseline">
            <div className="text-2xl font-semibold text-gray-900">{stat.value}</div>
            {stat.change && (
              <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                stat.changeType === 'increase' ? 'text-green-600' : 'text-red-600'
              }`}>
                {stat.changeType === 'increase' ? '+' : ''}{stat.change}
              </div>
            )}
          </dd>
        </div>
      ))}
    </div>
  );
};
