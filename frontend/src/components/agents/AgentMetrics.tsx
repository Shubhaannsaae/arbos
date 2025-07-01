import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { useAgents } from '../../hooks/useAgents';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export const AgentMetrics: React.FC = () => {
  const { agents, getAgentMetrics } = useAgents();
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [metrics, setMetrics] = useState<any>(null);
  const [timeframe, setTimeframe] = useState<'24h' | '7d' | '30d' | '90d'>('7d');

  useEffect(() => {
    if (selectedAgent) {
      getAgentMetrics(selectedAgent)
        .then(setMetrics)
        .catch(console.error);
    }
  }, [selectedAgent, getAgentMetrics]);

  const performanceData = {
    labels: metrics?.performance?.map((p: any) => new Date(p.timestamp).toLocaleDateString()) || [],
    datasets: [
      {
        label: 'Cumulative Return (%)',
        data: metrics?.performance?.map((p: any) => p.cumulativeReturn) || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.1,
      },
      {
        label: 'Daily Return (%)',
        data: metrics?.performance?.map((p: any) => p.dailyReturn) || [],
        borderColor: 'rgb(16, 185, 129)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        tension: 0.1,
      }
    ],
  };

  const tradesData = {
    labels: ['Successful', 'Failed', 'Pending'],
    datasets: [
      {
        label: 'Trades',
        data: [
          metrics?.trades?.successful || 0,
          metrics?.trades?.failed || 0,
          metrics?.trades?.pending || 0
        ],
        backgroundColor: [
          'rgba(16, 185, 129, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(251, 191, 36, 0.8)'
        ],
        borderColor: [
          'rgb(16, 185, 129)',
          'rgb(239, 68, 68)',
          'rgb(251, 191, 36)'
        ],
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Agent Performance Metrics</h3>
        <div className="flex space-x-4">
          <select
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="">Select Agent</option>
            {agents.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.name}
              </option>
            ))}
          </select>
          
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value as any)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="24h">24 Hours</option>
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
            <option value="90d">90 Days</option>
          </select>
        </div>
      </div>

      {!selectedAgent ? (
        <div className="text-center py-8 text-gray-500">
          Select an agent to view detailed metrics
        </div>
      ) : !metrics ? (
        <div className="text-center py-8 text-gray-500">
          Loading metrics...
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-sm text-gray-500">Total Return</div>
              <div className={`text-2xl font-bold ${metrics.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(metrics.totalReturn)}
              </div>
              <div className="text-sm text-gray-500">
                {formatCurrency(metrics.totalReturnUSD)} USD
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-sm text-gray-500">Sharpe Ratio</div>
              <div className="text-2xl font-bold text-blue-600">
                {metrics.sharpeRatio?.toFixed(2) || '0.00'}
              </div>
              <div className="text-sm text-gray-500">Risk-adjusted return</div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-sm text-gray-500">Max Drawdown</div>
              <div className="text-2xl font-bold text-red-600">
                {formatPercentage(metrics.maxDrawdown)}
              </div>
              <div className="text-sm text-gray-500">Worst loss period</div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <div className="text-sm text-gray-500">Win Rate</div>
              <div className="text-2xl font-bold text-green-600">
                {formatPercentage(metrics.winRate)}
              </div>
              <div className="text-sm text-gray-500">
                {metrics.trades?.successful || 0} / {(metrics.trades?.successful || 0) + (metrics.trades?.failed || 0)} trades
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="text-md font-medium text-gray-900 mb-4">Performance Over Time</h4>
              <div className="h-64">
                <Line data={performanceData} options={chartOptions} />
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="text-md font-medium text-gray-900 mb-4">Trade Distribution</h4>
              <div className="h-64">
                <Bar data={tradesData} options={barOptions} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="text-md font-medium text-gray-900 mb-4">Recent Trades</h4>
              <div className="space-y-3">
                {metrics.recentTrades?.slice(0, 5).map((trade: any, index: number) => (
                  <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                    <div>
                      <div className="font-medium text-gray-900">{trade.pair}</div>
                      <div className="text-sm text-gray-500">
                        {new Date(trade.timestamp).toLocaleString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`font-medium ${trade.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {trade.profit >= 0 ? '+' : ''}{formatCurrency(trade.profit)}
                      </div>
                      <div className="text-sm text-gray-500">
                        {formatPercentage(trade.profitPercent)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h4 className="text-md font-medium text-gray-900 mb-4">Strategy Analysis</h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Trade Size</span>
                  <span className="font-medium">{formatCurrency(metrics.avgTradeSize)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Hold Time</span>
                  <span className="font-medium">{metrics.avgHoldTime} minutes</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Best Trade</span>
                  <span className="font-medium text-green-600">
                    {formatCurrency(metrics.bestTrade)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Worst Trade</span>
                  <span className="font-medium text-red-600">
                    {formatCurrency(metrics.worstTrade)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Gas Used</span>
                  <span className="font-medium">{formatCurrency(metrics.totalGasUsed)} ETH</span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
