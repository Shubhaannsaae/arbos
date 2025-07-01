import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { formatPercentage } from '../../utils/formatters';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface RiskMetric {
  name: string;
  value: number;
  benchmark: number;
  status: 'good' | 'warning' | 'danger';
}

const RISK_METRICS: RiskMetric[] = [
  { name: 'Sharpe Ratio', value: 1.85, benchmark: 1.0, status: 'good' },
  { name: 'Max Drawdown', value: -8.5, benchmark: -15.0, status: 'good' },
  { name: 'Volatility', value: 12.3, benchmark: 20.0, status: 'good' },
  { name: 'Beta', value: 0.85, benchmark: 1.0, status: 'good' },
  { name: 'VaR (95%)', value: -2.1, benchmark: -5.0, status: 'good' },
  { name: 'Correlation', value: 0.65, benchmark: 0.8, status: 'warning' }
];

export const RiskMetrics: React.FC = () => {
  // Mock volatility data
  const volatilityData = {
    labels: Array.from({ length: 30 }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (29 - i));
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }),
    datasets: [
      {
        label: 'Portfolio Volatility',
        data: Array.from({ length: 30 }, () => Math.random() * 5 + 10),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.1,
      },
      {
        label: 'Market Volatility',
        data: Array.from({ length: 30 }, () => Math.random() * 8 + 15),
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.1,
      }
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Volatility (%)'
        }
      }
    },
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-600 bg-green-100';
      case 'warning': return 'text-yellow-600 bg-yellow-100';
      case 'danger': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good': return '✓';
      case 'warning': return '⚠';
      case 'danger': return '✗';
      default: return '?';
    }
  };

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-medium text-gray-900">Risk Metrics</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {RISK_METRICS.map((metric) => (
          <div key={metric.name} className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">{metric.name}</h4>
              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(metric.status)}`}>
                {getStatusIcon(metric.status)}
              </span>
            </div>
            
            <div className="flex items-baseline space-x-2">
              <div className="text-2xl font-bold text-gray-900">
                {metric.value > 0 && !metric.name.includes('Drawdown') && !metric.name.includes('VaR') ? '+' : ''}
                {metric.value.toFixed(2)}
                {metric.name.includes('Ratio') || metric.name.includes('Beta') || metric.name.includes('Correlation') ? '' : '%'}
              </div>
              <div className="text-sm text-gray-500">
                vs {metric.benchmark > 0 && !metric.name.includes('Drawdown') && !metric.name.includes('VaR') ? '+' : ''}
                {metric.benchmark.toFixed(1)}
                {metric.name.includes('Ratio') || metric.name.includes('Beta') || metric.name.includes('Correlation') ? '' : '%'}
              </div>
            </div>
            
            <div className="mt-2">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${metric.status === 'good' ? 'bg-green-500' : metric.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style={{ width: `${Math.min(100, Math.abs(metric.value / metric.benchmark) * 100)}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-6">
        <h4 className="text-md font-medium text-gray-900 mb-4">Volatility Trend (30 Days)</h4>
        <div className="h-64">
          <Line data={volatilityData} options={chartOptions} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Risk Assessment</h4>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Overall Risk Score</span>
              <span className="text-sm font-medium text-green-600">Low (3.2/10)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Diversification</span>
              <span className="text-sm font-medium text-green-600">Good</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Liquidity Risk</span>
              <span className="text-sm font-medium text-yellow-600">Medium</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm text-gray-600">Concentration Risk</span>
              <span className="text-sm font-medium text-green-600">Low</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Recommendations</h4>
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="text-green-600 mr-2">•</span>
              Portfolio shows strong risk-adjusted returns
            </li>
            <li className="flex items-start">
              <span className="text-yellow-600 mr-2">•</span>
              Consider reducing correlation with market
            </li>
            <li className="flex items-start">
              <span className="text-green-600 mr-2">•</span>
              Volatility remains within target range
            </li>
            <li className="flex items-start">
              <span className="text-blue-600 mr-2">•</span>
              Maintain current diversification strategy
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
