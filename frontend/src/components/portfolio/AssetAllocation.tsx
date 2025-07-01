import React, { useState } from 'react';
import { Pie } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { usePortfolio } from '../../hooks/usePortfolio';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

ChartJS.register(ArcElement, Tooltip, Legend);

export const AssetAllocation: React.FC = () => {
  const { positions, updateAllocation, loading } = usePortfolio();
  const [editing, setEditing] = useState(false);
  const [newAllocations, setNewAllocations] = useState<{ [token: string]: number }>({});

  const chartData = {
    labels: positions.map(p => p.token),
    datasets: [
      {
        data: positions.map(p => p.percentage),
        backgroundColor: [
          '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
          '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#6B7280'
        ],
        borderWidth: 2,
        borderColor: '#FFFFFF'
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const label = context.label || '';
            const value = context.parsed || 0;
            return `${label}: ${value.toFixed(1)}%`;
          }
        }
      }
    }
  };

  const handleAllocationChange = (token: string, value: string) => {
    const numValue = parseFloat(value) || 0;
    setNewAllocations(prev => ({ ...prev, [token]: numValue }));
  };

  const handleSaveAllocations = async () => {
    try {
      await updateAllocation(newAllocations);
      setEditing(false);
      setNewAllocations({});
    } catch (error) {
      console.error('Failed to update allocations:', error);
    }
  };

  const totalNewAllocation = Object.values(newAllocations).reduce((sum, val) => sum + val, 0);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Asset Allocation</h3>
        <button
          onClick={() => setEditing(!editing)}
          className="text-blue-600 hover:text-blue-500 text-sm font-medium"
        >
          {editing ? 'Cancel' : 'Rebalance'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Current Allocation</h4>
          <div className="h-64">
            <Pie data={chartData} options={chartOptions} />
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">
            {editing ? 'Target Allocation' : 'Position Details'}
          </h4>
          
          <div className="space-y-3">
            {positions.map((position) => (
              <div key={position.token} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-3 h-3 rounded-full" style={{
                    backgroundColor: chartData.datasets[0].backgroundColor[positions.indexOf(position)]
                  }}></div>
                  <div>
                    <div className="font-medium text-gray-900">{position.token}</div>
                    <div className="text-sm text-gray-500">
                      {formatCurrency(position.value)}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  {editing ? (
                    <input
                      type="number"
                      step="0.1"
                      min="0"
                      max="100"
                      value={newAllocations[position.token] || position.percentage}
                      onChange={(e) => handleAllocationChange(position.token, e.target.value)}
                      className="w-16 px-2 py-1 text-sm border border-gray-300 rounded text-right"
                    />
                  ) : (
                    <div className="font-medium text-gray-900">
                      {formatPercentage(position.percentage)}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {editing && (
              <div className="mt-4 space-y-3">
                <div className="flex justify-between text-sm">
                  <span>Total Allocation:</span>
                  <span className={totalNewAllocation === 100 ? 'text-green-600' : 'text-red-600'}>
                    {totalNewAllocation.toFixed(1)}%
                  </span>
                </div>
                
                <div className="flex space-x-2">
                  <button
                    onClick={handleSaveAllocations}
                    disabled={totalNewAllocation !== 100 || loading}
                    className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Saving...' : 'Save Changes'}
                  </button>
                  <button
                    onClick={() => setEditing(false)}
                    className="flex-1 bg-gray-200 text-gray-800 py-2 px-4 rounded-md text-sm hover:bg-gray-300"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
