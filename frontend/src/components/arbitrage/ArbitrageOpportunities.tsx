import React, { useState } from 'react';
import { useArbitrage } from '../../hooks/useArbitrage';
import { formatCurrency, formatPercentage } from '../../utils/formatters';
import { LoadingSpinner } from '../common/LoadingSpinner';

export const ArbitrageOpportunities: React.FC = () => {
  const { opportunities, executeArbitrage, loading, error } = useArbitrage();
  const [executingId, setExecutingId] = useState<string | null>(null);
  const [amount, setAmount] = useState<{ [id: string]: string }>({});

  const handleExecute = async (opportunityId: string) => {
    const executeAmount = amount[opportunityId];
    if (!executeAmount) return;

    setExecutingId(opportunityId);
    try {
      await executeArbitrage(opportunityId, executeAmount);
      setAmount(prev => ({ ...prev, [opportunityId]: '' }));
    } catch (error) {
      console.error('Execution failed:', error);
    } finally {
      setExecutingId(null);
    }
  };

  const handleAmountChange = (id: string, value: string) => {
    setAmount(prev => ({ ...prev, [id]: value }));
  };

  if (loading && opportunities.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Live Opportunities</h3>
        <div className="flex items-center text-sm text-gray-500">
          {loading && <LoadingSpinner size="sm" className="mr-2" />}
          Auto-refreshing every 30s
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      <div className="grid gap-4">
        {opportunities.map((opportunity) => (
          <div key={opportunity.id} className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
              <div>
                <h4 className="font-medium text-gray-900">{opportunity.tokenPair}</h4>
                <p className="text-sm text-gray-500">
                  {opportunity.sourceExchange} → {opportunity.targetExchange}
                </p>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600">
                  {formatPercentage(opportunity.profitPercent)}
                </div>
                <div className="text-sm text-gray-500">Profit Margin</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-medium text-gray-900">
                  {formatCurrency(opportunity.maxProfit)}
                </div>
                <div className="text-sm text-gray-500">
                  Max: {formatCurrency(opportunity.maxAmount)}
                </div>
              </div>
              
              <div className="flex flex-col space-y-2">
                <input
                  type="number"
                  placeholder="Amount"
                  value={amount[opportunity.id] || ''}
                  onChange={(e) => handleAmountChange(opportunity.id, e.target.value)}
                  className="border border-gray-300 rounded-md px-3 py-2 text-sm"
                  max={opportunity.maxAmount}
                />
                <button
                  onClick={() => handleExecute(opportunity.id)}
                  disabled={!amount[opportunity.id] || executingId === opportunity.id}
                  className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {executingId === opportunity.id ? (
                    <div className="flex items-center">
                      <LoadingSpinner size="sm" className="mr-2" />
                      Executing...
                    </div>
                  ) : (
                    'Execute'
                  )}
                </button>
              </div>
            </div>
            
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm text-gray-600">
              <div>
                <span className="font-medium">Source Price:</span> {formatCurrency(opportunity.sourcePrice)}
              </div>
              <div>
                <span className="font-medium">Target Price:</span> {formatCurrency(opportunity.targetPrice)}
              </div>
              <div>
                <span className="font-medium">Gas Cost:</span> {formatCurrency(opportunity.gasCost)}
              </div>
              <div>
                <span className="font-medium">Updated:</span> {new Date(opportunity.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        
        {opportunities.length === 0 && !loading && (
          <div className="text-center py-8 text-gray-500">
            No arbitrage opportunities found
          </div>
        )}
      </div>
    </div>
  );
};
