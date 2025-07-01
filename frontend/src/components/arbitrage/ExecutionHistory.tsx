import React from 'react';
import { useArbitrage } from '../../hooks/useArbitrage';
import { formatCurrency, formatPercentage } from '../../utils/formatters';
import { getStatusColor } from '../../utils/helpers';

export const ExecutionHistory: React.FC = () => {
  const { executions } = useArbitrage();

  const getStatusBadge = (status: string) => {
    const colors = getStatusColor(status);
    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors}`}>
        {status}
      </span>
    );
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-medium text-gray-900">Execution History</h3>
      
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <ul className="divide-y divide-gray-200">
          {executions.map((execution) => (
            <li key={execution.id}>
              <div className="px-4 py-4 flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {execution.tokenPair}
                      </p>
                      <p className="text-sm text-gray-500">
                        {execution.sourceExchange} → {execution.targetExchange}
                      </p>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">
                          {formatCurrency(execution.amount)}
                        </p>
                        <p className={`text-sm ${execution.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {execution.profit >= 0 ? '+' : ''}{formatCurrency(execution.profit)}
                        </p>
                      </div>
                      
                      <div className="text-right">
                        <p className="text-sm text-gray-500">
                          {formatPercentage(execution.profitPercent)}
                        </p>
                        <p className="text-sm text-gray-500">
                          {new Date(execution.timestamp).toLocaleDateString()}
                        </p>
                      </div>
                      
                      {getStatusBadge(execution.status)}
                    </div>
                  </div>
                  
                  {execution.txHash && (
                    <div className="mt-2">
                      <a
                        href={`https://etherscan.io/tx/${execution.txHash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-600 hover:text-blue-500"
                      >
                        View Transaction
                      </a>
                    </div>
                  )}
                </div>
              </div>
            </li>
          ))}
          
          {executions.length === 0 && (
            <li className="px-4 py-8 text-center text-gray-500">
              No executions yet
            </li>
          )}
        </ul>
      </div>
    </div>
  );
};
