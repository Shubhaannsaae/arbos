import React, { useState } from 'react';
import { useChainlink } from '../../hooks/useChainlink';
import { formatCurrency } from '../../utils/formatters';
import { LoadingSpinner } from '../common/LoadingSpinner';

const MONITORED_TOKENS = ['ETH', 'BTC', 'LINK', 'USDC', 'USDT'];

export const PriceMonitor: React.FC = () => {
  const { priceFeeds, requestPrice, loading } = useChainlink();
  const [selectedTokens, setSelectedTokens] = useState<Set<string>>(new Set(MONITORED_TOKENS));

  const handleTokenToggle = (token: string) => {
    const newSelected = new Set(selectedTokens);
    if (newSelected.has(token)) {
      newSelected.delete(token);
    } else {
      newSelected.add(token);
      requestPrice(token); // Request price when adding token
    }
    setSelectedTokens(newSelected);
  };

  const getPriceChange = (token: string) => {
    // This would come from historical data in a real implementation
    return Math.random() * 10 - 5; // Mock change between -5% and +5%
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Price Monitor</h3>
        <div className="text-sm text-gray-500">
          Real-time Chainlink price feeds
        </div>
      </div>

      <div className="grid gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Select Tokens to Monitor</h4>
          <div className="flex flex-wrap gap-2">
            {MONITORED_TOKENS.map((token) => (
              <button
                key={token}
                onClick={() => handleTokenToggle(token)}
                className={`px-3 py-1 rounded-full text-sm font-medium ${
                  selectedTokens.has(token)
                    ? 'bg-blue-100 text-blue-800 border border-blue-200'
                    : 'bg-gray-100 text-gray-700 border border-gray-200'
                }`}
              >
                {token}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from(selectedTokens).map((token) => {
            const price = priceFeeds[token];
            const change = getPriceChange(token);
            const isPositive = change >= 0;
            
            return (
              <div key={token} className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900">{token}</h4>
                    <p className="text-sm text-gray-500">USD Price</p>
                  </div>
                  
                  <div className="text-right">
                    {price ? (
                      <>
                        <div className="text-xl font-bold text-gray-900">
                          {formatCurrency(price)}
                        </div>
                        <div className={`text-sm font-medium ${
                          isPositive ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {isPositive ? '+' : ''}{change.toFixed(2)}%
                        </div>
                      </>
                    ) : loading ? (
                      <LoadingSpinner size="sm" />
                    ) : (
                      <button
                        onClick={() => requestPrice(token)}
                        className="text-blue-600 hover:text-blue-500 text-sm"
                      >
                        Load Price
                      </button>
                    )}
                  </div>
                </div>
                
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Last Updated</span>
                    <span>{new Date().toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
