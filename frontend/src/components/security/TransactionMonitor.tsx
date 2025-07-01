import React, { useState, useEffect } from 'react';
import { useWeb3 } from '../../hooks/useWeb3';
import { formatCurrency } from '../../utils/formatters';

interface Transaction {
  hash: string;
  type: 'arbitrage' | 'portfolio' | 'yield' | 'admin';
  status: 'pending' | 'confirmed' | 'failed';
  amount: number;
  gasUsed: number;
  gasPrice: number;
  timestamp: number;
  riskScore: 'low' | 'medium' | 'high';
  details: string;
}

const MOCK_TRANSACTIONS: Transaction[] = [
  {
    hash: '0x1234...5678',
    type: 'arbitrage',
    status: 'confirmed',
    amount: 2.5,
    gasUsed: 150000,
    gasPrice: 20,
    timestamp: Date.now() - 300000,
    riskScore: 'low',
    details: 'ETH/USDC arbitrage on Uniswap vs Sushiswap'
  },
  {
    hash: '0x2345...6789',
    type: 'portfolio',
    status: 'pending',
    amount: 5.0,
    gasUsed: 200000,
    gasPrice: 25,
    timestamp: Date.now() - 120000,
    riskScore: 'medium',
    details: 'Portfolio rebalancing - increasing LINK allocation'
  },
  {
    hash: '0x3456...7890',
    type: 'yield',
    status: 'confirmed',
    amount: 1.2,
    gasUsed: 180000,
    gasPrice: 22,
    timestamp: Date.now() - 600000,
    riskScore: 'low',
    details: 'Yield farming rewards claim from Compound'
  }
];

export const TransactionMonitor: React.FC = () => {
  const { chainId } = useWeb3();
  const [transactions, setTransactions] = useState<Transaction[]>(MOCK_TRANSACTIONS);
  const [filter, setFilter] = useState<'all' | 'pending' | 'confirmed' | 'failed'>('all');
  const [riskFilter, setRiskFilter] = useState<'all' | 'low' | 'medium' | 'high'>('all');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'confirmed':
        return 'text-green-600 bg-green-100';
      case 'pending':
        return 'text-yellow-600 bg-yellow-100';
      case 'failed':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low':
        return 'text-green-600 bg-green-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'high':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'arbitrage':
        return '⚡';
      case 'portfolio':
        return '📊';
      case 'yield':
        return '🌾';
      case 'admin':
        return '⚙️';
      default:
        return '📄';
    }
  };

  const filteredTransactions = transactions.filter(tx => {
    if (filter !== 'all' && tx.status !== filter) return false;
    if (riskFilter !== 'all' && tx.riskScore !== riskFilter) return false;
    return true;
  });

  const getExplorerUrl = (hash: string) => {
    const explorers: { [key: number]: string } = {
      1: 'https://etherscan.io',
      43114: 'https://snowtrace.io',
      137: 'https://polygonscan.com',
      42161: 'https://arbiscan.io'
    };
    
    const baseUrl = explorers[chainId || 1] || 'https://etherscan.io';
    return `${baseUrl}/tx/${hash}`;
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Transaction Monitor</h3>
        <div className="flex space-x-4">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="all">All Status</option>
            <option value="pending">Pending</option>
            <option value="confirmed">Confirmed</option>
            <option value="failed">Failed</option>
          </select>
          
          <select
            value={riskFilter}
            onChange={(e) => setRiskFilter(e.target.value as any)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="all">All Risk Levels</option>
            <option value="low">Low Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="high">High Risk</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Total Transactions (24h)</div>
          <div className="text-2xl font-bold text-gray-900">{transactions.length}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Success Rate</div>
          <div className="text-2xl font-bold text-green-600">
            {((transactions.filter(t => t.status === 'confirmed').length / transactions.length) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Total Gas Used</div>
          <div className="text-2xl font-bold text-blue-600">
            {formatCurrency(transactions.reduce((sum, tx) => sum + (tx.gasUsed * tx.gasPrice / 1e9), 0))} ETH
          </div>
        </div>
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        <ul className="divide-y divide-gray-200">
          {filteredTransactions.map((transaction) => (
            <li key={transaction.hash}>
              <div className="px-4 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="text-2xl">{getTypeIcon(transaction.type)}</div>
                    <div>
                      <div className="flex items-center space-x-2">
                        <p className="text-sm font-medium text-gray-900">
                          {transaction.hash}
                        </p>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(transaction.status)}`}>
                          {transaction.status}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(transaction.riskScore)}`}>
                          {transaction.riskScore} risk
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-1">
                        {transaction.details}
                      </p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                        <span>Type: {transaction.type}</span>
                        <span>Amount: {formatCurrency(transaction.amount)} ETH</span>
                        <span>Gas: {transaction.gasUsed.toLocaleString()}</span>
                        <span>{new Date(transaction.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <a
                      href={getExplorerUrl(transaction.hash)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-500 text-sm"
                    >
                      View on Explorer
                    </a>
                  </div>
                </div>
              </div>
            </li>
          ))}
          
          {filteredTransactions.length === 0 && (
            <li className="px-4 py-8 text-center text-gray-500">
              No transactions match the current filters
            </li>
          )}
        </ul>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <h5 className="text-sm font-medium text-yellow-900 mb-2">Transaction Security Features</h5>
        <ul className="text-sm text-yellow-800 space-y-1">
          <li>• All transactions validated against Chainlink price feeds</li>
          <li>• Automatic risk scoring based on transaction patterns</li>
          <li>• Real-time monitoring for unusual activity</li>
          <li>• Gas optimization to prevent front-running attacks</li>
          <li>• Multi-signature approval for high-value transactions</li>
        </ul>
      </div>
    </div>
  );
};
