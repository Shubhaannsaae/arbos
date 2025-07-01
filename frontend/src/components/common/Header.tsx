import React from 'react';
import { ConnectWallet } from './ConnectWallet';
import { useWeb3 } from '../../hooks/useWeb3';
import { formatAddress } from '../../utils/helpers';

export const Header: React.FC = () => {
  const { account, chainId, isConnected } = useWeb3();

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <h1 className="text-2xl font-bold text-gray-900">ArbOS</h1>
            </div>
            <nav className="hidden md:ml-6 md:flex md:space-x-8">
              <a href="/dashboard" className="text-gray-900 hover:text-blue-600 px-3 py-2 text-sm font-medium">
                Dashboard
              </a>
              <a href="/arbitrage" className="text-gray-500 hover:text-blue-600 px-3 py-2 text-sm font-medium">
                Arbitrage
              </a>
              <a href="/portfolio" className="text-gray-500 hover:text-blue-600 px-3 py-2 text-sm font-medium">
                Portfolio
              </a>
              <a href="/agents" className="text-gray-500 hover:text-blue-600 px-3 py-2 text-sm font-medium">
                Agents
              </a>
              <a href="/security" className="text-gray-500 hover:text-blue-600 px-3 py-2 text-sm font-medium">
                Security
              </a>
            </nav>
          </div>
          
          <div className="flex items-center space-x-4">
            {isConnected && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">
                  Chain: {chainId}
                </span>
                <span className="text-sm font-medium text-gray-900">
                  {formatAddress(account)}
                </span>
              </div>
            )}
            <ConnectWallet />
          </div>
        </div>
      </div>
    </header>
  );
};
