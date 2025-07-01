import React, { useState } from 'react';
import { useWeb3 } from '../../hooks/useWeb3';
import { LoadingSpinner } from './LoadingSpinner';

export const ConnectWallet: React.FC = () => {
  const { connect, disconnect, isConnected, isConnecting, account } = useWeb3();
  const [showDisconnect, setShowDisconnect] = useState(false);

  const handleConnect = async () => {
    try {
      await connect();
    } catch (error) {
      console.error('Failed to connect wallet:', error);
    }
  };

  const handleDisconnect = () => {
    disconnect();
    setShowDisconnect(false);
  };

  if (isConnecting) {
    return (
      <button disabled className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-md">
        <LoadingSpinner size="sm" />
        <span>Connecting...</span>
      </button>
    );
  }

  if (isConnected) {
    return (
      <div className="relative">
        <button
          onClick={() => setShowDisconnect(!showDisconnect)}
          className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors"
        >
          Connected
        </button>
        
        {showDisconnect && (
          <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-200 z-10">
            <div className="py-1">
              <div className="px-4 py-2 text-sm text-gray-700 border-b">
                {account}
              </div>
              <button
                onClick={handleDisconnect}
                className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50"
              >
                Disconnect
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <button
      onClick={handleConnect}
      className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
    >
      Connect Wallet
    </button>
  );
};
