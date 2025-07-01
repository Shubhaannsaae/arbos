import React, { useState } from 'react';
import { formatCurrency, formatPercentage } from '../../utils/formatters';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface YieldFarm {
  id: string;
  protocol: string;
  pair: string;
  apy: number;
  tvl: number;
  rewards: string[];
  userStaked: number;
  userRewards: number;
  riskLevel: 'low' | 'medium' | 'high';
}

const YIELD_FARMS: YieldFarm[] = [
  {
    id: '1',
    protocol: 'Uniswap V3',
    pair: 'ETH/USDC',
    apy: 12.5,
    tvl: 450000000,
    rewards: ['FEES', 'UNI'],
    userStaked: 5000,
    userRewards: 125.50,
    riskLevel: 'low'
  },
  {
    id: '2',
    protocol: 'Curve',
    pair: '3CRV',
    apy: 8.2,
    tvl: 1200000000,
    rewards: ['CRV', 'FEES'],
    userStaked: 10000,
    userRewards: 220.75,
    riskLevel: 'low'
  },
  {
    id: '3',
    protocol: 'Compound',
    pair: 'USDC',
    apy: 4.1,
    tvl: 800000000,
    rewards: ['COMP'],
    userStaked: 15000,
    userRewards: 85.25,
    riskLevel: 'low'
  }
];

export const YieldFarming: React.FC = () => {
  const [selectedFarm, setSelectedFarm] = useState<string | null>(null);
  const [stakeAmount, setStakeAmount] = useState('');
  const [loading, setLoading] = useState(false);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const handleStake = async (farmId: string) => {
    if (!stakeAmount) return;
    
    setLoading(true);
    try {
      // Implement staking logic here
      await new Promise(resolve => setTimeout(resolve, 2000)); // Mock delay
      setStakeAmount('');
      setSelectedFarm(null);
    } catch (error) {
      console.error('Staking failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClaim = async (farmId: string) => {
    setLoading(true);
    try {
      // Implement claim logic here
      await new Promise(resolve => setTimeout(resolve, 1500)); // Mock delay
    } catch (error) {
      console.error('Claim failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Yield Farming</h3>
        <div className="text-sm text-gray-500">
          Total Staked: {formatCurrency(YIELD_FARMS.reduce((sum, farm) => sum + farm.userStaked, 0))}
        </div>
      </div>

      <div className="grid gap-4">
        {YIELD_FARMS.map((farm) => (
          <div key={farm.id} className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-center">
              <div>
                <h4 className="font-medium text-gray-900">{farm.protocol}</h4>
                <p className="text-sm text-gray-500">{farm.pair}</p>
                <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium mt-1 ${getRiskColor(farm.riskLevel)}`}>
                  {farm.riskLevel} risk
                </span>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-semibold text-green-600">
                  {formatPercentage(farm.apy)}
                </div>
                <div className="text-sm text-gray-500">APY</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-medium text-gray-900">
                  {formatCurrency(farm.tvl, 0)}
                </div>
                <div className="text-sm text-gray-500">TVL</div>
              </div>
              
              <div className="text-center">
                <div className="text-lg font-medium text-blue-600">
                  {formatCurrency(farm.userStaked)}
                </div>
                <div className="text-sm text-gray-500">Your Stake</div>
              </div>
              
              <div className="flex flex-col space-y-2">
                {farm.userRewards > 0 && (
                  <div className="text-center">
                    <div className="text-sm font-medium text-green-600">
                      {formatCurrency(farm.userRewards)} rewards
                    </div>
                    <button
                      onClick={() => handleClaim(farm.id)}
                      disabled={loading}
                      className="w-full bg-green-600 text-white px-3 py-1 rounded text-xs hover:bg-green-700 disabled:opacity-50"
                    >
                      Claim
                    </button>
                  </div>
                )}
                
                <button
                  onClick={() => setSelectedFarm(selectedFarm === farm.id ? null : farm.id)}
                  className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700"
                >
                  {farm.userStaked > 0 ? 'Add More' : 'Stake'}
                </button>
              </div>
            </div>
            
            {selectedFarm === farm.id && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex space-x-4">
                  <input
                    type="number"
                    placeholder="Amount to stake"
                    value={stakeAmount}
                    onChange={(e) => setStakeAmount(e.target.value)}
                    className="flex-1 border border-gray-300 rounded-md px-3 py-2 text-sm"
                  />
                  <button
                    onClick={() => handleStake(farm.id)}
                    disabled={!stakeAmount || loading}
                    className="bg-blue-600 text-white px-6 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
                  >
                    {loading ? <LoadingSpinner size="sm" /> : 'Stake'}
                  </button>
                </div>
              </div>
            )}
            
            <div className="mt-4 grid grid-cols-2 gap-4 text-sm text-gray-600">
              <div>
                <span className="font-medium">Rewards:</span> {farm.rewards.join(', ')}
              </div>
              <div>
                <span className="font-medium">Daily Yield:</span> {formatCurrency(farm.userStaked * farm.apy / 365 / 100)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
