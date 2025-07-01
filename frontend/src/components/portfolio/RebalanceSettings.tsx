import React, { useState } from 'react';
import { usePortfolio } from '../../hooks/usePortfolio';
import { LoadingSpinner } from '../common/LoadingSpinner';

interface RebalanceSettings {
  threshold: number;
  frequency: 'manual' | 'daily' | 'weekly' | 'monthly';
  maxSlippage: number;
  enableAutoRebalance: boolean;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';
}

export const RebalanceSettings: React.FC = () => {
  const { rebalance, loading } = usePortfolio();
  const [settings, setSettings] = useState<RebalanceSettings>({
    threshold: 5.0,
    frequency: 'weekly',
    maxSlippage: 1.0,
    enableAutoRebalance: false,
    riskLevel: 'moderate'
  });

  const handleSettingChange = (key: keyof RebalanceSettings, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleManualRebalance = async () => {
    try {
      await rebalance();
    } catch (error) {
      console.error('Rebalance failed:', error);
    }
  };

  const handleSaveSettings = async () => {
    // In a real implementation, this would save to the smart contract
    console.log('Saving settings:', settings);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Rebalance Settings</h3>
        <button
          onClick={handleManualRebalance}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? (
            <div className="flex items-center">
              <LoadingSpinner size="sm" className="mr-2" />
              Rebalancing...
            </div>
          ) : (
            'Rebalance Now'
          )}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Automation Settings</h4>
          
          <div className="space-y-4">
            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={settings.enableAutoRebalance}
                  onChange={(e) => handleSettingChange('enableAutoRebalance', e.target.checked)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="ml-2 text-sm text-gray-700">Enable Auto-Rebalancing</span>
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Rebalance Frequency
              </label>
              <select
                value={settings.frequency}
                onChange={(e) => handleSettingChange('frequency', e.target.value)}
                disabled={!settings.enableAutoRebalance}
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm disabled:opacity-50"
              >
                <option value="manual">Manual Only</option>
                <option value="daily">Daily</option>
                <option value="weekly">Weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Rebalance Threshold ({settings.threshold}%)
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="0.5"
                value={settings.threshold}
                onChange={(e) => handleSettingChange('threshold', parseFloat(e.target.value))}
                disabled={!settings.enableAutoRebalance}
                className="w-full disabled:opacity-50"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1%</span>
                <span>20%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Risk Settings</h4>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Risk Level
              </label>
              <select
                value={settings.riskLevel}
                onChange={(e) => handleSettingChange('riskLevel', e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
              >
                <option value="conservative">Conservative</option>
                <option value="moderate">Moderate</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Slippage ({settings.maxSlippage}%)
              </label>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={settings.maxSlippage}
                onChange={(e) => handleSettingChange('maxSlippage', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0.1%</span>
                <span>5.0%</span>
              </div>
            </div>

            <div className="pt-4">
              <button
                onClick={handleSaveSettings}
                className="w-full bg-gray-900 text-white px-4 py-2 rounded-md text-sm hover:bg-gray-800"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h5 className="text-sm font-medium text-blue-900 mb-2">Rebalancing Information</h5>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Automatic rebalancing uses Chainlink Automation for reliable execution</li>
          <li>• Threshold determines when portfolio drift triggers rebalancing</li>
          <li>• Gas costs are optimized using Chainlink price feeds</li>
          <li>• All trades are executed through secure DEX aggregators</li>
        </ul>
      </div>
    </div>
  );
};
