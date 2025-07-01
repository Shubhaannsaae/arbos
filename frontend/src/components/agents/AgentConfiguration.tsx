import React, { useState, useEffect } from 'react';
import { useAgents } from '../../hooks/useAgents';
import { Agent, AgentConfig } from '../../types/agents';
import { LoadingSpinner } from '../common/LoadingSpinner';

export const AgentConfiguration: React.FC = () => {
  const { agents, updateAgent, loading } = useAgents();
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [config, setConfig] = useState<Partial<AgentConfig>>({});
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    const agent = agents.find(a => a.id === selectedAgent);
    if (agent) {
      setConfig(agent.config);
      setHasChanges(false);
    }
  }, [selectedAgent, agents]);

  const handleConfigChange = (key: string, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleParameterChange = (key: string, value: any) => {
    setConfig(prev => ({
      ...prev,
      parameters: { ...prev.parameters, [key]: value }
    }));
    setHasChanges(true);
  };

  const handleSaveConfig = async () => {
    if (!selectedAgent || !hasChanges) return;
    
    try {
      await updateAgent(selectedAgent, config);
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to update agent configuration:', error);
    }
  };

  const selectedAgentData = agents.find(a => a.id === selectedAgent);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Agent Configuration</h3>
        <select
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          className="border border-gray-300 rounded-md px-3 py-2 text-sm"
        >
          <option value="">Select Agent</option>
          {agents.map((agent) => (
            <option key={agent.id} value={agent.id}>
              {agent.name}
            </option>
          ))}
        </select>
      </div>

      {!selectedAgent ? (
        <div className="text-center py-8 text-gray-500">
          Select an agent to configure its settings
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="text-md font-medium text-gray-900 mb-4">Basic Settings</h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Agent Name</label>
                <input
                  type="text"
                  value={config.name || ''}
                  onChange={(e) => handleConfigChange('name', e.target.value)}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Strategy</label>
                <select
                  value={config.strategy || ''}
                  onChange={(e) => handleConfigChange('strategy', e.target.value)}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                >
                  <option value="conservative">Conservative</option>
                  <option value="moderate">Moderate</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Risk Level</label>
                <select
                  value={config.riskLevel || ''}
                  onChange={(e) => handleConfigChange('riskLevel', e.target.value)}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Max Amount (ETH)</label>
                <input
                  type="number"
                  step="0.01"
                  value={config.maxAmount || ''}
                  onChange={(e) => handleConfigChange('maxAmount', e.target.value)}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                />
              </div>

              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.enabled || false}
                    onChange={(e) => handleConfigChange('enabled', e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Enable Agent</span>
                </label>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="text-md font-medium text-gray-900 mb-4">Advanced Parameters</h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Min Profit Threshold ({config.parameters?.minProfitThreshold || 0.5}%)
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={config.parameters?.minProfitThreshold || 0.5}
                  onChange={(e) => handleParameterChange('minProfitThreshold', parseFloat(e.target.value))}
                  className="mt-1 block w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.1%</span>
                  <span>5.0%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Max Slippage ({config.parameters?.maxSlippage || 1.0}%)
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={config.parameters?.maxSlippage || 1.0}
                  onChange={(e) => handleParameterChange('maxSlippage', parseFloat(e.target.value))}
                  className="mt-1 block w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>0.1%</span>
                  <span>5.0%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Gas Limit</label>
                <input
                  type="number"
                  value={config.parameters?.gasLimit || 500000}
                  onChange={(e) => handleParameterChange('gasLimit', parseInt(e.target.value))}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Stop Loss ({config.parameters?.stopLoss || 5.0}%)
                </label>
                <input
                  type="range"
                  min="1.0"
                  max="20.0"
                  step="0.5"
                  value={config.parameters?.stopLoss || 5.0}
                  onChange={(e) => handleParameterChange('stopLoss', parseFloat(e.target.value))}
                  className="mt-1 block w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1.0%</span>
                  <span>20.0%</span>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Take Profit ({config.parameters?.takeProfit || 10.0}%)
                </label>
                <input
                  type="range"
                  min="2.0"
                  max="50.0"
                  step="1.0"
                  value={config.parameters?.takeProfit || 10.0}
                  onChange={(e) => handleParameterChange('takeProfit', parseFloat(e.target.value))}
                  className="mt-1 block w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>2.0%</span>
                  <span>50.0%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="text-md font-medium text-gray-900 mb-4">Chainlink Integration</h4>
            
            <div className="space-y-4">
              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.parameters?.useChainlinkDataFeeds || true}
                    onChange={(e) => handleParameterChange('useChainlinkDataFeeds', e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Use Chainlink Data Feeds</span>
                </label>
              </div>

              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.parameters?.useChainlinkVRF || false}
                    onChange={(e) => handleParameterChange('useChainlinkVRF', e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Use Chainlink VRF for randomization</span>
                </label>
              </div>

              <div>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={config.parameters?.useChainlinkAutomation || true}
                    onChange={(e) => handleParameterChange('useChainlinkAutomation', e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Use Chainlink Automation</span>
                </label>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Data Feed Update Frequency</label>
                <select
                  value={config.parameters?.dataFeedFrequency || '1m'}
                  onChange={(e) => handleParameterChange('dataFeedFrequency', e.target.value)}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                >
                  <option value="30s">30 seconds</option>
                  <option value="1m">1 minute</option>
                  <option value="5m">5 minutes</option>
                  <option value="15m">15 minutes</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h4 className="text-md font-medium text-gray-900 mb-4">Status & Actions</h4>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Current Status</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  selectedAgentData?.status === 'active' ? 'bg-green-100 text-green-800' :
                  selectedAgentData?.status === 'paused' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {selectedAgentData?.status}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-700">Last Update</span>
                <span className="text-sm text-gray-500">
                  {selectedAgentData?.lastExecution ? 
                    new Date(selectedAgentData.lastExecution.timestamp).toLocaleString() : 
                    'Never'
                  }
                </span>
              </div>

              <div className="pt-4 space-y-2">
                <button
                  onClick={handleSaveConfig}
                  disabled={!hasChanges || loading}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? <LoadingSpinner size="sm" /> : 'Save Configuration'}
                </button>
                
                {hasChanges && (
                  <div className="text-xs text-yellow-600 text-center">
                    You have unsaved changes
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
