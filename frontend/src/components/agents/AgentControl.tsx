import React, { useState } from 'react';
import { useAgents } from '../../hooks/useAgents';
import { Agent, AgentConfig } from '../../types/agents';
import { LoadingSpinner } from '../common/LoadingSpinner';
import { formatCurrency, formatPercentage } from '../../utils/formatters';

export const AgentControl: React.FC = () => {
  const { agents, loading, startAgent, stopAgent, createAgent } = useAgents();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newAgentConfig, setNewAgentConfig] = useState<Partial<AgentConfig>>({
    name: '',
    type: 'arbitrage',
    strategy: 'conservative',
    maxAmount: '1000',
    riskLevel: 'medium'
  });

  const handleStartAgent = async (id: string) => {
    try {
      await startAgent(id);
    } catch (error) {
      console.error('Failed to start agent:', error);
    }
  };

  const handleStopAgent = async (id: string) => {
    try {
      await stopAgent(id);
    } catch (error) {
      console.error('Failed to stop agent:', error);
    }
  };

  const handleCreateAgent = async () => {
    if (!newAgentConfig.name || !newAgentConfig.maxAmount) return;
    
    try {
      await createAgent({
        name: newAgentConfig.name,
        type: newAgentConfig.type as 'arbitrage' | 'portfolio' | 'yield',
        strategy: newAgentConfig.strategy as 'conservative' | 'moderate' | 'aggressive',
        maxAmount: newAgentConfig.maxAmount,
        riskLevel: newAgentConfig.riskLevel as 'low' | 'medium' | 'high',
        enabled: true,
        parameters: {
          minProfitThreshold: 0.5,
          maxSlippage: 1.0,
          gasLimit: 500000
        }
      });
      
      setShowCreateForm(false);
      setNewAgentConfig({
        name: '',
        type: 'arbitrage',
        strategy: 'conservative',
        maxAmount: '1000',
        riskLevel: 'medium'
      });
    } catch (error) {
      console.error('Failed to create agent:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800';
      case 'paused': return 'bg-yellow-100 text-yellow-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">AI Agent Control</h3>
        <button
          onClick={() => setShowCreateForm(true)}
          className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm hover:bg-blue-700"
        >
          Create Agent
        </button>
      </div>

      {showCreateForm && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h4 className="text-lg font-medium text-gray-900 mb-4">Create New Agent</h4>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Name</label>
                <input
                  type="text"
                  value={newAgentConfig.name}
                  onChange={(e) => setNewAgentConfig({...newAgentConfig, name: e.target.value})}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Type</label>
                <select
                  value={newAgentConfig.type}
                  onChange={(e) => setNewAgentConfig({...newAgentConfig, type: e.target.value})}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                >
                  <option value="arbitrage">Arbitrage</option>
                  <option value="portfolio">Portfolio Management</option>
                  <option value="yield">Yield Farming</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Strategy</label>
                <select
                  value={newAgentConfig.strategy}
                  onChange={(e) => setNewAgentConfig({...newAgentConfig, strategy: e.target.value})}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                >
                  <option value="conservative">Conservative</option>
                  <option value="moderate">Moderate</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Max Amount (ETH)</label>
                <input
                  type="number"
                  value={newAgentConfig.maxAmount}
                  onChange={(e) => setNewAgentConfig({...newAgentConfig, maxAmount: e.target.value})}
                  className="mt-1 block w-full border border-gray-300 rounded-md px-3 py-2"
                />
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={handleCreateAgent}
                  disabled={loading}
                  className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? <LoadingSpinner size="sm" /> : 'Create'}
                </button>
                <button
                  onClick={() => setShowCreateForm(false)}
                  className="flex-1 bg-gray-300 text-gray-700 py-2 px-4 rounded-md hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-6">
        {agents.map((agent) => (
          <div key={agent.id} className="bg-white border border-gray-200 rounded-lg p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div>
                  <h4 className="text-lg font-medium text-gray-900">{agent.name}</h4>
                  <p className="text-sm text-gray-500 capitalize">{agent.type} • {agent.strategy}</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(agent.status)}`}>
                  {agent.status}
                </span>
              </div>
              
              <div className="flex space-x-2">
                {agent.status === 'active' ? (
                  <button
                    onClick={() => handleStopAgent(agent.id)}
                    disabled={loading}
                    className="bg-red-600 text-white px-4 py-2 rounded-md text-sm hover:bg-red-700 disabled:opacity-50"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    onClick={() => handleStartAgent(agent.id)}
                    disabled={loading}
                    className="bg-green-600 text-white px-4 py-2 rounded-md text-sm hover:bg-green-700 disabled:opacity-50"
                  >
                    Start
                  </button>
                )}
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-gray-500">Total Return</div>
                <div className={`text-lg font-semibold ${agent.performance.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {formatPercentage(agent.performance.totalReturn)}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Trades</div>
                <div className="text-lg font-semibold text-gray-900">{agent.performance.tradesExecuted}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Success Rate</div>
                <div className="text-lg font-semibold text-blue-600">
                  {formatPercentage(agent.performance.successRate)}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Max Amount</div>
                <div className="text-lg font-semibold text-gray-900">
                  {formatCurrency(parseFloat(agent.config.maxAmount))}
                </div>
              </div>
            </div>

            {agent.lastExecution && (
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="text-sm text-gray-500">
                  Last execution: {new Date(agent.lastExecution.timestamp).toLocaleString()}
                </div>
                <div className="text-sm text-gray-700">
                  Action: {agent.lastExecution.action} • 
                  Result: <span className={agent.lastExecution.success ? 'text-green-600' : 'text-red-600'}>
                    {agent.lastExecution.success ? 'Success' : 'Failed'}
                  </span>
                </div>
              </div>
            )}
          </div>
        ))}
        
        {agents.length === 0 && !loading && (
          <div className="text-center py-8 text-gray-500">
            No agents created yet. Create your first AI agent to get started.
          </div>
        )}
      </div>
    </div>
  );
};
