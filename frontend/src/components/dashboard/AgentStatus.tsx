import React from 'react';
import { Agent } from '../../types/agents';
import { formatPercentage } from '../../utils/formatters';

interface AgentStatusProps {
  agents: Agent[];
}

export const AgentStatus: React.FC<AgentStatusProps> = ({ agents }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'error':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-4">
      {agents.map((agent) => (
        <div key={agent.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
              {agent.status}
            </div>
            <div>
              <h4 className="text-sm font-medium text-gray-900">{agent.name}</h4>
              <p className="text-sm text-gray-500">{agent.type}</p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-sm font-medium text-gray-900">
              {formatPercentage(agent.performance.totalReturn)}
            </div>
            <div className="text-xs text-gray-500">
              {agent.performance.tradesExecuted} trades
            </div>
          </div>
        </div>
      ))}
      
      {agents.length === 0 && (
        <div className="text-center text-gray-500 py-8">
          No agents configured
        </div>
      )}
    </div>
  );
};
