import React from 'react';
import { PortfolioOverview } from './PortfolioOverview';
import { ProfitLossChart } from './ProfitLossChart';
import { AgentStatus } from './AgentStatus';
import { usePortfolio } from '../../hooks/usePortfolio';
import { useAgents } from '../../hooks/useAgents';
import { LoadingSpinner } from '../common/LoadingSpinner';

export const Dashboard: React.FC = () => {
  const { portfolio, loading: portfolioLoading } = usePortfolio();
  const { agents, loading: agentsLoading } = useAgents();

  if (portfolioLoading || agentsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Portfolio Overview
          </h3>
          <PortfolioOverview portfolio={portfolio} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Profit & Loss
            </h3>
            <ProfitLossChart data={portfolio?.history || []} />
          </div>
        </div>

        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              AI Agent Status
            </h3>
            <AgentStatus agents={agents} />
          </div>
        </div>
      </div>
    </div>
  );
};
