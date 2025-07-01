import Head from 'next/head';
import { AgentControl } from '../components/agents/AgentControl';
import { AgentMetrics } from '../components/agents/AgentMetrics';
import { AgentConfiguration } from '../components/agents/AgentConfiguration';

export default function AgentsPage() {
  return (
    <>
      <Head>
        <title>AI Agents - ArbOS</title>
        <meta name="description" content="AI trading agents with performance metrics and configuration" />
      </Head>
      
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">AI Trading Agents</h1>
            <p className="text-gray-600 mt-2">
              Automated trading strategies powered by Chainlink oracles
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Chainlink automation active</span>
          </div>
        </div>
        
        <AgentControl />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <AgentMetrics />
          </div>
          <div>
            <AgentConfiguration />
          </div>
        </div>
      </div>
    </>
  );
}
