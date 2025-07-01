import Head from 'next/head';
import { ArbitrageOpportunities } from '../components/arbitrage/ArbitrageOpportunities';
import { ExecutionHistory } from '../components/arbitrage/ExecutionHistory';
import { PriceMonitor } from '../components/arbitrage/PriceMonitor';

export default function ArbitragePage() {
  return (
    <>
      <Head>
        <title>Arbitrage - ArbOS</title>
        <meta name="description" content="Cross-chain arbitrage opportunities and execution history" />
      </Head>
      
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">Arbitrage Trading</h1>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Live monitoring active</span>
          </div>
        </div>
        
        <ArbitrageOpportunities />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <ExecutionHistory />
          </div>
          <div>
            <PriceMonitor />
          </div>
        </div>
      </div>
    </>
  );
}
