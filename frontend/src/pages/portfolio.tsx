import Head from 'next/head';
import { AssetAllocation } from '../components/portfolio/AssetAllocation';
import { RebalanceSettings } from '../components/portfolio/RebalanceSettings';
import { YieldFarming } from '../components/portfolio/YieldFarming';
import { RiskMetrics } from '../components/portfolio/RiskMetrics';

export default function PortfolioPage() {
  return (
    <>
      <Head>
        <title>Portfolio - ArbOS</title>
        <meta name="description" content="Portfolio management with asset allocation and yield farming" />
      </Head>
      
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">Portfolio Management</h1>
          <div className="flex space-x-4">
            <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors">
              Export Report
            </button>
          </div>
        </div>
        
        <AssetAllocation />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-8">
            <RebalanceSettings />
            <YieldFarming />
          </div>
          <div>
            <RiskMetrics />
          </div>
        </div>
      </div>
    </>
  );
}
