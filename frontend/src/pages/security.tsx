import Head from 'next/head';
import { SecurityDashboard } from '../components/security/SecurityDashboard';
import { TransactionMonitor } from '../components/security/TransactionMonitor';
import { RiskAlerts } from '../components/security/RiskAlerts';

export default function SecurityPage() {
  return (
    <>
      <Head>
        <title>Security - ArbOS</title>
        <meta name="description" content="Security dashboard with transaction monitoring and risk alerts" />
      </Head>
      
      <div className="space-y-8">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Security Center</h1>
            <p className="text-gray-600 mt-2">
              Real-time security monitoring and risk management
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-600">All systems secure</span>
            </div>
          </div>
        </div>
        
        <SecurityDashboard />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <TransactionMonitor />
          </div>
          <div>
            <RiskAlerts />
          </div>
        </div>
      </div>
    </>
  );
}
