import Head from 'next/head';
import { Dashboard } from '../components/dashboard/Dashboard';

export default function DashboardPage() {
  return (
    <>
      <Head>
        <title>Dashboard - ArbOS</title>
        <meta name="description" content="ArbOS trading dashboard with portfolio overview and agent status" />
      </Head>
      
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <div className="text-sm text-gray-500">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
        
        <Dashboard />
      </div>
    </>
  );
}
