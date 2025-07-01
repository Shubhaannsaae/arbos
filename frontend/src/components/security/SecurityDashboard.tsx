import React, { useState, useEffect } from 'react';
import { ShieldCheckIcon, ExclamationTriangleIcon, CheckCircleIcon } from '@heroicons/react/24/outline';

interface SecurityMetric {
  id: string;
  name: string;
  status: 'healthy' | 'warning' | 'critical';
  value: string;
  description: string;
  lastCheck: number;
}

interface SecurityAlert {
  id: string;
  type: 'info' | 'warning' | 'error';
  title: string;
  message: string;
  timestamp: number;
  resolved: boolean;
}

const SECURITY_METRICS: SecurityMetric[] = [
  {
    id: '1',
    name: 'Smart Contract Security',
    status: 'healthy',
    value: '98%',
    description: 'All contracts audited and secure',
    lastCheck: Date.now() - 300000 // 5 minutes ago
  },
  {
    id: '2',
    name: 'Oracle Reliability',
    status: 'healthy',
    value: '99.9%',
    description: 'Chainlink oracles functioning normally',
    lastCheck: Date.now() - 60000 // 1 minute ago
  },
  {
    id: '3',
    name: 'Transaction Monitoring',
    status: 'warning',
    value: '2 alerts',
    description: 'Unusual transaction patterns detected',
    lastCheck: Date.now() - 120000 // 2 minutes ago
  },
  {
    id: '4',
    name: 'Access Control',
    status: 'healthy',
    value: 'Secure',
    description: 'All admin functions properly protected',
    lastCheck: Date.now() - 180000 // 3 minutes ago
  }
];

const RECENT_ALERTS: SecurityAlert[] = [
  {
    id: '1',
    type: 'warning',
    title: 'High Gas Price Detected',
    message: 'Gas prices above 100 gwei detected. Consider delaying non-urgent transactions.',
    timestamp: Date.now() - 1800000, // 30 minutes ago
    resolved: false
  },
  {
    id: '2',
    type: 'info',
    title: 'Oracle Update',
    message: 'Chainlink price feeds updated successfully for all monitored assets.',
    timestamp: Date.now() - 3600000, // 1 hour ago
    resolved: true
  },
  {
    id: '3',
    type: 'error',
    title: 'Failed Transaction',
    message: 'Arbitrage transaction failed due to insufficient liquidity.',
    timestamp: Date.now() - 7200000, // 2 hours ago
    resolved: true
  }
];

export const SecurityDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SecurityMetric[]>(SECURITY_METRICS);
  const [alerts, setAlerts] = useState<SecurityAlert[]>(RECENT_ALERTS);
  const [overallStatus, setOverallStatus] = useState<'healthy' | 'warning' | 'critical'>('healthy');

  useEffect(() => {
    // Calculate overall status based on individual metrics
    const hasError = metrics.some(m => m.status === 'critical');
    const hasWarning = metrics.some(m => m.status === 'warning');
    
    if (hasError) {
      setOverallStatus('critical');
    } else if (hasWarning) {
      setOverallStatus('warning');
    } else {
      setOverallStatus('healthy');
    }
  }, [metrics]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="w-5 h-5 text-green-600" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600" />;
      case 'critical':
        return <ExclamationTriangleIcon className="w-5 h-5 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'warning':
        return 'text-yellow-600 bg-yellow-100';
      case 'critical':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <ExclamationTriangleIcon className="w-4 h-4 text-red-600" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-4 h-4 text-yellow-600" />;
      case 'info':
        return <CheckCircleIcon className="w-4 h-4 text-blue-600" />;
      default:
        return null;
    }
  };

  const resolveAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, resolved: true } : alert
    ));
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Security Dashboard</h3>
        <div className="flex items-center space-x-2">
          <ShieldCheckIcon className="w-5 h-5 text-gray-400" />
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(overallStatus)}`}>
            {overallStatus === 'healthy' ? 'All Systems Secure' : 
             overallStatus === 'warning' ? 'Monitoring Issues' : 'Critical Issues'}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric) => (
          <div key={metric.id} className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">{metric.name}</h4>
              {getStatusIcon(metric.status)}
            </div>
            
            <div className="text-2xl font-bold text-gray-900 mb-1">
              {metric.value}
            </div>
            
            <p className="text-xs text-gray-500 mb-2">
              {metric.description}
            </p>
            
            <div className="text-xs text-gray-400">
              Last checked: {new Date(metric.lastCheck).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Security Checklist</h4>
          
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700">Multi-signature wallet protection</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700">Chainlink oracle price validation</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700">Smart contract audits completed</span>
            </div>
            <div className="flex items-center space-x-3">
              <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600" />
              <span className="text-sm text-gray-700">Emergency pause mechanism active</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700">Rate limiting implemented</span>
            </div>
            <div className="flex items-center space-x-3">
              <CheckCircleIcon className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700">Access control verified</span>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Recent Security Alerts</h4>
          
          <div className="space-y-3">
            {alerts.slice(0, 5).map((alert) => (
              <div key={alert.id} className={`p-3 rounded-lg border ${
                alert.resolved ? 'bg-gray-50 border-gray-200' : 
                alert.type === 'error' ? 'bg-red-50 border-red-200' :
                alert.type === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                'bg-blue-50 border-blue-200'
              }`}>
                <div className="flex items-start space-x-2">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h5 className="text-sm font-medium text-gray-900">{alert.title}</h5>
                      {!alert.resolved && (
                        <button
                          onClick={() => resolveAlert(alert.id)}
                          className="text-xs text-blue-600 hover:text-blue-500"
                        >
                          Resolve
                        </button>
                      )}
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{alert.message}</p>
                    <div className="text-xs text-gray-500 mt-1">
                      {new Date(alert.timestamp).toLocaleString()}
                      {alert.resolved && <span className="ml-2 text-green-600">• Resolved</span>}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h5 className="text-sm font-medium text-blue-900 mb-2">Security Best Practices</h5>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• All transactions are validated using Chainlink price feeds</li>
          <li>• Smart contracts undergo regular security audits</li>
          <li>• Multi-signature wallets protect admin functions</li>
          <li>• Emergency pause mechanisms can halt operations if needed</li>
          <li>• Real-time monitoring detects unusual activity patterns</li>
        </ul>
      </div>
    </div>
  );
};
