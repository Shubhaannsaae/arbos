import React, { useState, useEffect } from 'react';
import { ExclamationTriangleIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';

interface RiskAlert {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'smart_contract' | 'oracle' | 'market' | 'liquidity' | 'gas' | 'security';
  title: string;
  description: string;
  recommendation: string;
  timestamp: number;
  acknowledged: boolean;
  resolved: boolean;
  affectedComponents: string[];
}

const RISK_ALERTS: RiskAlert[] = [
  {
    id: '1',
    severity: 'medium',
    category: 'gas',
    title: 'High Gas Prices Detected',
    description: 'Network gas prices are currently above 100 gwei, which may affect transaction profitability.',
    recommendation: 'Consider delaying non-urgent transactions or increasing minimum profit thresholds.',
    timestamp: Date.now() - 1800000, // 30 minutes ago
    acknowledged: false,
    resolved: false,
    affectedComponents: ['Arbitrage Engine', 'Portfolio Rebalancer']
  },
  {
    id: '2',
    severity: 'high',
    category: 'oracle',
    title: 'Oracle Price Deviation',
    description: 'Significant price deviation detected between Chainlink feeds and market prices for LINK/USD.',
    recommendation: 'Pause trading for affected pairs until price feeds stabilize.',
    timestamp: Date.now() - 3600000, // 1 hour ago
    acknowledged: true,
    resolved: false,
    affectedComponents: ['LINK Trading', 'Price Monitor']
  },
  {
    id: '3',
    severity: 'low',
    category: 'liquidity',
    title: 'Reduced Liquidity Warning',
    description: 'Liquidity for ETH/USDT pair on Uniswap has decreased by 15% in the last hour.',
    recommendation: 'Monitor for potential slippage increases and adjust position sizes accordingly.',
    timestamp: Date.now() - 7200000, // 2 hours ago
    acknowledged: true,
    resolved: true,
    affectedComponents: ['ETH/USDT Arbitrage']
  },
  {
    id: '4',
    severity: 'critical',
    category: 'security',
    title: 'Suspicious Transaction Pattern',
    description: 'Multiple failed transactions from the same address attempting to exploit contract vulnerabilities.',
    recommendation: 'Immediate investigation required. Consider enabling emergency pause if threat persists.',
    timestamp: Date.now() - 600000, // 10 minutes ago
    acknowledged: false,
    resolved: false,
    affectedComponents: ['Security Module', 'All Trading Functions']
  }
];

export const RiskAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState<RiskAlert[]>(RISK_ALERTS);
  const [filter, setFilter] = useState<'all' | 'unacknowledged' | 'acknowledged' | 'resolved'>('all');
  const [severityFilter, setSeverityFilter] = useState<'all' | 'low' | 'medium' | 'high' | 'critical'>('all');

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
        return 'text-blue-600 bg-blue-100 border-blue-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 border-yellow-200';
      case 'high':
        return 'text-orange-600 bg-orange-100 border-orange-200';
      case 'critical':
        return 'text-red-600 bg-red-100 border-red-200';
      default:
        return 'text-gray-600 bg-gray-100 border-gray-200';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'high':
        return <XCircleIcon className="w-5 h-5" />;
      case 'medium':
        return <ExclamationTriangleIcon className="w-5 h-5" />;
      case 'low':
        return <CheckCircleIcon className="w-5 h-5" />;
      default:
        return null;
    }
  };

  const getCategoryIcon = (category: string) => {
    const icons: { [key: string]: string } = {
      smart_contract: '📜',
      oracle: '🔮',
      market: '📈',
      liquidity: '💧',
      gas: '⛽',
      security: '🛡️'
    };
    return icons[category] || '⚠️';
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  };

  const resolveAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, resolved: true, acknowledged: true } : alert
    ));
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'unacknowledged' && alert.acknowledged) return false;
    if (filter === 'acknowledged' && (!alert.acknowledged || alert.resolved)) return false;
    if (filter === 'resolved' && !alert.resolved) return false;
    if (severityFilter !== 'all' && alert.severity !== severityFilter) return false;
    return true;
  });

  const getAlertCounts = () => {
    return {
      total: alerts.length,
      unacknowledged: alerts.filter(a => !a.acknowledged && !a.resolved).length,
      critical: alerts.filter(a => a.severity === 'critical' && !a.resolved).length,
      high: alerts.filter(a => a.severity === 'high' && !a.resolved).length
    };
  };

  const counts = getAlertCounts();

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900">Risk Alerts</h3>
        <div className="flex space-x-4">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="all">All Alerts</option>
            <option value="unacknowledged">Unacknowledged</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="resolved">Resolved</option>
          </select>
          
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value as any)}
            className="border border-gray-300 rounded-md px-3 py-2 text-sm"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Total Alerts</div>
          <div className="text-2xl font-bold text-gray-900">{counts.total}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Unacknowledged</div>
          <div className="text-2xl font-bold text-red-600">{counts.unacknowledged}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">Critical</div>
          <div className="text-2xl font-bold text-red-600">{counts.critical}</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-sm text-gray-500">High Severity</div>
          <div className="text-2xl font-bold text-orange-600">{counts.high}</div>
        </div>
      </div>

      <div className="space-y-4">
        {filteredAlerts.map((alert) => (
          <div
            key={alert.id}
            className={`border rounded-lg p-6 ${getSeverityColor(alert.severity)} ${
              alert.resolved ? 'opacity-60' : ''
            }`}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <div className="flex items-center space-x-2">
                  <span className="text-xl">{getCategoryIcon(alert.category)}</span>
                  {getSeverityIcon(alert.severity)}
                </div>
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <h4 className="font-semibold text-gray-900">{alert.title}</h4>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                      {alert.severity}
                    </span>
                    <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-700">
                      {alert.category.replace('_', ' ')}
                    </span>
                  </div>
                  
                  <p className="text-gray-700 mb-3">{alert.description}</p>
                  
                  <div className="bg-white bg-opacity-50 rounded p-3 mb-3">
                    <h5 className="font-medium text-gray-900 mb-1">Recommendation:</h5>
                    <p className="text-gray-700 text-sm">{alert.recommendation}</p>
                  </div>
                  
                  <div className="flex flex-wrap gap-1 mb-3">
                    <span className="text-xs text-gray-600">Affected components:</span>
                    {alert.affectedComponents.map((component, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                        {component}
                      </span>
                    ))}
                  </div>
                  
                  <div className="text-xs text-gray-600">
                    {new Date(alert.timestamp).toLocaleString()}
                    {alert.acknowledged && <span className="ml-2 text-blue-600">• Acknowledged</span>}
                    {alert.resolved && <span className="ml-2 text-green-600">• Resolved</span>}
                  </div>
                </div>
              </div>
              
              <div className="flex flex-col space-y-2">
                {!alert.acknowledged && (
                  <button
                    onClick={() => acknowledgeAlert(alert.id)}
                    className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
                  >
                    Acknowledge
                  </button>
                )}
                
                {alert.acknowledged && !alert.resolved && (
                  <button
                    onClick={() => resolveAlert(alert.id)}
                    className="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700"
                  >
                    Resolve
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
        
        {filteredAlerts.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            No alerts match the current filters
          </div>
        )}
      </div>

      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h5 className="text-sm font-medium text-green-900 mb-2">Risk Management Features</h5>
        <ul className="text-sm text-green-800 space-y-1">
          <li>• Real-time monitoring of all system components</li>
          <li>• Automated risk scoring and alert generation</li>
          <li>• Integration with Chainlink oracles for accurate data</li>
          <li>• Emergency pause mechanisms for critical situations</li>
          <li>• Comprehensive logging and audit trails</li>
        </ul>
      </div>
    </div>
  );
};
