import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useWeb3 } from '../hooks/useWeb3';
import { ConnectWallet } from '../components/common/ConnectWallet';
import { formatCurrency } from '../utils/formatters';

const FEATURES = [
  {
    title: 'AI-Powered Arbitrage',
    description: 'Automated cross-chain arbitrage opportunities powered by Chainlink price feeds',
    icon: '⚡',
    benefits: ['Real-time opportunity detection', 'MEV protection', 'Gas optimization']
  },
  {
    title: 'Smart Portfolio Management',
    description: 'Intelligent portfolio rebalancing with risk management and yield optimization',
    icon: '📊',
    benefits: ['Automated rebalancing', 'Risk analytics', 'Yield farming']
  },
  {
    title: 'Cross-Chain Operations',
    description: 'Seamless cross-chain transactions using Chainlink CCIP',
    icon: '🌐',
    benefits: ['Multi-chain support', 'Secure messaging', 'Unified liquidity']
  },
  {
    title: 'Enterprise Security',
    description: 'Bank-grade security with comprehensive monitoring and alerts',
    icon: '🛡️',
    benefits: ['Real-time monitoring', 'Risk alerts', 'Audit trails']
  }
];

const SUPPORTED_CHAINS = [
  { name: 'Ethereum', logo: '⟠', color: 'text-blue-600' },
  { name: 'Avalanche', logo: '🔺', color: 'text-red-600' },
  { name: 'Polygon', logo: '⬢', color: 'text-purple-600' },
  { name: 'Arbitrum', logo: '🔷', color: 'text-blue-500' }
];

export default function Home() {
  const { isConnected, account } = useWeb3();
  const [stats, setStats] = useState({
    totalValue: 0,
    dailyVolume: 0,
    activeUsers: 0,
    totalTrades: 0
  });

  useEffect(() => {
    // Simulate loading stats
    setStats({
      totalValue: 156789234,
      dailyVolume: 2345678,
      activeUsers: 1247,
      totalTrades: 45623
    });
  }, []);

  return (
    <>
      <Head>
        <title>ArbOS - AI-Powered Cross-Chain Trading Platform</title>
        <meta name="description" content="Advanced arbitrage and portfolio management with Chainlink integration" />
      </Head>

      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
        {/* Header */}
        <header className="relative z-10">
          <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center">
                <h1 className="text-2xl font-bold text-white">ArbOS</h1>
              </div>
              <div className="flex items-center space-x-4">
                <ConnectWallet />
                {isConnected && (
                  <Link href="/dashboard" className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors">
                    Dashboard
                  </Link>
                )}
              </div>
            </div>
          </nav>
        </header>

        {/* Hero Section */}
        <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              AI-Powered
              <span className="block gradient-text">Cross-Chain Trading</span>
            </h1>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Advanced arbitrage and portfolio management platform powered by Chainlink's 
              decentralized oracle networks for maximum security and reliability.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
              {!isConnected ? (
                <ConnectWallet />
              ) : (
                <Link href="/dashboard" className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors">
                  Go to Dashboard
                </Link>
              )}
              <Link href="#features" className="border border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-gray-900 transition-colors">
                Learn More
              </Link>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto">
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-2">
                  {formatCurrency(stats.totalValue, 0)}
                </div>
                <div className="text-gray-400">Total Value Locked</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-2">
                  {formatCurrency(stats.dailyVolume, 0)}
                </div>
                <div className="text-gray-400">24h Volume</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-2">
                  {stats.activeUsers.toLocaleString()}+
                </div>
                <div className="text-gray-400">Active Users</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-white mb-2">
                  {stats.totalTrades.toLocaleString()}+
                </div>
                <div className="text-gray-400">Total Trades</div>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">
              Powered by Chainlink
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Built on the most secure and reliable decentralized oracle network
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {FEATURES.map((feature, index) => (
              <div key={index} className="glass-effect rounded-xl p-6 text-center">
                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                <p className="text-gray-300 mb-4">{feature.description}</p>
                <ul className="space-y-2">
                  {feature.benefits.map((benefit, i) => (
                    <li key={i} className="text-sm text-gray-400 flex items-center">
                      <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                      {benefit}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Supported Chains */}
        <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-4">
              Multi-Chain Support
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Trade across multiple blockchains with unified liquidity
            </p>
          </div>

          <div className="flex justify-center items-center space-x-8 md:space-x-16">
            {SUPPORTED_CHAINS.map((chain, index) => (
              <div key={index} className="text-center">
                <div className={`text-4xl mb-2 ${chain.color}`}>
                  {chain.logo}
                </div>
                <div className="text-white font-semibold">{chain.name}</div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="glass-effect rounded-xl p-12 text-center">
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to Start Trading?
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Join thousands of traders using ArbOS to maximize their DeFi returns
            </p>
            
            {!isConnected ? (
              <ConnectWallet />
            ) : (
              <Link href="/dashboard" className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors">
                Access Platform
              </Link>
            )}
          </div>
        </section>

        {/* Footer */}
        <footer className="relative z-10 border-t border-gray-800 mt-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
              <div className="col-span-1 md:col-span-2">
                <h3 className="text-white font-bold text-xl mb-4">ArbOS</h3>
                <p className="text-gray-400 mb-4">
                  AI-powered cross-chain trading platform built on Chainlink's decentralized oracle network.
                </p>
                <div className="flex space-x-4">
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">Twitter</a>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">Discord</a>
                  <a href="#" className="text-gray-400 hover:text-white transition-colors">GitHub</a>
                </div>
              </div>
              
              <div>
                <h4 className="text-white font-semibold mb-4">Platform</h4>
                <ul className="space-y-2">
                  <li><a href="/dashboard" className="text-gray-400 hover:text-white transition-colors">Dashboard</a></li>
                  <li><a href="/arbitrage" className="text-gray-400 hover:text-white transition-colors">Arbitrage</a></li>
                  <li><a href="/portfolio" className="text-gray-400 hover:text-white transition-colors">Portfolio</a></li>
                  <li><a href="/agents" className="text-gray-400 hover:text-white transition-colors">AI Agents</a></li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-white font-semibold mb-4">Resources</h4>
                <ul className="space-y-2">
                  <li><a href="#" className="text-gray-400 hover:text-white transition-colors">Documentation</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white transition-colors">API</a></li>
                  <li><a href="#" className="text-gray-400 hover:text-white transition-colors">Support</a></li>
                  <li><a href="/security" className="text-gray-400 hover:text-white transition-colors">Security</a></li>
                </ul>
              </div>
            </div>
            
            <div className="border-t border-gray-800 mt-8 pt-8 text-center">
              <p className="text-gray-400">
                © 2025 ArbOS. All rights reserved. Powered by Chainlink.
              </p>
            </div>
          </div>
        </footer>

        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
          <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
          <div className="absolute top-40 left-40 w-80 h-80 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
        </div>
      </div>
    </>
  );
}
