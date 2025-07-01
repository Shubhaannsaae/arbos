import { BigNumber, ethers } from 'ethers';

export class Formatter {
  // BigNumber formatting
  static formatBigNumber(value: BigNumber, decimals: number = 18, precision: number = 4): string {
    return parseFloat(ethers.formatUnits(value, decimals)).toFixed(precision);
  }

  static formatWei(value: BigNumber, precision: number = 4): string {
    return this.formatBigNumber(value, 18, precision);
  }

  static formatGwei(value: BigNumber, precision: number = 2): string {
    return this.formatBigNumber(value, 9, precision);
  }

  static formatUnits(value: BigNumber, decimals: number, precision: number = 4): string {
    return this.formatBigNumber(value, decimals, precision);
  }

  // Currency formatting
  static formatUSD(value: number, precision: number = 2): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: precision,
      maximumFractionDigits: precision
    }).format(value);
  }

  static formatPercentage(value: number, precision: number = 2): string {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: precision,
      maximumFractionDigits: precision
    }).format(value / 100);
  }

  static formatNumber(value: number, precision: number = 2): string {
    return new Intl.NumberFormat('en-US', {
      minimumFractionDigits: precision,
      maximumFractionDigits: precision
    }).format(value);
  }

  // Address formatting
  static shortenAddress(address: string, start: number = 6, end: number = 4): string {
    if (!ethers.isAddress(address)) {
      throw new Error('Invalid Ethereum address');
    }
    return `${address.slice(0, start)}...${address.slice(-end)}`;
  }

  static formatAddress(address: string): string {
    return ethers.getAddress(address); // Checksummed address
  }

  // Transaction hash formatting
  static shortenHash(hash: string, start: number = 8, end: number = 6): string {
    if (!/^0x[a-fA-F0-9]{64}$/.test(hash)) {
      throw new Error('Invalid transaction hash');
    }
    return `${hash.slice(0, start)}...${hash.slice(-end)}`;
  }

  // Time formatting
  static formatTimestamp(timestamp: number, includeTime: boolean = true): string {
    const date = new Date(timestamp * 1000);
    const options: Intl.DateTimeFormatOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    };

    if (includeTime) {
      options.hour = '2-digit';
      options.minute = '2-digit';
      options.second = '2-digit';
      options.timeZoneName = 'short';
    }

    return new Intl.DateTimeFormat('en-US', options).format(date);
  }

  static formatRelativeTime(timestamp: number): string {
    const now = Date.now();
    const diffMs = now - (timestamp * 1000);
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHour < 24) return `${diffHour}h ago`;
    if (diffDay < 7) return `${diffDay}d ago`;
    
    return this.formatTimestamp(timestamp, false);
  }

  static formatDuration(milliseconds: number): string {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }

  // Token amount formatting
  static formatTokenAmount(
    amount: BigNumber,
    decimals: number,
    symbol: string,
    precision: number = 4
  ): string {
    const formatted = this.formatBigNumber(amount, decimals, precision);
    return `${formatted} ${symbol}`;
  }

  static formatTokenValue(
    amount: BigNumber,
    decimals: number,
    priceUsd: number,
    precision: number = 2
  ): string {
    const tokenAmount = parseFloat(ethers.formatUnits(amount, decimals));
    const usdValue = tokenAmount * priceUsd;
    return this.formatUSD(usdValue, precision);
  }

  // Gas formatting
  static formatGasPrice(gasPrice: BigNumber): string {
    const gwei = this.formatGwei(gasPrice, 2);
    return `${gwei} Gwei`;
  }

  static formatGasCost(gasUsed: BigNumber, gasPrice: BigNumber, ethPrice: number): {
    eth: string;
    usd: string;
    gwei: string;
  } {
    const costWei = gasUsed.mul(gasPrice);
    const ethCost = parseFloat(ethers.formatEther(costWei));
    const usdCost = ethCost * ethPrice;

    return {
      eth: this.formatWei(costWei, 6),
      usd: this.formatUSD(usdCost, 2),
      gwei: this.formatGwei(gasPrice, 2)
    };
  }

  // Performance metrics formatting
  static formatAPR(apr: number): string {
    return `${apr.toFixed(2)}% APR`;
  }

  static formatAPY(apy: number): string {
    return `${apy.toFixed(2)}% APY`;
  }

  static formatVolatility(volatility: number): string {
    return `${(volatility * 100).toFixed(2)}%`;
  }

  static formatSharpeRatio(ratio: number): string {
    return ratio.toFixed(3);
  }

  static formatRiskScore(score: number): {
    score: string;
    level: string;
    color: string;
  } {
    const scoreStr = score.toFixed(1);
    let level: string;
    let color: string;

    if (score <= 25) {
      level = 'Low';
      color = 'green';
    } else if (score <= 50) {
      level = 'Medium';
      color = 'yellow';
    } else if (score <= 75) {
      level = 'High';
      color = 'orange';
    } else {
      level = 'Critical';
      color = 'red';
    }

    return { score: scoreStr, level, color };
  }

  // Market data formatting
  static formatPriceChange(change: number): {
    value: string;
    isPositive: boolean;
    formatted: string;
  } {
    const isPositive = change >= 0;
    const sign = isPositive ? '+' : '';
    const percentage = this.formatPercentage(Math.abs(change));
    
    return {
      value: change.toFixed(2),
      isPositive,
      formatted: `${sign}${percentage}`
    };
  }

  static formatVolume(volume: BigNumber | number): string {
    const value = typeof volume === 'number' ? volume : parseFloat(ethers.formatEther(volume));
    
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
    
    return this.formatUSD(value);
  }

  static formatTVL(tvl: BigNumber | number): string {
    return this.formatVolume(tvl);
  }

  // Arbitrage opportunity formatting
  static formatArbitrageOpportunity(opportunity: {
    tokenPair: string;
    priceDifferencePercentage: number;
    potentialProfitPercentage: number;
    netProfit: BigNumber;
    riskScore: number;
    confidence: number;
  }): {
    title: string;
    profitability: string;
    risk: string;
    confidence: string;
  } {
    const priceDiff = this.formatPercentage(opportunity.priceDifferencePercentage);
    const profitPercent = this.formatPercentage(opportunity.potentialProfitPercentage);
    const netProfit = this.formatWei(opportunity.netProfit, 4);
    const risk = this.formatRiskScore(opportunity.riskScore);
    const confidence = this.formatPercentage(opportunity.confidence * 100);

    return {
      title: `${opportunity.tokenPair} (${priceDiff})`,
      profitability: `${profitPercent} (${netProfit} ETH)`,
      risk: `${risk.level} (${risk.score})`,
      confidence
    };
  }

  // Portfolio formatting
  static formatPortfolioAllocation(positions: Array<{
    symbol: string;
    percentage: number;
    value: BigNumber;
  }>): string {
    return positions
      .sort((a, b) => b.percentage - a.percentage)
      .map(pos => `${pos.symbol}: ${this.formatPercentage(pos.percentage)}`)
      .join(', ');
  }

  static formatPortfolioPerformance(performance: {
    totalReturn: BigNumber;
    totalReturnPercentage: number;
    sharpeRatio: number;
    maxDrawdown: number;
  }): {
    return: string;
    sharpe: string;
    drawdown: string;
  } {
    return {
      return: `${this.formatWei(performance.totalReturn, 4)} ETH (${this.formatPercentage(performance.totalReturnPercentage)})`,
      sharpe: this.formatSharpeRatio(performance.sharpeRatio),
      drawdown: this.formatPercentage(performance.maxDrawdown)
    };
  }

  // Chain and network formatting
  static getChainName(chainId: number): string {
    const chains: Record<number, string> = {
      1: 'Ethereum',
      137: 'Polygon',
      42161: 'Arbitrum',
      43114: 'Avalanche',
      56: 'BSC',
      250: 'Fantom',
      10: 'Optimism',
      11155111: 'Sepolia',
      80001: 'Mumbai'
    };
    
    return chains[chainId] || `Chain ${chainId}`;
  }

  static getExplorerUrl(chainId: number, hash: string, type: 'tx' | 'address' = 'tx'): string {
    const explorers: Record<number, string> = {
      1: 'https://etherscan.io',
      137: 'https://polygonscan.com',
      42161: 'https://arbiscan.io',
      43114: 'https://snowtrace.io',
      56: 'https://bscscan.com',
      250: 'https://ftmscan.com',
      10: 'https://optimistic.etherscan.io',
      11155111: 'https://sepolia.etherscan.io',
      80001: 'https://mumbai.polygonscan.com'
    };

    const baseUrl = explorers[chainId] || `https://etherscan.io`;
    return `${baseUrl}/${type}/${hash}`;
  }

  // Utility functions
  static truncateString(str: string, maxLength: number): string {
    if (str.length <= maxLength) return str;
    return str.slice(0, maxLength - 3) + '...';
  }

  static capitalizeFirst(str: string): string {
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }

  static humanizeBytes(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    if (bytes === 0) return '0 Bytes';
    
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }

  static formatLatency(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  // JSON formatting for logs
  static formatLogData(data: any): string {
    try {
      return JSON.stringify(data, (key, value) => {
        if (typeof value === 'bigint') {
          return value.toString();
        }
        if (value && typeof value === 'object' && value._isBigNumber) {
          return value.toString();
        }
        return value;
      }, 2);
    } catch (error) {
      return String(data);
    }
  }
}

export default Formatter;
