export interface Agent {
  id: string;
  name: string;
  type: AgentType;
  status: AgentStatus;
  config: AgentConfig;
  performance: AgentPerformance;
  lastExecution?: AgentExecution;
}

export interface AgentConfig {
  name: string;
  type: AgentType;
  strategy: AgentStrategy;
  maxAmount: string;
  riskLevel: RiskLevel;
  enabled: boolean;
  parameters: AgentParameters;
}

export interface AgentParameters {
  minProfitThreshold: number;
  maxSlippage: number;
  gasLimit: number;
  stopLoss?: number;
  takeProfit?: number;
  useChainlinkDataFeeds?: boolean;
  useChainlinkVRF?: boolean;
  useChainlinkAutomation?: boolean;
  dataFeedFrequency?: string;
}

export interface AgentPerformance {
  totalReturn: number;
  tradesExecuted: number;
  successRate: number;
  lastExecution: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
  volatility?: number;
}

export interface AgentExecution {
  timestamp: number;
  action: string;
  success: boolean;
  amount: number;
  result: string;
  txHash?: string;
  gasUsed?: number;
  profit?: number;
}

export interface AgentMetrics {
  totalReturn: number;
  totalReturnUSD: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  trades: {
    successful: number;
    failed: number;
    pending: number;
  };
  recentTrades: AgentTrade[];
  performance: AgentPerformanceData[];
  avgTradeSize: number;
  avgHoldTime: number;
  bestTrade: number;
  worstTrade: number;
  totalGasUsed: number;
}

export interface AgentTrade {
  timestamp: number;
  pair: string;
  profit: number;
  profitPercent: number;
  success: boolean;
  txHash?: string;
}

export interface AgentPerformanceData {
  timestamp: number;
  cumulativeReturn: number;
  dailyReturn: number;
}

export interface AgentAlert {
  id: string;
  agentId: string;
  type: AlertType;
  severity: AlertSeverity;
  message: string;
  timestamp: number;
  acknowledged: boolean;
}

// Enums
export enum AgentType {
  ARBITRAGE = 'arbitrage',
  PORTFOLIO = 'portfolio',
  YIELD = 'yield',
  MARKET_MAKING = 'market_making'
}

export enum AgentStatus {
  ACTIVE = 'active',
  PAUSED = 'paused',
  ERROR = 'error',
  STOPPED = 'stopped'
}

export enum AgentStrategy {
  CONSERVATIVE = 'conservative',
  MODERATE = 'moderate',
  AGGRESSIVE = 'aggressive'
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high'
}

export enum AlertType {
  PERFORMANCE = 'performance',
  ERROR = 'error',
  RISK = 'risk',
  EXECUTION = 'execution'
}

export enum AlertSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Agent creation and management
export interface CreateAgentRequest {
  name: string;
  type: AgentType;
  strategy: AgentStrategy;
  maxAmount: string;
  riskLevel: RiskLevel;
  parameters: Partial<AgentParameters>;
}

export interface UpdateAgentRequest {
  agentId: string;
  config: Partial<AgentConfig>;
}

export interface AgentBacktest {
  agentId: string;
  startDate: number;
  endDate: number;
  initialCapital: number;
  results: BacktestResults;
}

export interface BacktestResults {
  finalValue: number;
  totalReturn: number;
  annualizedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  winRate: number;
  totalTrades: number;
  profitFactor: number;
  dailyReturns: Array<{
    date: number;
    return: number;
    cumulativeReturn: number;
  }>;
}
