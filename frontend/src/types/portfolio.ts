export interface Portfolio {
  id: string;
  owner: string;
  totalValue: number;
  dayChange: number;
  dayChangePercent: number;
  dayPnL: number;
  activePositions: number;
  availableBalance: number;
  allocations: PortfolioAllocation[];
  history: PortfolioHistoryPoint[];
  riskMetrics: PortfolioRiskMetrics;
}

export interface PortfolioPosition {
  token: string;
  symbol: string;
  amount: number;
  value: number;
  percentage: number;
  dayChange: number;
  dayChangePercent: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
}

export interface PortfolioAllocation {
  token: string;
  symbol: string;
  targetPercentage: number;
  currentPercentage: number;
  rebalanceThreshold: number;
  minAllocation: number;
  maxAllocation: number;
}

export interface PortfolioHistoryPoint {
  timestamp: number;
  value: number;
  pnl: number;
  dayChange: number;
  positions: Record<string, number>;
}

export interface PortfolioRiskMetrics {
  sharpeRatio: number;
  maxDrawdown: number;
  volatility: number;
  beta: number;
  var95: number;
  cvar95: number;
  correlation: number;
  diversificationRatio: number;
}

export interface RebalanceSettings {
  enabled: boolean;
  frequency: RebalanceFrequency;
  threshold: number;
  maxSlippage: number;
  riskLevel: RiskLevel;
  useChainlinkPrices: boolean;
  gasOptimization: boolean;
}

export interface RebalanceExecution {
  id: string;
  timestamp: number;
  status: RebalanceStatus;
  trades: RebalanceTrade[];
  gasCost: number;
  slippage: number;
  duration: number;
  reason: string;
}

export interface RebalanceTrade {
  fromToken: string;
  toToken: string;
  fromAmount: number;
  toAmount: number;
  price: number;
  slippage: number;
  txHash: string;
  gasUsed: number;
}

export interface YieldPosition {
  id: string;
  protocol: string;
  pool: string;
  tokens: string[];
  stakedAmount: number;
  rewards: YieldReward[];
  apy: number;
  apr: number;
  tvl: number;
  riskLevel: RiskLevel;
  autoCompound: boolean;
  lockupPeriod?: number;
  withdrawalFee?: number;
}

export interface YieldReward {
  token: string;
  amount: number;
  valueUSD: number;
  claimable: boolean;
  vestingPeriod?: number;
}

export interface YieldStrategy {
  id: string;
  name: string;
  description: string;
  protocols: string[];
  expectedAPY: number;
  riskLevel: RiskLevel;
  minimumAmount: number;
  lockupPeriod?: number;
  autoCompound: boolean;
  rebalanceFrequency: number;
}

export interface PortfolioAnalytics {
  performance: PerformanceAnalytics;
  risk: RiskAnalytics;
  allocation: AllocationAnalytics;
  yield: YieldAnalytics;
  transactions: TransactionAnalytics;
}

export interface PerformanceAnalytics {
  totalReturn: number;
  annualizedReturn: number;
  monthlyReturns: number[];
  bestDay: number;
  worstDay: number;
  winningDays: number;
  totalDays: number;
  calmarRatio: number;
  sortinoRatio: number;
}

export interface RiskAnalytics {
  volatility: number;
  downwardVolatility: number;
  maxDrawdown: number;
  currentDrawdown: number;
  var95: number;
  cvar95: number;
  beta: number;
  correlation: number;
  riskScore: number;
}

export interface AllocationAnalytics {
  diversificationScore: number;
  concentrationRisk: number;
  sectorExposure: Record<string, number>;
  marketCapExposure: Record<string, number>;
  geographicExposure: Record<string, number>;
  correlationMatrix: number[][];
}

export interface YieldAnalytics {
  totalYieldEarned: number;
  averageAPY: number;
  yieldByProtocol: Record<string, number>;
  compoundingEffect: number;
  impermanentLoss: number;
  rewardTokensEarned: Record<string, number>;
}

export interface TransactionAnalytics {
  totalTransactions: number;
  totalFees: number;
  averageFee: number;
  gasOptimizationSavings: number;
  failedTransactions: number;
  mevProtectionSavings: number;
}

// Enums
export enum RebalanceFrequency {
  MANUAL = 'manual',
  DAILY = 'daily',
  WEEKLY = 'weekly',
  MONTHLY = 'monthly',
  THRESHOLD = 'threshold'
}

export enum RebalanceStatus {
  PENDING = 'pending',
  EXECUTING = 'executing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum RiskLevel {
  CONSERVATIVE = 'conservative',
  MODERATE = 'moderate',
  AGGRESSIVE = 'aggressive'
}
