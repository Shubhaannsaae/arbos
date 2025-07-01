export interface Portfolio {
  id: string;
  userId: string;
  name: string;
  description?: string;
  targetAllocation: TokenAllocation[];
  currentAllocation: TokenAllocation[];
  totalValueUSD: number;
  totalValueETH: number;
  performanceMetrics: PerformanceMetrics;
  riskScore: number;
  rebalanceSettings: RebalanceSettings;
  restrictions: PortfolioRestrictions;
  createdAt: Date;
  updatedAt: Date;
  lastRebalanced: Date;
  isActive: boolean;
  chainId: number;
}

export interface TokenAllocation {
  tokenAddress: string;
  symbol: string;
  name: string;
  percentage: number;
  amount: string;
  valueUSD: number;
  decimals: number;
  priceUSD: number;
  chainId: number;
}

export interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPercentage: number;
  dailyReturn: number;
  weeklyReturn: number;
  monthlyReturn: number;
  yearlyReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  beta: number;
  alpha: number;
  winRate: number;
  profitFactor: number;
}

export interface RebalanceSettings {
  enabled: boolean;
  trigger: RebalanceTrigger;
  threshold: number;
  frequency: RebalanceFrequency;
  slippageTolerance: number;
  gasOptimization: boolean;
  minTradeSize: number;
}

export interface PortfolioRestrictions {
  maxTokens: number;
  minTokenPercentage: number;
  maxTokenPercentage: number;
  allowedTokens?: string[];
  blockedTokens?: string[];
  allowedProtocols?: string[];
  blockedProtocols?: string[];
}

export enum RebalanceTrigger {
  THRESHOLD = 'threshold',
  TIME = 'time',
  VOLATILITY = 'volatility',
  MARKET_CONDITIONS = 'market_conditions'
}

export enum RebalanceFrequency {
  DAILY = 'daily',
  WEEKLY = 'weekly',
  MONTHLY = 'monthly',
  QUARTERLY = 'quarterly'
}

export interface CreatePortfolioDto {
  name: string;
  description?: string;
  targetAllocation: TokenAllocation[];
  rebalanceSettings: RebalanceSettings;
  restrictions?: PortfolioRestrictions;
  chainId: number;
}

export interface UpdatePortfolioDto {
  name?: string;
  description?: string;
  targetAllocation?: TokenAllocation[];
  rebalanceSettings?: RebalanceSettings;
  restrictions?: PortfolioRestrictions;
}
