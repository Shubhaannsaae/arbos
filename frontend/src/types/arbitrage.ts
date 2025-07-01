export interface ArbitrageOpportunity {
  id: string;
  tokenPair: string;
  sourceExchange: string;
  targetExchange: string;
  sourcePrice: number;
  targetPrice: number;
  profitPercent: number;
  maxProfit: number;
  maxAmount: number;
  gasCost: number;
  netProfit: number;
  confidence: number;
  timestamp: number;
  chainId: number;
  liquidityScore: number;
  slippageImpact: number;
  executionTime: number;
}

export interface ArbitrageExecution {
  id: string;
  opportunityId: string;
  tokenPair: string;
  sourceExchange: string;
  targetExchange: string;
  amount: number;
  expectedProfit: number;
  actualProfit: number;
  profitPercent: number;
  status: ArbitrageStatus;
  txHash: string;
  blockNumber?: number;
  gasUsed?: number;
  gasCost: number;
  slippage: number;
  executionTime: number;
  timestamp: number;
  chainId: number;
  errorMessage?: string;
}

export interface ArbitrageRoute {
  id: string;
  path: string[];
  exchanges: string[];
  expectedReturn: number;
  gasEstimate: number;
  complexity: number;
  riskScore: number;
  executionSteps: ArbitrageStep[];
}

export interface ArbitrageStep {
  stepNumber: number;
  action: 'buy' | 'sell' | 'swap' | 'transfer';
  exchange: string;
  tokenIn: string;
  tokenOut: string;
  amountIn: number;
  amountOut: number;
  price: number;
  gasEstimate: number;
  slippageTolerance: number;
}

export interface ArbitrageStrategy {
  id: string;
  name: string;
  description: string;
  type: ArbitrageType;
  minProfitThreshold: number;
  maxGasPrice: number;
  maxSlippage: number;
  riskLevel: RiskLevel;
  enabled: boolean;
  parameters: ArbitrageParameters;
}

export interface ArbitrageParameters {
  minLiquidity: number;
  maxPositionSize: number;
  priceImpactThreshold: number;
  monitoredPairs: string[];
  excludedExchanges: string[];
  gasOptimization: boolean;
  mevProtection: boolean;
  chainlinkPriceValidation: boolean;
  flashLoanEnabled: boolean;
  flashLoanProviders: string[];
}

export interface ArbitrageMetrics {
  totalExecutions: number;
  successfulExecutions: number;
  failedExecutions: number;
  totalProfit: number;
  totalGasCost: number;
  netProfit: number;
  averageProfit: number;
  averageExecutionTime: number;
  successRate: number;
  profitability: number;
  riskAdjustedReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
}

export interface ArbitrageAlert {
  id: string;
  type: ArbitrageAlertType;
  severity: AlertSeverity;
  title: string;
  message: string;
  opportunityId?: string;
  executionId?: string;
  timestamp: number;
  acknowledged: boolean;
  autoResolved: boolean;
}

export interface PriceMonitor {
  tokenPair: string;
  exchanges: ExchangePrice[];
  chainlinkPrice?: number;
  deviation: number;
  maxDeviation: number;
  lastUpdated: number;
  alerts: PriceAlert[];
}

export interface ExchangePrice {
  exchange: string;
  price: number;
  liquidity: number;
  timestamp: number;
  spread: number;
  volume24h: number;
}

export interface PriceAlert {
  type: 'deviation' | 'manipulation' | 'stale';
  message: string;
  severity: AlertSeverity;
  timestamp: number;
  threshold: number;
  actualValue: number;
}

export interface FlashLoanOpportunity {
  id: string;
  provider: string;
  asset: string;
  maxAmount: number;
  fee: number;
  feePercent: number;
  routes: ArbitrageRoute[];
  estimatedProfit: number;
  gasEstimate: number;
  riskScore: number;
}

export interface MEVProtection {
  enabled: boolean;
  provider: string;
  protectionLevel: 'basic' | 'standard' | 'maximum';
  cost: number;
  successRate: number;
  avgSavings: number;
}

export interface GasOptimization {
  enabled: boolean;
  strategy: 'time' | 'cost' | 'speed';
  maxGasPrice: number;
  gasBuffer: number;
  bundleTransactions: boolean;
  estimatedSavings: number;
}

// Enums
export enum ArbitrageStatus {
  PENDING = 'pending',
  EXECUTING = 'executing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum ArbitrageType {
  SIMPLE = 'simple',
  TRIANGULAR = 'triangular',
  CROSS_CHAIN = 'cross_chain',
  FLASH_LOAN = 'flash_loan',
  STATISTICAL = 'statistical'
}

export enum ArbitrageAlertType {
  OPPORTUNITY_FOUND = 'opportunity_found',
  EXECUTION_STARTED = 'execution_started',
  EXECUTION_COMPLETED = 'execution_completed',
  EXECUTION_FAILED = 'execution_failed',
  HIGH_SLIPPAGE = 'high_slippage',
  MEV_DETECTED = 'mev_detected',
  PRICE_MANIPULATION = 'price_manipulation'
}

export enum AlertSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high'
}

// Request/Response types for API
export interface ArbitrageOpportunityRequest {
  tokenPairs?: string[];
  exchanges?: string[];
  minProfit?: number;
  maxGasPrice?: number;
  chainIds?: number[];
}

export interface ExecuteArbitrageRequest {
  opportunityId: string;
  amount: number;
  slippageTolerance: number;
  gasPrice?: number;
  deadline?: number;
}

export interface ArbitrageHistoryRequest {
  startDate?: number;
  endDate?: number;
  status?: ArbitrageStatus;
  tokenPair?: string;
  limit?: number;
  offset?: number;
}
