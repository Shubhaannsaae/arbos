export interface Transaction {
  id: string;
  userId: string;
  portfolioId?: string;
  agentId?: string;
  txHash: string;
  chainId: number;
  type: TransactionType;
  status: TransactionStatus;
  fromAddress: string;
  toAddress: string;
  tokenAddress?: string;
  amount: string;
  amountUSD?: number;
  gasUsed?: string;
  gasPrice: string;
  gasCost: string;
  gasCostUSD?: number;
  blockNumber?: string;
  blockTimestamp?: Date;
  nonce: number;
  data?: string;
  logs?: TransactionLog[];
  metadata: TransactionMetadata;
  createdAt: Date;
  updatedAt: Date;
  confirmedAt?: Date;
  failureReason?: string;
}

export interface TransactionLog {
  address: string;
  topics: string[];
  data: string;
  logIndex: number;
  transactionIndex: number;
  removed: boolean;
}

export interface TransactionMetadata {
  source: TransactionSource;
  category: TransactionCategory;
  tags: string[];
  description?: string;
  relatedTransactions?: string[];
  arbitrageOpportunityId?: string;
  estimatedProfit?: number;
  actualProfit?: number;
  slippage?: number;
  priceImpact?: number;
}

export enum TransactionType {
  SWAP = 'swap',
  TRANSFER = 'transfer',
  APPROVE = 'approve',
  DEPOSIT = 'deposit',
  WITHDRAW = 'withdraw',
  STAKE = 'stake',
  UNSTAKE = 'unstake',
  CLAIM = 'claim',
  ARBITRAGE = 'arbitrage',
  REBALANCE = 'rebalance',
  LIQUIDATION = 'liquidation',
  CROSS_CHAIN = 'cross_chain'
}

export enum TransactionStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
  REPLACED = 'replaced'
}

export enum TransactionSource {
  USER = 'user',
  ARBITRAGE_AGENT = 'arbitrage_agent',
  PORTFOLIO_AGENT = 'portfolio_agent',
  YIELD_AGENT = 'yield_agent',
  SECURITY_AGENT = 'security_agent',
  ORCHESTRATOR_AGENT = 'orchestrator_agent'
}

export enum TransactionCategory {
  TRADING = 'trading',
  PORTFOLIO_MANAGEMENT = 'portfolio_management',
  YIELD_FARMING = 'yield_farming',
  ARBITRAGE = 'arbitrage',
  SECURITY = 'security',
  CROSS_CHAIN = 'cross_chain'
}

export interface CreateTransactionDto {
  txHash: string;
  chainId: number;
  type: TransactionType;
  fromAddress: string;
  toAddress: string;
  tokenAddress?: string;
  amount: string;
  gasPrice: string;
  nonce: number;
  metadata: TransactionMetadata;
  portfolioId?: string;
  agentId?: string;
}
