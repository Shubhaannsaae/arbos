import { ethers } from 'ethers';

// Base blockchain types
export interface ChainConfig {
  chainId: number;
  name: string;
  symbol: string;
  decimals: number;
  rpcUrl: string;
  blockExplorer: string;
  faucet?: string;
  testnet: boolean;
}

// Chainlink service interfaces
export interface ChainlinkDataFeed {
  address: string;
  pair: string;
  decimals: number;
  heartbeat: number;
  deviation: number;
}

export interface ChainlinkPriceData {
  price: number;
  decimals: number;
  timestamp: number;
  roundId: string;
  updatedAt: number;
}

export interface ChainlinkVRFRequest {
  requestId: string;
  keyHash: string;
  subId: number;
  callbackGasLimit: number;
  requestConfirmations: number;
  numWords: number;
  sender: string;
  timestamp: number;
}

export interface ChainlinkVRFResponse {
  requestId: string;
  randomWords: number[];
  payment: bigint;
  success: boolean;
  timestamp: number;
}

export interface ChainlinkCCIPMessage {
  messageId: string;
  sourceChainSelector: bigint;
  destinationChainSelector: bigint;
  sender: string;
  receiver: string;
  data: string;
  tokenAmounts: {
    token: string;
    amount: bigint;
  }[];
  feeToken: string;
  fee: bigint;
  gasLimit: number;
  status: 'pending' | 'success' | 'failed';
  timestamp: number;
}

export interface ChainlinkFunctionRequest {
  requestId: string;
  source: string;
  args: string[];
  subscriptionId: number;
  gasLimit: number;
  donId: string;
  timestamp: number;
}

export interface ChainlinkFunctionResponse {
  requestId: string;
  result: string;
  error: string;
  timestamp: number;
}

export interface ChainlinkAutomationUpkeep {
  upkeepId: string;
  target: string;
  executeGas: number;
  checkData: string;
  balance: bigint;
  admin: string;
  maxValidBlocknumber: number;
  lastPerformBlockNumber: number;
  amountSpent: bigint;
  paused: boolean;
}

// Transaction types
export interface Transaction {
  hash: string;
  from: string;
  to: string;
  value: bigint;
  gasLimit: bigint;
  gasPrice: bigint;
  gasUsed?: bigint;
  blockNumber?: number;
  timestamp?: number;
  status?: 'pending' | 'confirmed' | 'failed';
}

export interface ContractCall {
  address: string;
  abi: any[];
  method: string;
  params: any[];
  value?: bigint;
  gasLimit?: number;
}

// API Response types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// Error types
export interface AppError {
  code: string;
  message: string;
  details?: any;
  timestamp: number;
}

export interface ValidationError {
  field: string;
  message: string;
  value: any;
}

// UI State types
export interface LoadingState {
  loading: boolean;
  error: string | null;
  lastUpdated?: number;
}

export interface ModalState {
  isOpen: boolean;
  title?: string;
  content?: React.ReactNode;
  onClose?: () => void;
}

// User preferences
export interface UserPreferences {
  theme: 'light' | 'dark';
  currency: 'USD' | 'EUR' | 'GBP';
  notifications: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
  defaultChain: number;
  slippageTolerance: number;
  gasPreference: 'slow' | 'standard' | 'fast';
}

// Network status
export interface NetworkStatus {
  chainId: number;
  blockNumber: number;
  gasPrice: bigint;
  timestamp: number;
  healthy: boolean;
}

// Token information
export interface TokenInfo {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  totalSupply?: bigint;
  logoURI?: string;
  chainId: number;
}

export interface TokenBalance {
  token: TokenInfo;
  balance: bigint;
  balanceFormatted: string;
  valueUSD?: number;
}

// Common enums
export enum TransactionStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  FAILED = 'failed'
}

export enum ChainlinkService {
  DATA_FEEDS = 'data_feeds',
  VRF = 'vrf',
  CCIP = 'ccip',
  FUNCTIONS = 'functions',
  AUTOMATION = 'automation'
}
