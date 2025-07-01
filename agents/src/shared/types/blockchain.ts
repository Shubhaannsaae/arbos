import { BigNumber } from 'ethers';

export interface ChainInfo {
  chainId: number;
  name: string;
  shortName: string;
  network: string;
  networkId: number;
  nativeCurrency: {
    name: string;
    symbol: string;
    decimals: number;
  };
  rpc: string[];
  faucets: string[];
  explorers: Array<{
    name: string;
    url: string;
    standard: string;
  }>;
  infoURL: string;
  slip44?: number;
  ens?: {
    registry: string;
  };
  parent?: {
    type: string;
    chain: string;
    bridges: Array<{
      url: string;
    }>;
  };
}

export interface TransactionDetails {
  hash: string;
  from: string;
  to: string;
  value: BigNumber;
  gasLimit: BigNumber;
  gasUsed: BigNumber;
  gasPrice: BigNumber;
  effectiveGasPrice: BigNumber;
  nonce: number;
  blockNumber: number;
  blockHash: string;
  transactionIndex: number;
  confirmations: number;
  status: 'pending' | 'confirmed' | 'failed';
  timestamp: number;
  data: string;
  logs: Array<{
    address: string;
    topics: string[];
    data: string;
    blockNumber: number;
    transactionHash: string;
    transactionIndex: number;
    blockHash: string;
    logIndex: number;
    removed: boolean;
  }>;
}

export interface ContractCall {
  target: string;
  callData: string;
  value: BigNumber;
  gasLimit: BigNumber;
  requireSuccess: boolean;
}

export interface MulticallRequest {
  calls: ContractCall[];
  chainId: number;
  blockNumber?: number;
  allowFailure: boolean;
}

export interface MulticallResponse {
  blockNumber: number;
  results: Array<{
    success: boolean;
    returnData: string;
    gasUsed: BigNumber;
  }>;
}

export interface BridgeTransaction {
  id: string;
  sourceChain: number;
  destinationChain: number;
  sourceToken: TokenInfo;
  destinationToken: TokenInfo;
  amount: BigNumber;
  bridgeProtocol: string;
  sourceTransactionHash: string;
  destinationTransactionHash?: string;
  status: 'pending' | 'bridging' | 'completed' | 'failed' | 'refunded';
  estimatedTime: number;
  actualTime?: number;
  fees: {
    bridgeFee: BigNumber;
    gasFee: BigNumber;
    protocolFee: BigNumber;
    totalFee: BigNumber;
  };
  slippage: number;
  recipient: string;
  deadline: number;
  ccipMessageId?: string;
  createdAt: number;
  completedAt?: number;
}

export interface ChainlinkFeed {
  pair: string;
  address: string;
  chainId: number;
  decimals: number;
  heartbeat: number;
  deviation: number;
  description: string;
  version: number;
  aggregator: string;
  status: 'active' | 'deprecated' | 'inactive';
  category: string;
  feedType: 'price' | 'rates' | 'commodity' | 'weather' | 'sports';
}

export interface ChainlinkAutomationUpkeep {
  upkeepId: string;
  target: string;
  executeGas: BigNumber;
  checkData: string;
  balance: BigNumber;
  admin: string;
  maxValidBlocknumber: BigNumber;
  lastPerformed: BigNumber;
  amountSpent: BigNumber;
  paused: boolean;
  offchainConfig: string;
}

export interface ChainlinkFunctionsRequest {
  requestId: string;
  subscriptionId: number;
  consumer: string;
  initiatedRequests: BigNumber;
  completedRequests: BigNumber;
  codeLocation: number;
  secretsLocation: number;
  source: string;
  args: string[];
  subscriptionOwner: string;
  callbackGasLimit: BigNumber;
  estimatedCost: BigNumber;
  donFee: BigNumber;
  registryFee: BigNumber;
  estimatedGasPrice: BigNumber;
  fulfillmentCode: number;
  response: string;
  errors: string;
  createdAt: number;
  fulfilledAt?: number;
}

export interface ChainlinkVRFRequest {
  requestId: BigNumber;
  preSeed: BigNumber;
  blockNum: BigNumber;
  subId: BigNumber;
  callbackGasLimit: BigNumber;
  numWords: number;
  sender: string;
  paid: BigNumber;
  fulfilled: boolean;
  randomWords: BigNumber[];
  createdAt: number;
  fulfilledAt?: number;
}

export interface CCIPMessage {
  messageId: string;
  sourceChainSelector: BigNumber;
  destinationChainSelector: BigNumber;
  sequenceNumber: BigNumber;
  feeToken: string;
  feeTokenAmount: BigNumber;
  data: string;
  tokenAmounts: Array<{
    token: string;
    amount: BigNumber;
  }>;
  sourceTokenData: string[];
  sender: string;
  receiver: string;
  gasLimit: BigNumber;
  status: 'sent' | 'committed' | 'blessed' | 'executed' | 'failed';
  createdAt: number;
  executedAt?: number;
}

export interface MEVProtection {
  enabled: boolean;
  type: 'flashbots' | 'eden' | 'manifold' | 'custom';
  configuration: {
    maxBribe: BigNumber;
    minConfirmations: number;
    protectFromSandwich: boolean;
    protectFromFrontrunning: boolean;
    usePrivateMempool: boolean;
  };
}

export interface GasEstimation {
  gasLimit: BigNumber;
  gasPrice: BigNumber;
  maxFeePerGas: BigNumber;
  maxPriorityFeePerGas: BigNumber;
  totalCost: BigNumber;
  confidence: number;
  estimatedWait: number;
  type: 'legacy' | 'eip1559';
}

export interface NetworkStatus {
  chainId: number;
  blockNumber: number;
  blockTime: number;
  gasPrice: BigNumber;
  baseFee?: BigNumber;
  nextBaseFee?: BigNumber;
  congestion: number;
  tps: number;
  pendingTransactions: number;
  networkHashrate?: BigNumber;
  difficulty?: BigNumber;
  validators?: number;
  stakingRate?: number;
  totalValueLocked: BigNumber;
  bridgeCapacity: BigNumber;
  lastUpdated: number;
}

export interface SecurityAlert {
  id: string;
  type: 'transaction' | 'contract' | 'bridge' | 'governance' | 'oracle';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  affectedAddresses: string[];
  affectedChains: number[];
  indicators: Array<{
    type: string;
    value: any;
    threshold: any;
    description: string;
  }>;
  recommendations: string[];
  sources: string[];
  confidence: number;
  falsePositiveRate: number;
  detectedAt: number;
  resolvedAt?: number;
  resolution?: string;
}

export interface TokenInfo {
  address: string;
  symbol: string;
  name: string;
  decimals: number;
  chainId: number;
  logoURI?: string;
  tags: string[];
  isStable: boolean;
  isNative: boolean;
}

export interface DeFiProtocol {
  id: string;
  name: string;
  description: string;
  website: string;
  logo: string;
  chains: number[];
  category: string[];
  tvl: BigNumber;
  volume24h: BigNumber;
  fees24h: BigNumber;
  revenue24h: BigNumber;
  users24h: number;
  transactions24h: number;
  auditStatus: {
    audited: boolean;
    auditors: string[];
    auditDate?: number;
    securityScore: number;
  };
  governance: {
    hasGovernanceToken: boolean;
    governanceToken?: TokenInfo;
    decentralized: boolean;
    multisig?: string;
  };
  contracts: Array<{
    name: string;
    address: string;
    chainId: number;
    verified: boolean;
    proxy: boolean;
  }>;
  risks: {
    smartContract: number;
    governance: number;
    oracle: number;
    bridge: number;
    custody: number;
    overall: number;
  };
}
