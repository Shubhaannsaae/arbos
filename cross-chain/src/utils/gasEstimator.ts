import { ethers, Provider, TransactionRequest } from 'ethers';

export interface GasEstimate {
  gasLimit: bigint;
  gasPrice: bigint;
  maxFeePerGas?: bigint;
  maxPriorityFeePerGas?: bigint;
  totalCost: bigint;
  costInUSD?: number;
}

export interface GasConfig {
  bufferMultiplier: number;
  priorityLevel: 'slow' | 'standard' | 'fast' | 'instant';
  maxGasPrice?: bigint;
  customGasPrice?: bigint;
}

/**
 * Estimate gas with buffer for transaction
 */
export async function estimateGasWithBuffer(
  provider: Provider,
  txRequest: TransactionRequest,
  config: Partial<GasConfig> = {}
): Promise<GasEstimate> {
  const {
    bufferMultiplier = 1.2,
    priorityLevel = 'standard',
    maxGasPrice,
    customGasPrice
  } = config;

  // Estimate gas limit
  const estimatedGasLimit = await provider.estimateGas(txRequest);
  const gasLimit = BigInt(Math.ceil(Number(estimatedGasLimit) * bufferMultiplier));

  // Get gas price based on network type
  const feeData = await provider.getFeeData();
  const network = await provider.getNetwork();
  
  let gasPrice: bigint;
  let maxFeePerGas: bigint | undefined;
  let maxPriorityFeePerGas: bigint | undefined;

  if (customGasPrice) {
    gasPrice = customGasPrice;
  } else if (feeData.maxFeePerGas && feeData.maxPriorityFeePerGas) {
    // EIP-1559 network
    const baseFee = feeData.maxFeePerGas - feeData.maxPriorityFeePerGas;
    maxPriorityFeePerGas = calculatePriorityFee(feeData.maxPriorityFeePerGas, priorityLevel);
    maxFeePerGas = baseFee + maxPriorityFeePerGas;
    
    if (maxGasPrice && maxFeePerGas > maxGasPrice) {
      maxFeePerGas = maxGasPrice;
    }
    
    gasPrice = maxFeePerGas;
  } else {
    // Legacy network
    gasPrice = calculateLegacyGasPrice(feeData.gasPrice || 0n, priorityLevel);
    
    if (maxGasPrice && gasPrice > maxGasPrice) {
      gasPrice = maxGasPrice;
    }
  }

  const totalCost = gasLimit * gasPrice;

  return {
    gasLimit,
    gasPrice,
    maxFeePerGas,
    maxPriorityFeePerGas,
    totalCost
  };
}

/**
 * Calculate priority fee based on level
 */
function calculatePriorityFee(basePriorityFee: bigint, level: string): bigint {
  const multipliers: { [key: string]: number } = {
    slow: 0.8,
    standard: 1.0,
    fast: 1.5,
    instant: 2.0
  };
  
  const multiplier = multipliers[level] || 1.0;
  return BigInt(Math.floor(Number(basePriorityFee) * multiplier));
}

/**
 * Calculate legacy gas price based on level
 */
function calculateLegacyGasPrice(baseGasPrice: bigint, level: string): bigint {
  const multipliers: { [key: string]: number } = {
    slow: 1.0,
    standard: 1.1,
    fast: 1.25,
    instant: 1.5
  };
  
  const multiplier = multipliers[level] || 1.1;
  return BigInt(Math.floor(Number(baseGasPrice) * multiplier));
}

/**
 * Estimate gas for CCIP message
 */
export async function estimateCCIPGas(
  provider: Provider,
  destinationChainSelector: bigint,
  messageSize: number,
  tokenTransfers: number = 0
): Promise<bigint> {
  // Base gas for CCIP operations
  const baseGas = 200000n;
  
  // Gas per byte of message data
  const gasPerByte = 16n;
  const messageGas = BigInt(messageSize) * gasPerByte;
  
  // Additional gas for token transfers
  const tokenGas = BigInt(tokenTransfers) * 50000n;
  
  // Chain-specific multipliers
  const chainMultipliers: { [key: string]: number } = {
    '5009297550715157269': 1.2, // Ethereum
    '6433500567565415381': 1.1, // Avalanche
    '4051577828743386545': 1.3, // Polygon
    '4949039107694359620': 1.1  // Arbitrum
  };
  
  const multiplier = chainMultipliers[destinationChainSelector.toString()] || 1.2;
  const totalGas = baseGas + messageGas + tokenGas;
  
  return BigInt(Math.ceil(Number(totalGas) * multiplier));
}

/**
 * Estimate gas for cross-chain token transfer
 */
export async function estimateTokenTransferGas(
  provider: Provider,
  tokenAddress: string,
  amount: bigint,
  isNative: boolean = false
): Promise<bigint> {
  if (isNative) {
    return 21000n; // Standard ETH transfer
  }
  
  // ERC20 transfer gas estimation
  const erc20TransferGas = 65000n;
  
  // Additional gas for complex tokens (with hooks, etc.)
  const bufferGas = 10000n;
  
  return erc20TransferGas + bufferGas;
}

/**
 * Calculate gas cost in native token
 */
export function calculateGasCostNative(gasLimit: bigint, gasPrice: bigint): string {
  const cost = gasLimit * gasPrice;
  return ethers.formatEther(cost);
}

/**
 * Calculate gas cost in USD (requires price feed)
 */
export function calculateGasCostUSD(
  gasLimit: bigint,
  gasPrice: bigint,
  nativeTokenPriceUSD: number
): number {
  const costInNative = parseFloat(calculateGasCostNative(gasLimit, gasPrice));
  return costInNative * nativeTokenPriceUSD;
}

/**
 * Get recommended gas settings for different scenarios
 */
export function getRecommendedGasSettings(scenario: 'defi' | 'nft' | 'ccip' | 'standard'): Partial<GasConfig> {
  const settings: { [key: string]: Partial<GasConfig> } = {
    defi: {
      bufferMultiplier: 1.3,
      priorityLevel: 'fast'
    },
    nft: {
      bufferMultiplier: 1.2,
      priorityLevel: 'standard'
    },
    ccip: {
      bufferMultiplier: 1.5,
      priorityLevel: 'standard'
    },
    standard: {
      bufferMultiplier: 1.2,
      priorityLevel: 'standard'
    }
  };
  
  return settings[scenario] || settings.standard;
}

/**
 * Monitor gas price trends
 */
export class GasPriceMonitor {
  private prices: { price: bigint; timestamp: number }[] = [];
  private maxHistory = 100;

  addPrice(price: bigint): void {
    this.prices.push({ price, timestamp: Date.now() });
    
    if (this.prices.length > this.maxHistory) {
      this.prices.shift();
    }
  }

  getAveragePrice(minutes: number = 10): bigint {
    const cutoff = Date.now() - (minutes * 60 * 1000);
    const recentPrices = this.prices.filter(p => p.timestamp > cutoff);
    
    if (recentPrices.length === 0) {
      return 0n;
    }
    
    const sum = recentPrices.reduce((acc, p) => acc + p.price, 0n);
    return sum / BigInt(recentPrices.length);
  }

  getTrend(): 'rising' | 'falling' | 'stable' {
    if (this.prices.length < 2) {
      return 'stable';
    }
    
    const recent = this.prices.slice(-5);
    const first = recent[0].price;
    const last = recent[recent.length - 1].price;
    
    const change = Number(last - first) / Number(first);
    
    if (change > 0.1) return 'rising';
    if (change < -0.1) return 'falling';
    return 'stable';
  }
}

/**
 * Optimize gas settings based on urgency and cost tolerance
 */
export async function optimizeGasSettings(
  provider: Provider,
  urgency: 'low' | 'medium' | 'high',
  costTolerance: 'low' | 'medium' | 'high'
): Promise<Partial<GasConfig>> {
  const feeData = await provider.getFeeData();
  
  // Base multipliers
  const urgencyMultipliers = { low: 0.9, medium: 1.0, high: 1.5 };
  const costMultipliers = { low: 0.8, medium: 1.0, high: 1.3 };
  
  const urgencyMult = urgencyMultipliers[urgency];
  const costMult = costMultipliers[costTolerance];
  
  // Calculate optimal priority level
  let priorityLevel: 'slow' | 'standard' | 'fast' | 'instant';
  const combinedMult = urgencyMult * costMult;
  
  if (combinedMult <= 0.9) priorityLevel = 'slow';
  else if (combinedMult <= 1.1) priorityLevel = 'standard';
  else if (combinedMult <= 1.4) priorityLevel = 'fast';
  else priorityLevel = 'instant';
  
  return {
    priorityLevel,
    bufferMultiplier: 1.1 + (combinedMult - 1) * 0.3
  };
}
