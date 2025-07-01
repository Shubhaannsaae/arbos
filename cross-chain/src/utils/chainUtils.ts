import { ethers } from 'ethers';

// Official chain configurations based on Chainlink documentation
export const SUPPORTED_CHAINS = {
  1: {
    name: 'Ethereum',
    nativeToken: 'ETH',
    decimals: 18,
    blockTime: 12,
    ccipSelector: 5009297550715157269n,
    rpcUrls: ['https://eth.llamarpc.com', 'https://ethereum.publicnode.com'],
    explorers: ['https://etherscan.io']
  },
  43114: {
    name: 'Avalanche',
    nativeToken: 'AVAX',
    decimals: 18,
    blockTime: 2,
    ccipSelector: 6433500567565415381n,
    rpcUrls: ['https://api.avax.network/ext/bc/C/rpc', 'https://avalanche.public-rpc.com'],
    explorers: ['https://snowtrace.io']
  },
  137: {
    name: 'Polygon',
    nativeToken: 'MATIC',
    decimals: 18,
    blockTime: 2,
    ccipSelector: 4051577828743386545n,
    rpcUrls: ['https://polygon-rpc.com', 'https://polygon.llamarpc.com'],
    explorers: ['https://polygonscan.com']
  },
  42161: {
    name: 'Arbitrum',
    nativeToken: 'ETH',
    decimals: 18,
    blockTime: 1,
    ccipSelector: 4949039107694359620n,
    rpcUrls: ['https://arb1.arbitrum.io/rpc', 'https://arbitrum.llamarpc.com'],
    explorers: ['https://arbiscan.io']
  }
} as const;

export type SupportedChainId = keyof typeof SUPPORTED_CHAINS;

/**
 * Check if chain ID is supported
 */
export function isChainSupported(chainId: number): chainId is SupportedChainId {
  return chainId in SUPPORTED_CHAINS;
}

/**
 * Get chain configuration
 */
export function getChainConfig(chainId: number) {
  if (!isChainSupported(chainId)) {
    throw new Error(`Chain ${chainId} is not supported`);
  }
  return SUPPORTED_CHAINS[chainId];
}

/**
 * Get chain name
 */
export function getChainName(chainId: number): string {
  return isChainSupported(chainId) ? SUPPORTED_CHAINS[chainId].name : `Chain ${chainId}`;
}

/**
 * Get native token symbol
 */
export function getNativeToken(chainId: number): string {
  return isChainSupported(chainId) ? SUPPORTED_CHAINS[chainId].nativeToken : 'ETH';
}

/**
 * Get CCIP chain selector
 */
export function getCCIPSelector(chainId: number): bigint {
  if (!isChainSupported(chainId)) {
    throw new Error(`CCIP selector not available for chain ${chainId}`);
  }
  return SUPPORTED_CHAINS[chainId].ccipSelector;
}

/**
 * Get chain ID from CCIP selector
 */
export function getChainIdFromSelector(selector: bigint): number {
  for (const [chainId, config] of Object.entries(SUPPORTED_CHAINS)) {
    if (config.ccipSelector === selector) {
      return Number(chainId);
    }
  }
  throw new Error(`Chain ID not found for CCIP selector ${selector}`);
}

/**
 * Validate Ethereum address
 */
export function isValidAddress(address: string): boolean {
  try {
    ethers.isAddress(address);
    return ethers.isAddress(address);
  } catch {
    return false;
  }
}

/**
 * Normalize address to checksum format
 */
export function normalizeAddress(address: string): string {
  if (!isValidAddress(address)) {
    throw new Error(`Invalid address: ${address}`);
  }
  return ethers.getAddress(address);
}

/**
 * Check if transaction hash is valid
 */
export function isValidTxHash(hash: string): boolean {
  return ethers.isHexString(hash, 32);
}

/**
 * Get block explorer URL for transaction
 */
export function getExplorerTxUrl(chainId: number, txHash: string): string {
  if (!isChainSupported(chainId)) {
    return `https://etherscan.io/tx/${txHash}`;
  }
  
  const explorer = SUPPORTED_CHAINS[chainId].explorers[0];
  return `${explorer}/tx/${txHash}`;
}

/**
 * Get block explorer URL for address
 */
export function getExplorerAddressUrl(chainId: number, address: string): string {
  if (!isChainSupported(chainId)) {
    return `https://etherscan.io/address/${address}`;
  }
  
  const explorer = SUPPORTED_CHAINS[chainId].explorers[0];
  return `${explorer}/address/${address}`;
}

/**
 * Convert wei to native token units
 */
export function formatNativeToken(amount: bigint, chainId: number): string {
  const config = getChainConfig(chainId);
  return `${ethers.formatEther(amount)} ${config.nativeToken}`;
}

/**
 * Parse native token amount to wei
 */
export function parseNativeToken(amount: string): bigint {
  return ethers.parseEther(amount);
}

/**
 * Get estimated confirmation blocks for chain
 */
export function getConfirmationBlocks(chainId: number): number {
  const confirmations: { [key: number]: number } = {
    1: 12,     // Ethereum
    43114: 1,  // Avalanche
    137: 128,  // Polygon
    42161: 1   // Arbitrum
  };
  return confirmations[chainId] || 12;
}

/**
 * Calculate estimated confirmation time
 */
export function getEstimatedConfirmationTime(chainId: number): number {
  if (!isChainSupported(chainId)) {
    return 180; // 3 minutes default
  }
  
  const config = SUPPORTED_CHAINS[chainId];
  const confirmationBlocks = getConfirmationBlocks(chainId);
  return config.blockTime * confirmationBlocks;
}

/**
 * Check if two chains can communicate via CCIP
 */
export function canCommunicateViaCCIP(sourceChain: number, destChain: number): boolean {
  return isChainSupported(sourceChain) && isChainSupported(destChain) && sourceChain !== destChain;
}

/**
 * Get RPC URL for chain
 */
export function getRPCUrl(chainId: number, preferredIndex: number = 0): string {
  if (!isChainSupported(chainId)) {
    throw new Error(`RPC URL not available for chain ${chainId}`);
  }
  
  const urls = SUPPORTED_CHAINS[chainId].rpcUrls;
  return urls[Math.min(preferredIndex, urls.length - 1)];
}

/**
 * Validate hex string format
 */
export function isValidHex(hex: string, expectedLength?: number): boolean {
  if (!ethers.isHexString(hex)) {
    return false;
  }
  
  if (expectedLength && ethers.getBytes(hex).length !== expectedLength) {
    return false;
  }
  
  return true;
}

/**
 * Get all supported chain IDs
 */
export function getSupportedChainIds(): number[] {
  return Object.keys(SUPPORTED_CHAINS).map(Number);
}

/**
 * Check if chain supports specific token standard
 */
export function supportsTokenStandard(chainId: number, standard: 'ERC20' | 'ERC721' | 'ERC1155'): boolean {
  // All supported chains support these standards
  return isChainSupported(chainId);
}
