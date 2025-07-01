// Official CCIP configuration based on Chainlink documentation
export const CCIP_CONFIG = {
  // CCIP Router addresses per chain
  ROUTERS: {
    1: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D',        // Ethereum
    43114: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',     // Avalanche
    137: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43',      // Polygon
    42161: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8',    // Arbitrum
    11155111: '0x0BF3dE8c5D3e8A2B34D2BEeB17ABfCeBaf363A59', // Sepolia
    43113: '0xF694E193200268f9a4868e4Aa017A0118C9a8177'     // Avalanche Fuji
  },

  // Official CCIP Chain Selectors
  CHAIN_SELECTORS: {
    1: 5009297550715157269n,        // Ethereum
    43114: 6433500567565415381n,    // Avalanche
    137: 4051577828743386545n,      // Polygon
    42161: 4949039107694359620n,    // Arbitrum
    11155111: 16015286601757825753n, // Sepolia
    43113: 14767482510784806043n    // Avalanche Fuji
  },

  // Supported tokens for cross-chain transfers
  SUPPORTED_TOKENS: {
    1: {
      USDC: '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8',
      LINK: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
      WETH: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
    },
    43114: {
      USDC: '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
      LINK: '0x5947BB275c521040051D82396192181b413227A3',
      WAVAX: '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7'
    },
    137: {
      USDC: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
      LINK: '0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39',
      WMATIC: '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270'
    },
    42161: {
      USDC: '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
      LINK: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4',
      WETH: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1'
    }
  },

  // Gas limits for different message types
  GAS_LIMITS: {
    TOKEN_TRANSFER: 200000,
    MESSAGE_ONLY: 500000,
    MESSAGE_WITH_TOKENS: 800000,
    COMPLEX_OPERATION: 2000000
  },

  // Fee configuration
  FEES: {
    // Base fee in native token (wei)
    BASE_FEE: {
      1: '50000000000000000',      // 0.05 ETH
      43114: '1000000000000000000', // 1 AVAX
      137: '10000000000000000000',  // 10 MATIC
      42161: '10000000000000000'    // 0.01 ETH
    },
    
    // Per-byte fee for message data
    PER_BYTE_FEE: 100, // in wei
    
    // Token transfer fee (basis points)
    TOKEN_TRANSFER_FEE: 10 // 0.1%
  },

  // Rate limits (tokens per second)
  RATE_LIMITS: {
    OUTBOUND: {
      1: '1000000000000000000000',    // 1000 tokens/sec
      43114: '500000000000000000000', // 500 tokens/sec
      137: '2000000000000000000000',  // 2000 tokens/sec
      42161: '1000000000000000000000' // 1000 tokens/sec
    },
    INBOUND: {
      1: '1000000000000000000000',    // 1000 tokens/sec
      43114: '500000000000000000000', // 500 tokens/sec
      137: '2000000000000000000000',  // 2000 tokens/sec
      42161: '1000000000000000000000' // 1000 tokens/sec
    }
  },

  // Message execution settings
  EXECUTION: {
    // Default gas limit for message execution
    DEFAULT_GAS_LIMIT: 200000,
    
    // Maximum gas limit allowed
    MAX_GAS_LIMIT: 2000000,
    
    // Confirmation requirements
    CONFIRMATIONS: {
      1: 12,     // Ethereum
      43114: 1,  // Avalanche
      137: 128,  // Polygon
      42161: 1   // Arbitrum
    }
  },

  // Timeouts for different operations (in seconds)
  TIMEOUTS: {
    MESSAGE_DELIVERY: 1800, // 30 minutes
    TOKEN_TRANSFER: 3600,   // 1 hour
    CONFIRMATION: 300       // 5 minutes
  }
};

/**
 * Get CCIP router address for chain
 */
export function getCCIPRouter(chainId: number): string {
  const router = CCIP_CONFIG.ROUTERS[chainId as keyof typeof CCIP_CONFIG.ROUTERS];
  if (!router) {
    throw new Error(`CCIP router not available for chain ${chainId}`);
  }
  return router;
}

/**
 * Get CCIP chain selector
 */
export function getCCIPChainSelector(chainId: number): bigint {
  const selector = CCIP_CONFIG.CHAIN_SELECTORS[chainId as keyof typeof CCIP_CONFIG.CHAIN_SELECTORS];
  if (!selector) {
    throw new Error(`CCIP chain selector not available for chain ${chainId}`);
  }
  return selector;
}

/**
 * Get supported tokens for chain
 */
export function getSupportedTokens(chainId: number): { [symbol: string]: string } {
  const tokens = CCIP_CONFIG.SUPPORTED_TOKENS[chainId as keyof typeof CCIP_CONFIG.SUPPORTED_TOKENS];
  if (!tokens) {
    throw new Error(`Supported tokens not available for chain ${chainId}`);
  }
  return tokens;
}

/**
 * Check if token is supported for cross-chain transfer
 */
export function isTokenSupported(chainId: number, tokenAddress: string): boolean {
  const tokens = getSupportedTokens(chainId);
  return Object.values(tokens).includes(tokenAddress.toLowerCase());
}

/**
 * Get gas limit for message type
 */
export function getGasLimitForMessageType(
  messageType: 'TOKEN_TRANSFER' | 'MESSAGE_ONLY' | 'MESSAGE_WITH_TOKENS' | 'COMPLEX_OPERATION'
): number {
  return CCIP_CONFIG.GAS_LIMITS[messageType];
}

/**
 * Calculate CCIP fee estimate
 */
export function estimateCCIPFee(
  sourceChain: number,
  messageSize: number,
  tokenTransfers: number = 0
): bigint {
  const baseFee = BigInt(CCIP_CONFIG.FEES.BASE_FEE[sourceChain as keyof typeof CCIP_CONFIG.FEES.BASE_FEE] || '50000000000000000');
  const messageFee = BigInt(messageSize * CCIP_CONFIG.FEES.PER_BYTE_FEE);
  const tokenFee = BigInt(tokenTransfers) * baseFee / 100n; // 1% per token transfer
  
  return baseFee + messageFee + tokenFee;
}

/**
 * Get confirmation requirements for chain
 */
export function getConfirmationRequirements(chainId: number): number {
  return CCIP_CONFIG.EXECUTION.CONFIRMATIONS[chainId as keyof typeof CCIP_CONFIG.EXECUTION.CONFIRMATIONS] || 12;
}

/**
 * Check if chains can communicate via CCIP
 */
export function canCommunicate(sourceChain: number, destChain: number): boolean {
  return sourceChain in CCIP_CONFIG.ROUTERS && 
         destChain in CCIP_CONFIG.ROUTERS && 
         sourceChain !== destChain;
}
