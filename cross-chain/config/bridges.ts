// Bridge configurations for different protocols
export const BRIDGE_CONFIG = {
  // LayerZero configuration
  LAYERZERO: {
    ENDPOINTS: {
      1: '0x66A71Dcef29A0fFBDBE3c6a460a3B5BC225Cd675',     // Ethereum
      43114: '0x3c2269811836af69497E5F486A85D7316753cf62', // Avalanche
      137: '0x3c2269811836af69497E5F486A85D7316753cf62',   // Polygon
      42161: '0x3c2269811836af69497E5F486A85D7316753cf62', // Arbitrum
      56: '0x3c2269811836af69497E5F486A85D7316753cf62'     // BSC
    },
    
    CHAIN_IDS: {
      1: 101,     // Ethereum
      43114: 106, // Avalanche
      137: 109,   // Polygon
      42161: 110, // Arbitrum
      56: 102     // BSC
    },
    
    GAS_LIMITS: {
      SEND_MESSAGE: 200000,
      RECEIVE_MESSAGE: 150000,
      TOKEN_TRANSFER: 300000
    }
  },

  // Polygon PoS Bridge configuration
  POLYGON: {
    MAINNET: {
      ROOT_CHAIN_MANAGER: '0xA0c68C638235ee32657e8f720a23ceC1bFc77C77',
      CHECKPOINT_MANAGER: '0x86E4Dc95c7FBdBf52e33D563BbDB00823894C287',
      FX_ROOT: '0xfe5e5D361b2ad62c541bAb87C45a0B9B018389a2',
      FX_CHILD: '0x8397259c983751DAf40400790063935a11afa28a'
    },
    
    TESTNET: {
      ROOT_CHAIN_MANAGER: '0xBbD7cBFA79FaBC85B7d1C2e56007C7E4e9d55b66',
      CHECKPOINT_MANAGER: '0x2890bA17EfE978480615e330ecB65333b880928e',
      FX_ROOT: '0x3d1d3E34f7fB6D26245E6640E1c50710eFFf15bA',
      FX_CHILD: '0xCf73231F28B7331BBe3124B907840A94851f9f11'
    },
    
    SUPPORTED_TOKENS: {
      ETHEREUM: {
        USDC: '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8',
        LINK: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
        WBTC: '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
        WETH: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2'
      },
      POLYGON: {
        USDC: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        LINK: '0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39',
        WBTC: '0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6',
        WETH: '0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619'
      }
    }
  },

  // Arbitrum Bridge configuration
  ARBITRUM: {
    MAINNET: {
      L1_GATEWAY_ROUTER: '0x72Ce9c846789fdB6fC1f34aC4AD25Dd9ef7031ef',
      L2_GATEWAY_ROUTER: '0x5288c571Fd7aD117beA99bF60FE0846C4E84F933',
      INBOX: '0x4Dbd4fc535Ac27206064B68FfCf827b0A60BAB3f',
      OUTBOX: '0x0B9857ae2D4A3DBe74ffE1d7DF045bb7F96E4840'
    },
    
    TESTNET: {
      L1_GATEWAY_ROUTER: '0x70C143928eCfFaf9F5b406f7f4fC28Dc43d68380',
      L2_GATEWAY_ROUTER: '0x6c411aD3E74De3E7Bd422b94A27770f5B86C623B',
      INBOX: '0x578BAde599406A8fE3d24Fd7f7211c0911F5B29e',
      OUTBOX: '0x45Af9Ed1D03703e480CE7d328fB684bb67DA5049'
    },
    
    SUPPORTED_TOKENS: {
      ETHEREUM: {
        USDC: '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8',
        LINK: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
        USDT: '0xdAC17F958D2ee523a2206206994597C13D831ec7'
      },
      ARBITRUM: {
        USDC: '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
        LINK: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4',
        USDT: '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9'
      }
    }
  },

  // Avalanche Bridge configuration
  AVALANCHE: {
    MAINNET: {
      BRIDGE_ADDRESS: '0x8aB6a7C7f8f3f3a0c0C0a0C0a0C0a0C0a0C0a0C0'
    },
    
    TESTNET: {
      BRIDGE_ADDRESS: '0x1aB6a7C7f8f3f3a0c0C0a0C0a0C0a0C0a0C0a0C0'
    },
    
    DOMAIN_MAPPINGS: {
      1: 2,     // Ethereum
      43114: 1, // Avalanche
      137: 3,   // Polygon
      42161: 4  // Arbitrum
    }
  }
};

/**
 * Get LayerZero endpoint for chain
 */
export function getLayerZeroEndpoint(chainId: number): string {
  const endpoint = BRIDGE_CONFIG.LAYERZERO.ENDPOINTS[chainId as keyof typeof BRIDGE_CONFIG.LAYERZERO.ENDPOINTS];
  if (!endpoint) {
    throw new Error(`LayerZero endpoint not available for chain ${chainId}`);
  }
  return endpoint;
}

/**
 * Get LayerZero chain ID
 */
export function getLayerZeroChainId(chainId: number): number {
  const lzChainId = BRIDGE_CONFIG.LAYERZERO.CHAIN_IDS[chainId as keyof typeof BRIDGE_CONFIG.LAYERZERO.CHAIN_IDS];
  if (!lzChainId) {
    throw new Error(`LayerZero chain ID not available for chain ${chainId}`);
  }
  return lzChainId;
}

/**
 * Get Polygon bridge addresses
 */
export function getPolygonBridgeAddresses(isTestnet: boolean = false) {
  return isTestnet ? BRIDGE_CONFIG.POLYGON.TESTNET : BRIDGE_CONFIG.POLYGON.MAINNET;
}

/**
 * Get Arbitrum bridge addresses
 */
export function getArbitrumBridgeAddresses(isTestnet: boolean = false) {
  return isTestnet ? BRIDGE_CONFIG.ARBITRUM.TESTNET : BRIDGE_CONFIG.ARBITRUM.MAINNET;
}

/**
 * Get supported tokens for bridge protocol
 */
export function getBridgeSupportedTokens(
  protocol: 'POLYGON' | 'ARBITRUM',
  network: 'ETHEREUM' | 'POLYGON' | 'ARBITRUM'
): { [symbol: string]: string } {
  const tokens = BRIDGE_CONFIG[protocol].SUPPORTED_TOKENS[network];
  if (!tokens) {
    throw new Error(`Supported tokens not available for ${protocol} on ${network}`);
  }
  return tokens;
}

/**
 * Check if token is supported on bridge
 */
export function isBridgeTokenSupported(
  protocol: 'POLYGON' | 'ARBITRUM',
  network: 'ETHEREUM' | 'POLYGON' | 'ARBITRUM',
  tokenAddress: string
): boolean {
  try {
    const tokens = getBridgeSupportedTokens(protocol, network);
    return Object.values(tokens).includes(tokenAddress.toLowerCase());
  } catch {
    return false;
  }
}

/**
 * Get bridge protocol capabilities
 */
export function getBridgeCapabilities(protocol: string): {
  supportsTokens: boolean;
  supportsMessages: boolean;
  supportsNFTs: boolean;
  bidirectional: boolean;
} {
  const capabilities = {
    LAYERZERO: {
      supportsTokens: true,
      supportsMessages: true,
      supportsNFTs: true,
      bidirectional: true
    },
    POLYGON: {
      supportsTokens: true,
      supportsMessages: false,
      supportsNFTs: true,
      bidirectional: true
    },
    ARBITRUM: {
      supportsTokens: true,
      supportsMessages: true,
      supportsNFTs: true,
      bidirectional: true
    },
    AVALANCHE: {
      supportsTokens: true,
      supportsMessages: true,
      supportsNFTs: false,
      bidirectional: true
    }
  };

  return capabilities[protocol as keyof typeof capabilities] || {
    supportsTokens: false,
    supportsMessages: false,
    supportsNFTs: false,
    bidirectional: false
  };
}
