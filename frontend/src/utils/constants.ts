// Chainlink official contract addresses from docs.chain.link
export const CHAINLINK_CONTRACTS = {
  PRICE_FEEDS: {
    1: { // Ethereum Mainnet
      'ETH/USD': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
      'BTC/USD': '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',
      'LINK/USD': '0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c',
      'USDC/USD': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6'
    },
    43114: { // Avalanche
      'AVAX/USD': '0x0A77230d17318075983913bC2145DB16C7366156',
      'ETH/USD': '0x976B3D034E162d8bD72D6b9C989d545b839003b0',
      'BTC/USD': '0x2779D32d5166BAaa2B2b658333bA7e6Ec0C65743'
    },
    137: { // Polygon
      'MATIC/USD': '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0',
      'ETH/USD': '0xF9680D99D6C9589e2a93a78A04A279e509205945'
    },
    42161: { // Arbitrum
      'ETH/USD': '0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612',
      'BTC/USD': '0x6ce185860a4963106506C203335A2910413708e9'
    }
  },

  // CCIP Router addresses from official Chainlink docs
  CCIP_ROUTERS: {
    1: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D',
    43114: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',
    137: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43',
    42161: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8'
  },

  // VRF Coordinator addresses
  VRF_COORDINATORS: {
    1: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909',
    43114: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634',
    137: '0xAE975071Be8F8eE67addBC1A82488F1C24858067',
    42161: '0x41034678D6C633D8a95c75e1138A360a28bA15d1'
  },

  // Functions Router addresses
  FUNCTIONS_ROUTERS: {
    1: '0x65C939B26d3d949A6E2bE41B1F5659dB13b5f4a',
    137: '0x4f8a84C442F9675610c680990EdDb2CCDDB2E906',
    42161: '0x97083E831F8F0638855e2A515c90EdCF158DF238'
  }
};

// CCIP Chain Selectors from official documentation
export const CCIP_CHAIN_SELECTORS = {
  1: 5009297550715157269n,        // Ethereum
  43114: 6433500567565415381n,    // Avalanche
  137: 4051577828743386545n,      // Polygon
  42161: 4949039107694359620n     // Arbitrum
};

// Supported networks configuration
export const SUPPORTED_NETWORKS = {
  1: {
    name: 'Ethereum',
    symbol: 'ETH',
    decimals: 18,
    rpcUrl: 'https://eth.llamarpc.com',
    blockExplorer: 'https://etherscan.io',
    faucet: null
  },
  43114: {
    name: 'Avalanche',
    symbol: 'AVAX',
    decimals: 18,
    rpcUrl: 'https://api.avax.network/ext/bc/C/rpc',
    blockExplorer: 'https://snowtrace.io',
    faucet: null
  },
  137: {
    name: 'Polygon',
    symbol: 'MATIC',
    decimals: 18,
    rpcUrl: 'https://polygon-rpc.com',
    blockExplorer: 'https://polygonscan.com',
    faucet: null
  },
  42161: {
    name: 'Arbitrum',
    symbol: 'ETH',
    decimals: 18,
    rpcUrl: 'https://arb1.arbitrum.io/rpc',
    blockExplorer: 'https://arbiscan.io',
    faucet: null
  }
} as const;

// Token addresses for multi-chain support
export const TOKEN_ADDRESSES = {
  1: { // Ethereum
    USDC: '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8',
    USDT: '0xdAC17F958D2ee523a2206206994597C13D831ec7',
    LINK: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
    WETH: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
    WBTC: '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'
  },
  43114: { // Avalanche
    USDC: '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
    USDT: '0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7',
    LINK: '0x5947BB275c521040051D82396192181b413227A3',
    WAVAX: '0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7'
  },
  137: { // Polygon
    USDC: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
    USDT: '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',
    LINK: '0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39',
    WMATIC: '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270'
  },
  42161: { // Arbitrum
    USDC: '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
    USDT: '0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9',
    LINK: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4',
    WETH: '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1'
  }
} as const;

// Gas limits for different operations
export const GAS_LIMITS = {
  SIMPLE_TRANSFER: 21000,
  TOKEN_TRANSFER: 65000,
  SWAP: 150000,
  ARBITRAGE: 300000,
  CCIP_SEND: 500000,
  VRF_REQUEST: 200000,
  FUNCTION_CALL: 100000
} as const;

// API endpoints
export const API_ENDPOINTS = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'https://api.arbos.io',
  ARBITRAGE: '/arbitrage',
  PORTFOLIO: '/portfolio',
  AGENTS: '/agents',
  PRICES: '/prices',
  ANALYTICS: '/analytics'
} as const;

// Application configuration
export const APP_CONFIG = {
  NAME: 'ArbOS',
  VERSION: '1.0.0',
  DESCRIPTION: 'AI-Powered Cross-Chain Trading Platform with Chainlink Integration',
  DEFAULT_CHAIN_ID: 1,
  POLLING_INTERVAL: 30000, // 30 seconds
  CACHE_DURATION: 300000,  // 5 minutes
  MAX_SLIPPAGE: 500,       // 5%
  DEFAULT_GAS_PRICE_GWEI: 20
} as const;

// Time constants
export const TIME_CONSTANTS = {
  MINUTE: 60 * 1000,
  HOUR: 60 * 60 * 1000,
  DAY: 24 * 60 * 60 * 1000,
  WEEK: 7 * 24 * 60 * 60 * 1000,
  MONTH: 30 * 24 * 60 * 60 * 1000
} as const;

// Local storage keys
export const STORAGE_KEYS = {
  WALLET_CONNECTED: 'wallet_connected',
  SELECTED_CHAIN: 'selected_chain',
  USER_PREFERENCES: 'user_preferences',
  AGENT_CONFIGS: 'agent_configs',
  PORTFOLIO_SETTINGS: 'portfolio_settings'
} as const;
