import { ethers } from 'ethers';

export interface NetworkConfig {
  chainId: number;
  name: string;
  rpcUrl: string;
  nativeCurrency: {
    name: string;
    symbol: string;
    decimals: number;
  };
  blockExplorer: string;
  chainlinkContracts: {
    priceFeeds: Record<string, string>;
    automationRegistry?: string;
    functionsRouter?: string;
    vrfCoordinator?: string;
    ccipRouter?: string;
    linkToken?: string;
  };
  dexContracts: {
    uniswapV3Factory?: string;
    uniswapV3Router?: string;
    sushiswapRouter?: string;
    curveRegistry?: string;
  };
}

export const NETWORK_CONFIGS: Record<number, NetworkConfig> = {
  1: { // Ethereum Mainnet
    chainId: 1,
    name: 'Ethereum',
    rpcUrl: process.env.ETHEREUM_RPC_URL!,
    nativeCurrency: { name: 'Ethereum', symbol: 'ETH', decimals: 18 },
    blockExplorer: 'https://etherscan.io',
    chainlinkContracts: {
      priceFeeds: {
        'ETH/USD': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
        'BTC/USD': '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',
        'LINK/USD': '0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c',
        'USDC/USD': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6',
        'USDT/USD': '0x3E7d1eAB13ad0104d2750B8863b489D65364e32D'
      },
      automationRegistry: '0xE02Ed3110c78c8F79eABDF04b9d4df8d28C7D5e0',
      functionsRouter: '0x65C017B4a7F5f4d3E7dE5C9dD4e6f8e6F5e8a1D2',
      vrfCoordinator: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909',
      ccipRouter: '0xE561d5E02207fb5eB32cca20a699E0d8919a1476',
      linkToken: '0x514910771AF9Ca656af840dff83E8264EcF986CA'
    },
    dexContracts: {
      uniswapV3Factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984',
      uniswapV3Router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
      sushiswapRouter: '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
      curveRegistry: '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5'
    }
  },

  137: { // Polygon
    chainId: 137,
    name: 'Polygon',
    rpcUrl: process.env.POLYGON_RPC_URL!,
    nativeCurrency: { name: 'Polygon', symbol: 'MATIC', decimals: 18 },
    blockExplorer: 'https://polygonscan.com',
    chainlinkContracts: {
      priceFeeds: {
        'MATIC/USD': '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0',
        'ETH/USD': '0xF9680D99D6C9589e2a93a78A04A279e509205945',
        'BTC/USD': '0xc907E116054Ad103354f2D350FD2514433D57F6f',
        'LINK/USD': '0xd9FFdb71EbE7496cC440152d43986Aae0AB76665',
        'USDC/USD': '0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7'
      },
      automationRegistry: '0xE02Ed3110c78c8F79eABDF04b9d4df8d28C7D5e0',
      functionsRouter: '0xdc2AAF042Aeff2E68B3e8E33F19e4B9fA7C73F10',
      vrfCoordinator: '0xAE975071Be8F8eE67addBC1A82488F1C24858067',
      ccipRouter: '0x849c5ED5a80F5B408Dd4969b78c2C8fdf0565Bfe',
      linkToken: '0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39'
    },
    dexContracts: {
      uniswapV3Factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984',
      uniswapV3Router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
      sushiswapRouter: '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
    }
  },

  42161: { // Arbitrum One
    chainId: 42161,
    name: 'Arbitrum',
    rpcUrl: process.env.ARBITRUM_RPC_URL!,
    nativeCurrency: { name: 'Ethereum', symbol: 'ETH', decimals: 18 },
    blockExplorer: 'https://arbiscan.io',
    chainlinkContracts: {
      priceFeeds: {
        'ETH/USD': '0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612',
        'BTC/USD': '0x6ce185860a4963106506C203335A2910413708e9',
        'LINK/USD': '0x86E53CF1B870786351Da77A57575e79CB55812CB',
        'USDC/USD': '0x50834F3163758fcC1Df9973b6e91f0F0F0434aD3',
        'USDT/USD': '0x3f3f5dF88dC9F13eac63DF89EC16ef6e7E25DdE7'
      },
      automationRegistry: '0xE02Ed3110c78c8F79eABDF04b9d4df8d28C7D5e0',
      functionsRouter: '0x97083E831F8F0638855e2A515c90EdCF158DF92a',
      vrfCoordinator: '0x41034678D6C633D8a95c75e1138A360a28bA15d1',
      ccipRouter: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8',
      linkToken: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4'
    },
    dexContracts: {
      uniswapV3Factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984',
      uniswapV3Router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
      sushiswapRouter: '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
    }
  },

  43114: { // Avalanche
    chainId: 43114,
    name: 'Avalanche',
    rpcUrl: process.env.AVALANCHE_RPC_URL!,
    nativeCurrency: { name: 'Avalanche', symbol: 'AVAX', decimals: 18 },
    blockExplorer: 'https://snowtrace.io',
    chainlinkContracts: {
      priceFeeds: {
        'AVAX/USD': '0x0A77230d17318075983913bC2145DB16C7366156',
        'ETH/USD': '0x976B3D034E162d8bD72D6b9C989d545b839003b0',
        'BTC/USD': '0x2779D32d5166BAaa2B2b658333bA7e6Ec0C65743',
        'LINK/USD': '0x49ccd9ca821EfEab2b98c60dC60F518E765EDe9a',
        'USDC/USD': '0xF096872672F44d6EBA71458D74fe67F9a77a23B9'
      },
      automationRegistry: '0xE02Ed3110c78c8F79eABDF04b9d4df8d28C7D5e0',
      functionsRouter: '0x9d82A75235C8bE9CDD8A9230aF72f8c34B2E8D5A',
      vrfCoordinator: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634',
      ccipRouter: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',
      linkToken: '0x5947BB275c521040051D82396192181b413227A3'
    },
    dexContracts: {
      uniswapV3Factory: '0x740b1c1de25031C31FF4fC9A62f554A55cdC1baD',
      sushiswapRouter: '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
    }
  }
};

export interface AgentConfig {
  id: string;
  type: 'arbitrage' | 'portfolio' | 'yield' | 'security' | 'orchestrator';
  name: string;
  description: string;
  enabled: boolean;
  networks: number[];
  executionInterval: number; // milliseconds
  riskTolerance: 'low' | 'medium' | 'high';
  maxGasPrice: string; // in gwei
  constraints: {
    maxDailyTransactions: number;
    maxDailyVolume: string; // in USD
    maxPositionSize: string; // in USD
    minProfitThreshold: number; // percentage
    maxSlippage: number; // percentage
  };
  chainlinkServices: {
    dataFeeds: boolean;
    automation: boolean;
    functions: boolean;
    vrf: boolean;
    ccip: boolean;
  };
  bedrockModel: {
    modelId: string;
    maxTokens: number;
    temperature: number;
  };
}

export const AGENT_CONFIGS: Record<string, AgentConfig> = {
  arbitrage: {
    id: 'arbitrage-agent-001',
    type: 'arbitrage',
    name: 'Cross-Chain Arbitrage Agent',
    description: 'Detects and executes arbitrage opportunities across DEXes and chains',
    enabled: true,
    networks: [1, 137, 42161, 43114],
    executionInterval: 30000, // 30 seconds
    riskTolerance: 'medium',
    maxGasPrice: '50', // 50 gwei
    constraints: {
      maxDailyTransactions: 50,
      maxDailyVolume: '100000', // $100k
      maxPositionSize: '10000', // $10k
      minProfitThreshold: 0.5, // 0.5%
      maxSlippage: 1.0 // 1%
    },
    chainlinkServices: {
      dataFeeds: true,
      automation: true,
      functions: true,
      vrf: false,
      ccip: true
    },
    bedrockModel: {
      modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxTokens: 4096,
      temperature: 0.1
    }
  },

  portfolio: {
    id: 'portfolio-agent-001',
    type: 'portfolio',
    name: 'Portfolio Optimization Agent',
    description: 'Manages and optimizes portfolio allocations using modern portfolio theory',
    enabled: true,
    networks: [1, 137, 42161, 43114],
    executionInterval: 300000, // 5 minutes
    riskTolerance: 'medium',
    maxGasPrice: '40',
    constraints: {
      maxDailyTransactions: 20,
      maxDailyVolume: '500000', // $500k
      maxPositionSize: '50000', // $50k
      minProfitThreshold: 0.1, // 0.1%
      maxSlippage: 0.5 // 0.5%
    },
    chainlinkServices: {
      dataFeeds: true,
      automation: true,
      functions: true,
      vrf: true,
      ccip: true
    },
    bedrockModel: {
      modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxTokens: 4096,
      temperature: 0.2
    }
  },

  yield: {
    id: 'yield-agent-001',
    type: 'yield',
    name: 'Yield Farming Optimization Agent',
    description: 'Finds and optimizes yield farming opportunities across protocols',
    enabled: true,
    networks: [1, 137, 42161, 43114],
    executionInterval: 600000, // 10 minutes
    riskTolerance: 'high',
    maxGasPrice: '35',
    constraints: {
      maxDailyTransactions: 15,
      maxDailyVolume: '250000', // $250k
      maxPositionSize: '25000', // $25k
      minProfitThreshold: 1.0, // 1%
      maxSlippage: 1.5 // 1.5%
    },
    chainlinkServices: {
      dataFeeds: true,
      automation: true,
      functions: true,
      vrf: false,
      ccip: true
    },
    bedrockModel: {
      modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxTokens: 4096,
      temperature: 0.15
    }
  },

  security: {
    id: 'security-agent-001',
    type: 'security',
    name: 'Security Monitoring Agent',
    description: 'Monitors transactions and detects security threats and anomalies',
    enabled: true,
    networks: [1, 137, 42161, 43114],
    executionInterval: 10000, // 10 seconds
    riskTolerance: 'low',
    maxGasPrice: '100', // Higher for emergency responses
    constraints: {
      maxDailyTransactions: 100,
      maxDailyVolume: '1000000', // $1M for emergency actions
      maxPositionSize: '100000', // $100k
      minProfitThreshold: 0, // Not profit-focused
      maxSlippage: 5.0 // 5% for emergency exits
    },
    chainlinkServices: {
      dataFeeds: true,
      automation: true,
      functions: true,
      vrf: false,
      ccip: false
    },
    bedrockModel: {
      modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxTokens: 4096,
      temperature: 0.05
    }
  },

  orchestrator: {
    id: 'orchestrator-agent-001',
    type: 'orchestrator',
    name: 'System Orchestrator Agent',
    description: 'Coordinates all agents and manages system-wide operations',
    enabled: true,
    networks: [1, 137, 42161, 43114],
    executionInterval: 60000, // 1 minute
    riskTolerance: 'low',
    maxGasPrice: '30',
    constraints: {
      maxDailyTransactions: 10,
      maxDailyVolume: '10000', // $10k for coordination
      maxPositionSize: '5000', // $5k
      minProfitThreshold: 0, // Coordination-focused
      maxSlippage: 0.1 // 0.1%
    },
    chainlinkServices: {
      dataFeeds: true,
      automation: true,
      functions: true,
      vrf: true,
      ccip: true
    },
    bedrockModel: {
      modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
      maxTokens: 4096,
      temperature: 0.3
    }
  }
};

export function getNetworkConfig(chainId: number): NetworkConfig {
  const config = NETWORK_CONFIGS[chainId];
  if (!config) {
    throw new Error(`Network configuration not found for chain ID: ${chainId}`);
  }
  return config;
}

export function getAgentConfig(agentType: string): AgentConfig {
  const config = AGENT_CONFIGS[agentType];
  if (!config) {
    throw new Error(`Agent configuration not found for type: ${agentType}`);
  }
  return config;
}

export function getProvider(chainId: number): ethers.JsonRpcProvider {
  const config = getNetworkConfig(chainId);
  return new ethers.JsonRpcProvider(config.rpcUrl);
}

export function getWallet(chainId: number, privateKey?: string): ethers.Wallet {
  const provider = getProvider(chainId);
  const key = privateKey || process.env.AGENT_PRIVATE_KEY!;
  return new ethers.Wallet(key, provider);
}
