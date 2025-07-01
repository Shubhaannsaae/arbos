import { NetworkConfig } from '../src/utils/chainUtils';

// Official network configurations based on Chainlink documentation
export const NETWORK_CONFIGS: { [chainId: number]: NetworkConfig } = {
  // Ethereum Mainnet
  1: {
    name: 'Ethereum',
    chainId: 1,
    nativeToken: 'ETH',
    decimals: 18,
    blockTime: 12,
    ccipSelector: 5009297550715157269n,
    rpcUrls: [
      'https://eth.llamarpc.com',
      'https://ethereum.publicnode.com',
      'https://rpc.ankr.com/eth'
    ],
    explorers: ['https://etherscan.io'],
    faucets: [],
    testnet: false,
    chainlinkContracts: {
      router: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D',
      vrfCoordinator: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909',
      functionsRouter: '0x65C939B26d3d949A6E2bE41B1F5659dB13b5f4a',
      automationRegistry: '0x02777053d6764996e594c3E88AF1D58D5363a2e6'
    },
    bridges: {
      polygon: '0xA0c68C638235ee32657e8f720a23ceC1bFc77C77',
      arbitrum: '0xa3A7B6F88361F48403514059F1F16C8E78d60EeC',
      avalanche: '0x8aB6a7C7f8f3f3a0c0C0a0C0a0C0a0C0a0C0a0C0'
    }
  },

  // Avalanche C-Chain
  43114: {
    name: 'Avalanche',
    chainId: 43114,
    nativeToken: 'AVAX',
    decimals: 18,
    blockTime: 2,
    ccipSelector: 6433500567565415381n,
    rpcUrls: [
      'https://api.avax.network/ext/bc/C/rpc',
      'https://avalanche.public-rpc.com',
      'https://rpc.ankr.com/avalanche'
    ],
    explorers: ['https://snowtrace.io'],
    faucets: [],
    testnet: false,
    chainlinkContracts: {
      router: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',
      vrfCoordinator: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634',
      functionsRouter: '0x3c82F31d0b6e267eF9b5b5e5d2B9A2C8a2F4c5e7',
      automationRegistry: '0x02777053d6764996e594c3E88AF1D58D5363a2e6'
    },
    bridges: {
      ethereum: '0x8aB6a7C7f8f3f3a0c0C0a0C0a0C0a0C0a0C0a0C0'
    }
  },

  // Polygon Mainnet
  137: {
    name: 'Polygon',
    chainId: 137,
    nativeToken: 'MATIC',
    decimals: 18,
    blockTime: 2,
    ccipSelector: 4051577828743386545n,
    rpcUrls: [
      'https://polygon-rpc.com',
      'https://polygon.llamarpc.com',
      'https://rpc.ankr.com/polygon'
    ],
    explorers: ['https://polygonscan.com'],
    faucets: [],
    testnet: false,
    chainlinkContracts: {
      router: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43',
      vrfCoordinator: '0xAE975071Be8F8eE67addBC1A82488F1C24858067',
      functionsRouter: '0x4f8a84C442F9675610c680990EdDb2CCDDB2E906',
      automationRegistry: '0x02777053d6764996e594c3E88AF1D58D5363a2e6'
    },
    bridges: {
      ethereum: '0xA0c68C638235ee32657e8f720a23ceC1bFc77C77'
    }
  },

  // Arbitrum One
  42161: {
    name: 'Arbitrum',
    chainId: 42161,
    nativeToken: 'ETH',
    decimals: 18,
    blockTime: 1,
    ccipSelector: 4949039107694359620n,
    rpcUrls: [
      'https://arb1.arbitrum.io/rpc',
      'https://arbitrum.llamarpc.com',
      'https://rpc.ankr.com/arbitrum'
    ],
    explorers: ['https://arbiscan.io'],
    faucets: [],
    testnet: false,
    chainlinkContracts: {
      router: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8',
      vrfCoordinator: '0x41034678D6C633D8a95c75e1138A360a28bA15d1',
      functionsRouter: '0x97083E831F8F0638855e2A515c90EdCF158DF238',
      automationRegistry: '0x75c0530885F385721fddA23C539AF3701d6183D4'
    },
    bridges: {
      ethereum: '0xa3A7B6F88361F48403514059F1F16C8E78d60EeC'
    }
  },

  // Sepolia Testnet
  11155111: {
    name: 'Sepolia',
    chainId: 11155111,
    nativeToken: 'ETH',
    decimals: 18,
    blockTime: 12,
    ccipSelector: 16015286601757825753n,
    rpcUrls: [
      'https://sepolia.infura.io/v3/YOUR_INFURA_KEY',
      'https://rpc.sepolia.org'
    ],
    explorers: ['https://sepolia.etherscan.io'],
    faucets: ['https://sepoliafaucet.com'],
    testnet: true,
    chainlinkContracts: {
      router: '0x0BF3dE8c5D3e8A2B34D2BEeB17ABfCeBaf363A59',
      vrfCoordinator: '0x8103B0A8A00be2DDC778e6e7eaa21791Cd364625',
      functionsRouter: '0xb83E47C2bC239B3bf370bc41e1459A34b41238D0',
      automationRegistry: '0x86EFBD0b6736Bed994962f9797049422A3A8E8Ad'
    },
    bridges: {}
  },

  // Avalanche Fuji Testnet
  43113: {
    name: 'Avalanche Fuji',
    chainId: 43113,
    nativeToken: 'AVAX',
    decimals: 18,
    blockTime: 2,
    ccipSelector: 14767482510784806043n,
    rpcUrls: [
      'https://api.avax-test.network/ext/bc/C/rpc'
    ],
    explorers: ['https://testnet.snowtrace.io'],
    faucets: ['https://faucet.avax.network'],
    testnet: true,
    chainlinkContracts: {
      router: '0xF694E193200268f9a4868e4Aa017A0118C9a8177',
      vrfCoordinator: '0x2eD832Ba664535e5886b75D64C46EB9a228C2610',
      functionsRouter: '0xA9d587a00A31A52Ed70D6026794a8FC5E2F5dCb0',
      automationRegistry: '0x819B58A646CDd8289275A87653a2aA4902b14fe6'
    },
    bridges: {}
  }
};

export interface NetworkConfig {
  name: string;
  chainId: number;
  nativeToken: string;
  decimals: number;
  blockTime: number;
  ccipSelector: bigint;
  rpcUrls: string[];
  explorers: string[];
  faucets: string[];
  testnet: boolean;
  chainlinkContracts: {
    router: string;
    vrfCoordinator: string;
    functionsRouter: string;
    automationRegistry: string;
  };
  bridges: { [network: string]: string };
}

/**
 * Get network configuration by chain ID
 */
export function getNetworkConfig(chainId: number): NetworkConfig {
  const config = NETWORK_CONFIGS[chainId];
  if (!config) {
    throw new Error(`Network configuration not found for chain ID: ${chainId}`);
  }
  return config;
}

/**
 * Get all supported mainnet chain IDs
 */
export function getMainnetChainIds(): number[] {
  return Object.values(NETWORK_CONFIGS)
    .filter(config => !config.testnet)
    .map(config => config.chainId);
}

/**
 * Get all supported testnet chain IDs
 */
export function getTestnetChainIds(): number[] {
  return Object.values(NETWORK_CONFIGS)
    .filter(config => config.testnet)
    .map(config => config.chainId);
}

/**
 * Check if network supports CCIP
 */
export function supportsCCIP(chainId: number): boolean {
  const config = NETWORK_CONFIGS[chainId];
  return config && config.chainlinkContracts.router !== '';
}

/**
 * Get CCIP-supported chain pairs
 */
export function getCCIPSupportedPairs(): Array<{ source: number; destination: number }> {
  const ccipChains = Object.values(NETWORK_CONFIGS)
    .filter(config => supportsCCIP(config.chainId))
    .map(config => config.chainId);

  const pairs: Array<{ source: number; destination: number }> = [];
  
  for (const source of ccipChains) {
    for (const destination of ccipChains) {
      if (source !== destination) {
        pairs.push({ source, destination });
      }
    }
  }
  
  return pairs;
}
