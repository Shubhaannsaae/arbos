import { ChainInfo } from '../types/blockchain';

export const SUPPORTED_NETWORKS: Record<number, ChainInfo> = {
  1: {
    chainId: 1,
    name: 'Ethereum Mainnet',
    shortName: 'eth',
    network: 'mainnet',
    networkId: 1,
    nativeCurrency: {
      name: 'Ethereum',
      symbol: 'ETH',
      decimals: 18
    },
    rpc: [
      'https://mainnet.infura.io/v3/${INFURA_API_KEY}',
      'https://eth-mainnet.alchemyapi.io/v2/${ALCHEMY_API_KEY}',
      'https://cloudflare-eth.com'
    ],
    faucets: [],
    explorers: [
      {
        name: 'Etherscan',
        url: 'https://etherscan.io',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://ethereum.org'
  },

  137: {
    chainId: 137,
    name: 'Polygon Mainnet',
    shortName: 'matic',
    network: 'matic',
    networkId: 137,
    nativeCurrency: {
      name: 'MATIC',
      symbol: 'MATIC',
      decimals: 18
    },
    rpc: [
      'https://polygon-rpc.com',
      'https://rpc-mainnet.matic.network',
      'https://matic-mainnet.chainstacklabs.com'
    ],
    faucets: [],
    explorers: [
      {
        name: 'Polygonscan',
        url: 'https://polygonscan.com',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://polygon.technology/',
    parent: {
      type: 'L2',
      chain: 'eip155-1',
      bridges: [
        {
          url: 'https://wallet.polygon.technology/bridge'
        }
      ]
    }
  },

  42161: {
    chainId: 42161,
    name: 'Arbitrum One',
    shortName: 'arb1',
    network: 'arbitrum',
    networkId: 42161,
    nativeCurrency: {
      name: 'Ethereum',
      symbol: 'ETH',
      decimals: 18
    },
    rpc: [
      'https://arb1.arbitrum.io/rpc',
      'https://arbitrum-mainnet.infura.io/v3/${INFURA_API_KEY}'
    ],
    faucets: [],
    explorers: [
      {
        name: 'Arbiscan',
        url: 'https://arbiscan.io',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://arbitrum.io',
    parent: {
      type: 'L2',
      chain: 'eip155-1',
      bridges: [
        {
          url: 'https://bridge.arbitrum.io'
        }
      ]
    }
  },

  43114: {
    chainId: 43114,
    name: 'Avalanche C-Chain',
    shortName: 'avax',
    network: 'avalanche',
    networkId: 43114,
    nativeCurrency: {
      name: 'Avalanche',
      symbol: 'AVAX',
      decimals: 18
    },
    rpc: [
      'https://api.avax.network/ext/bc/C/rpc',
      'https://avalanche-mainnet.infura.io/v3/${INFURA_API_KEY}'
    ],
    faucets: [],
    explorers: [
      {
        name: 'Snowtrace',
        url: 'https://snowtrace.io',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://www.avax.network/'
  },

  56: {
    chainId: 56,
    name: 'BNB Smart Chain Mainnet',
    shortName: 'bnb',
    network: 'bsc',
    networkId: 56,
    nativeCurrency: {
      name: 'BNB Chain Native Token',
      symbol: 'BNB',
      decimals: 18
    },
    rpc: [
      'https://bsc-dataseed.binance.org',
      'https://bsc-dataseed1.defibit.io',
      'https://bsc-dataseed1.ninicoin.io'
    ],
    faucets: [],
    explorers: [
      {
        name: 'BscScan',
        url: 'https://bscscan.com',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://www.bnbchain.world'
  },

  // Testnets
  11155111: {
    chainId: 11155111,
    name: 'Sepolia',
    shortName: 'sep',
    network: 'sepolia',
    networkId: 11155111,
    nativeCurrency: {
      name: 'Sepolia Ethereum',
      symbol: 'SEP',
      decimals: 18
    },
    rpc: [
      'https://sepolia.infura.io/v3/${INFURA_API_KEY}',
      'https://eth-sepolia.public.blastapi.io'
    ],
    faucets: [
      'https://sepoliafaucet.com',
      'https://faucet.sepolia.dev'
    ],
    explorers: [
      {
        name: 'Etherscan',
        url: 'https://sepolia.etherscan.io',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://sepolia.otterscan.io'
  },

  80001: {
    chainId: 80001,
    name: 'Mumbai',
    shortName: 'maticmum',
    network: 'mumbai',
    networkId: 80001,
    nativeCurrency: {
      name: 'MATIC',
      symbol: 'MATIC',
      decimals: 18
    },
    rpc: [
      'https://rpc-mumbai.maticvigil.com',
      'https://polygon-mumbai.infura.io/v3/${INFURA_API_KEY}'
    ],
    faucets: [
      'https://faucet.polygon.technology'
    ],
    explorers: [
      {
        name: 'Polygonscan',
        url: 'https://mumbai.polygonscan.com',
        standard: 'EIP3091'
      }
    ],
    infoURL: 'https://polygon.technology/',
    parent: {
      type: 'L2',
      chain: 'eip155-5',
      bridges: [
        {
          url: 'https://wallet.polygon.technology/bridge'
        }
      ]
    }
  }
};

export const CHAINLINK_CCIP_CHAIN_SELECTORS: Record<number, string> = {
  1: '5009297550715157269',        // Ethereum
  137: '4051577828743386545',      // Polygon
  42161: '4949039107694359620',    // Arbitrum
  43114: '6433500567565415381',    // Avalanche
  56: '11344663589394136015',      // BSC
  11155111: '16015286601757825753', // Sepolia
  80001: '12532609583862916517'    // Mumbai
};

export const NETWORK_FEES: Record<number, {
  baseFee: number;        // Base gas fee in gwei
  priorityFee: number;    // Priority fee in gwei
  bridgeFee: number;      // Bridge fee percentage
  dexFee: number;         // Average DEX fee percentage
}> = {
  1: {
    baseFee: 20,
    priorityFee: 2,
    bridgeFee: 0.05,
    dexFee: 0.3
  },
  137: {
    baseFee: 30,
    priorityFee: 30,
    bridgeFee: 0.01,
    dexFee: 0.3
  },
  42161: {
    baseFee: 0.1,
    priorityFee: 0.01,
    bridgeFee: 0.02,
    dexFee: 0.3
  },
  43114: {
    baseFee: 25,
    priorityFee: 1,
    bridgeFee: 0.02,
    dexFee: 0.3
  },
  56: {
    baseFee: 3,
    priorityFee: 1,
    bridgeFee: 0.02,
    dexFee: 0.25
  }
};

export const NETWORK_BLOCK_TIMES: Record<number, number> = {
  1: 12,      // 12 seconds
  137: 2,     // 2 seconds
  42161: 0.25, // 250ms
  43114: 2,   // 2 seconds
  56: 3       // 3 seconds
};

export const NETWORK_CONFIRMATION_BLOCKS: Record<number, number> = {
  1: 12,      // ~2.4 minutes
  137: 30,    // ~1 minute
  42161: 1,   // Instant finality
  43114: 1,   // Instant finality
  56: 20      // ~1 minute
};

export function getNetworkInfo(chainId: number): ChainInfo {
  const network = SUPPORTED_NETWORKS[chainId];
  if (!network) {
    throw new Error(`Unsupported network: ${chainId}`);
  }
  return network;
}

export function getCCIPSelector(chainId: number): string {
  const selector = CHAINLINK_CCIP_CHAIN_SELECTORS[chainId];
  if (!selector) {
    throw new Error(`CCIP not supported on chain: ${chainId}`);
  }
  return selector;
}

export function isTestnet(chainId: number): boolean {
  return [11155111, 80001, 421613, 43113].includes(chainId);
}

export function isMainnet(chainId: number): boolean {
  return [1, 137, 42161, 43114, 56, 250, 10].includes(chainId);
}

export function getNetworkFees(chainId: number) {
  return NETWORK_FEES[chainId] || NETWORK_FEES[1];
}

export function getBlockTime(chainId: number): number {
  return NETWORK_BLOCK_TIMES[chainId] || 12;
}

export function getConfirmationBlocks(chainId: number): number {
  return NETWORK_CONFIRMATION_BLOCKS[chainId] || 12;
}
