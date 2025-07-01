export interface DeFiProtocolInfo {
  id: string;
  name: string;
  category: string[];
  tvlRank?: number;
  website: string;
  documentation: string;
  logo: string;
  chains: number[];
  contracts: Record<number, {
    router?: string;
    factory?: string;
    registry?: string;
    staking?: string;
    governance?: string;
  }>;
  fees: {
    trading: number;
    withdrawal?: number;
    performance?: number;
  };
  risks: {
    smartContract: number;
    governance: number;
    oracle: number;
    overall: number;
  };
  audits: Array<{
    auditor: string;
    date: string;
    url: string;
    score?: number;
  }>;
}

export const DEFI_PROTOCOLS: Record<string, DeFiProtocolInfo> = {
  uniswap_v3: {
    id: 'uniswap_v3',
    name: 'Uniswap V3',
    category: ['dex', 'amm'],
    tvlRank: 1,
    website: 'https://uniswap.org',
    documentation: 'https://docs.uniswap.org',
    logo: 'https://assets.coingecko.com/coins/images/12504/small/uniswap-uni.png',
    chains: [1, 137, 42161, 43114, 56],
    contracts: {
      1: {
        router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984'
      },
      137: {
        router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984'
      },
      42161: {
        router: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        factory: '0x1F98431c8aD98523631AE4a59f267346ea31F984'
      }
    },
    fees: {
      trading: 0.3
    },
    risks: {
      smartContract: 15,
      governance: 10,
      oracle: 5,
      overall: 12
    },
    audits: [
      {
        auditor: 'Trail of Bits',
        date: '2021-03-01',
        url: 'https://github.com/Uniswap/v3-core/blob/main/audits/tob/audit.pdf',
        score: 95
      }
    ]
  },

  sushiswap: {
    id: 'sushiswap',
    name: 'SushiSwap',
    category: ['dex', 'amm', 'yield'],
    tvlRank: 3,
    website: 'https://sushi.com',
    documentation: 'https://docs.sushi.com',
    logo: 'https://assets.coingecko.com/coins/images/12271/small/512x512_Logo_no_chop.png',
    chains: [1, 137, 42161, 43114, 56],
    contracts: {
      1: {
        router: '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',
        factory: '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'
      },
      137: {
        router: '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
        factory: '0xc35DADB65012eC5796536bD9864eD8773aBc74C4'
      },
      42161: {
        router: '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506',
        factory: '0xc35DADB65012eC5796536bD9864eD8773aBc74C4'
      }
    },
    fees: {
      trading: 0.3,
      performance: 2.0
    },
    risks: {
      smartContract: 20,
      governance: 25,
      oracle: 10,
      overall: 20
    },
    audits: [
      {
        auditor: 'PeckShield',
        date: '2020-11-15',
        url: 'https://github.com/sushiswap/sushiswap/blob/master/audit/PeckShield-Audit-Report-SushiSwap-v1.0.pdf'
      }
    ]
  },

  curve: {
    id: 'curve',
    name: 'Curve Finance',
    category: ['dex', 'amm', 'stablecoin'],
    tvlRank: 2,
    website: 'https://curve.fi',
    documentation: 'https://curve.readthedocs.io',
    logo: 'https://assets.coingecko.com/coins/images/12124/small/Curve.png',
    chains: [1, 137, 42161, 43114],
    contracts: {
      1: {
        registry: '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5',
        factory: '0x0959158b6040D32d04c301A72CBFD6b39E21c9AE'
      },
      137: {
        registry: '0x094d12e5b541784701FD8d65F11fc0598FBC6332',
        factory: '0x722272D36ef0Da72FF51c5A65Db7b870E2e8D4ee'
      }
    },
    fees: {
      trading: 0.04,
      withdrawal: 0.02
    },
    risks: {
      smartContract: 18,
      governance: 20,
      oracle: 8,
      overall: 17
    },
    audits: [
      {
        auditor: 'MixBytes',
        date: '2020-08-01',
        url: 'https://github.com/curvefi/curve-contract/blob/master/audits/mixbytes.pdf'
      }
    ]
  },

  aave: {
    id: 'aave',
    name: 'Aave',
    category: ['lending', 'borrowing'],
    tvlRank: 4,
    website: 'https://aave.com',
    documentation: 'https://docs.aave.com',
    logo: 'https://assets.coingecko.com/coins/images/12645/small/AAVE.png',
    chains: [1, 137, 42161, 43114],
    contracts: {
      1: {
        registry: '0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9'
      },
      137: {
        registry: '0x8dFf5E27EA6b7AC08EbFdf9eB090F32ee9a30fcf'
      }
    },
    fees: {
      trading: 0,
      withdrawal: 0.09
    },
    risks: {
      smartContract: 15,
      governance: 15,
      oracle: 20,
      overall: 17
    },
    audits: [
      {
        auditor: 'OpenZeppelin',
        date: '2020-10-01',
        url: 'https://blog.openzeppelin.com/aave-protocol-audit/'
      }
    ]
  },

  compound: {
    id: 'compound',
    name: 'Compound',
    category: ['lending', 'borrowing'],
    tvlRank: 8,
    website: 'https://compound.finance',
    documentation: 'https://docs.compound.finance',
    logo: 'https://assets.coingecko.com/coins/images/10775/small/COMP.png',
    chains: [1, 137],
    contracts: {
      1: {
        registry: '0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B'
      }
    },
    fees: {
      trading: 0,
      withdrawal: 0.05
    },
    risks: {
      smartContract: 12,
      governance: 18,
      oracle: 15,
      overall: 15
    },
    audits: [
      {
        auditor: 'OpenZeppelin',
        date: '2019-09-01',
        url: 'https://blog.openzeppelin.com/compound-audit/'
      }
    ]
  },

  balancer: {
    id: 'balancer',
    name: 'Balancer',
    category: ['dex', 'amm', 'portfolio'],
    tvlRank: 7,
    website: 'https://balancer.fi',
    documentation: 'https://docs.balancer.fi',
    logo: 'https://assets.coingecko.com/coins/images/11683/small/Balancer.png',
    chains: [1, 137, 42161],
    contracts: {
      1: {
        router: '0xBA12222222228d8Ba445958a75a0704d566BF2C8',
        factory: '0x8E9aa87E45f79d6787Daa9154b8158508eF3F5ca'
      },
      137: {
        router: '0xBA12222222228d8Ba445958a75a0704d566BF2C8'
      }
    },
    fees: {
      trading: 0.1,
      performance: 2.0
    },
    risks: {
      smartContract: 20,
      governance: 22,
      oracle: 12,
      overall: 19
    },
    audits: [
      {
        auditor: 'Trail of Bits',
        date: '2020-05-01',
        url: 'https://github.com/balancer-labs/balancer-core/blob/master/audits/trail-of-bits/audit.pdf'
      }
    ]
  },

  yearn: {
    id: 'yearn',
    name: 'Yearn Finance',
    category: ['yield', 'vault', 'strategy'],
    tvlRank: 12,
    website: 'https://yearn.finance',
    documentation: 'https://docs.yearn.finance',
    logo: 'https://assets.coingecko.com/coins/images/11849/small/yfi-192x192.png',
    chains: [1, 137, 42161],
    contracts: {
      1: {
        registry: '0x50c1a2eA0a861A967D9d0FFE2AE4012c2E053804'
      }
    },
    fees: {
      trading: 0,
      performance: 20.0,
      withdrawal: 0.5
    },
    risks: {
      smartContract: 25,
      governance: 30,
      oracle: 15,
      overall: 25
    },
    audits: [
      {
        auditor: 'ChainSecurity',
        date: '2020-08-01',
        url: 'https://chainsecurity.com/security-audit/yearn-finance/'
      }
    ]
  }
};

export const PROTOCOL_CATEGORIES = {
  dex: 'Decentralized Exchange',
  amm: 'Automated Market Maker',
  lending: 'Lending Protocol',
  borrowing: 'Borrowing Protocol',
  yield: 'Yield Farming',
  vault: 'Yield Vault',
  strategy: 'Strategy Platform',
  stablecoin: 'Stablecoin Protocol',
  portfolio: 'Portfolio Management',
  insurance: 'Insurance Protocol',
  derivatives: 'Derivatives',
  options: 'Options Trading',
  futures: 'Futures Trading',
  bridge: 'Cross-chain Bridge',
  governance: 'Governance Platform'
};

export function getProtocolInfo(protocolId: string): DeFiProtocolInfo {
  const protocol = DEFI_PROTOCOLS[protocolId];
  if (!protocol) {
    throw new Error(`Unknown protocol: ${protocolId}`);
  }
  return protocol;
}

export function getProtocolsByChain(chainId: number): DeFiProtocolInfo[] {
  return Object.values(DEFI_PROTOCOLS).filter(protocol => 
    protocol.chains.includes(chainId)
  );
}

export function getProtocolsByCategory(category: string): DeFiProtocolInfo[] {
  return Object.values(DEFI_PROTOCOLS).filter(protocol =>
    protocol.category.includes(category)
  );
}

export function getProtocolRiskScore(protocolId: string): number {
  const protocol = getProtocolInfo(protocolId);
  return protocol.risks.overall;
}

export function getProtocolContracts(protocolId: string, chainId: number) {
  const protocol = getProtocolInfo(protocolId);
  return protocol.contracts[chainId] || {};
}

export function isProtocolAudited(protocolId: string): boolean {
  const protocol = getProtocolInfo(protocolId);
  return protocol.audits.length > 0;
}

export function getProtocolAuditScore(protocolId: string): number | null {
  const protocol = getProtocolInfo(protocolId);
  const auditedScores = protocol.audits
    .map(audit => audit.score)
    .filter(score => score !== undefined) as number[];
  
  if (auditedScores.length === 0) return null;
  
  return auditedScores.reduce((sum, score) => sum + score, 0) / auditedScores.length;
}
