import { ethers } from 'ethers';

// Chain configurations based on official sources
export const SUPPORTED_CHAINS = {
  1: {
    name: 'Ethereum',
    rpc: 'https://eth.llamarpc.com',
    explorer: 'https://etherscan.io',
    currency: 'ETH'
  },
  43114: {
    name: 'Avalanche',
    rpc: 'https://api.avax.network/ext/bc/C/rpc',
    explorer: 'https://snowtrace.io',
    currency: 'AVAX'
  },
  137: {
    name: 'Polygon',
    rpc: 'https://polygon-rpc.com',
    explorer: 'https://polygonscan.com',
    currency: 'MATIC'
  },
  42161: {
    name: 'Arbitrum',
    rpc: 'https://arb1.arbitrum.io/rpc',
    explorer: 'https://arbiscan.io',
    currency: 'ETH'
  }
} as const;

export type SupportedChainId = keyof typeof SUPPORTED_CHAINS;

export class Web3Service {
  private providers: Map<number, ethers.JsonRpcProvider> = new Map();

  constructor() {
    this.initializeProviders();
  }

  private initializeProviders() {
    Object.entries(SUPPORTED_CHAINS).forEach(([chainId, config]) => {
      const provider = new ethers.JsonRpcProvider(config.rpc);
      this.providers.set(Number(chainId), provider);
    });
  }

  getProvider(chainId: number): ethers.JsonRpcProvider {
    const provider = this.providers.get(chainId);
    if (!provider) {
      throw new Error(`Provider not found for chain ${chainId}`);
    }
    return provider;
  }

  async getBalance(address: string, chainId: number): Promise<bigint> {
    const provider = this.getProvider(chainId);
    return await provider.getBalance(address);
  }

  async getTokenBalance(
    tokenAddress: string,
    holderAddress: string,
    chainId: number
  ): Promise<bigint> {
    const provider = this.getProvider(chainId);
    const tokenContract = new ethers.Contract(
      tokenAddress,
      ['function balanceOf(address) view returns (uint256)'],
      provider
    );
    return await tokenContract.balanceOf(holderAddress);
  }

  async getBlockNumber(chainId: number): Promise<number> {
    const provider = this.getProvider(chainId);
    return await provider.getBlockNumber();
  }

  async getGasPrice(chainId: number): Promise<bigint> {
    const provider = this.getProvider(chainId);
    const feeData = await provider.getFeeData();
    return feeData.gasPrice || 0n;
  }

  async estimateGas(
    transaction: ethers.TransactionRequest,
    chainId: number
  ): Promise<bigint> {
    const provider = this.getProvider(chainId);
    return await provider.estimateGas(transaction);
  }

  async waitForTransaction(
    txHash: string,
    chainId: number,
    confirmations: number = 1
  ): Promise<ethers.TransactionReceipt | null> {
    const provider = this.getProvider(chainId);
    return await provider.waitForTransaction(txHash, confirmations);
  }

  getExplorerUrl(txHash: string, chainId: number): string {
    const chain = SUPPORTED_CHAINS[chainId as SupportedChainId];
    return `${chain?.explorer || 'https://etherscan.io'}/tx/${txHash}`;
  }

  getExplorerAddressUrl(address: string, chainId: number): string {
    const chain = SUPPORTED_CHAINS[chainId as SupportedChainId];
    return `${chain?.explorer || 'https://etherscan.io'}/address/${address}`;
  }

  isChainSupported(chainId: number): chainId is SupportedChainId {
    return chainId in SUPPORTED_CHAINS;
  }

  getChainName(chainId: number): string {
    const chain = SUPPORTED_CHAINS[chainId as SupportedChainId];
    return chain?.name || `Chain ${chainId}`;
  }

  getCurrency(chainId: number): string {
    const chain = SUPPORTED_CHAINS[chainId as SupportedChainId];
    return chain?.currency || 'ETH';
  }

  async getCurrentBlock(chainId: number): Promise<ethers.Block | null> {
    const provider = this.getProvider(chainId);
    return await provider.getBlock('latest');
  }

  formatEther(value: bigint): string {
    return ethers.formatEther(value);
  }

  parseEther(value: string): bigint {
    return ethers.parseEther(value);
  }

  isAddress(address: string): boolean {
    return ethers.isAddress(address);
  }

  getAddress(address: string): string {
    return ethers.getAddress(address);
  }
}

export const web3Service = new Web3Service();
