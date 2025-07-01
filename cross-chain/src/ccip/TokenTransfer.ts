import { ethers, Contract, Provider, Signer } from 'ethers';
import { CCIPManager } from './CCIPManager';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// ERC20 ABI for token operations
const ERC20_ABI = [
  'function balanceOf(address account) external view returns (uint256)',
  'function allowance(address owner, address spender) external view returns (uint256)',
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function transfer(address to, uint256 amount) external returns (bool)',
  'function transferFrom(address from, address to, uint256 amount) external returns (bool)',
  'function decimals() external view returns (uint8)',
  'function symbol() external view returns (string)',
  'function name() external view returns (string)'
];

// CCIP Token Pool ABI
const TOKEN_POOL_ABI = [
  'function getPoolConfig() external view returns ((address token, address pool, uint64[] destChainSelectors, bytes extraData))',
  'function isSupportedToken(address token) external view returns (bool)',
  'function getCurrentInboundRateLimiterState(uint64 remoteChainSelector) external view returns ((uint256 tokens, uint256 lastUpdated, bool isEnabled, uint256 capacity, uint256 rate))',
  'function getCurrentOutboundRateLimiterState(uint64 remoteChainSelector) external view returns ((uint256 tokens, uint256 lastUpdated, bool isEnabled, uint256 capacity, uint256 rate))'
];

export interface TokenInfo {
  address: string;
  name: string;
  symbol: string;
  decimals: number;
  poolAddress?: string;
}

export interface TransferParams {
  token: string;
  amount: bigint;
  recipient: string;
  destinationChainId: number;
  gasLimit?: number;
  allowFailure?: boolean;
}

export interface TransferResult {
  messageId: string;
  txHash: string;
  fee: bigint;
  timestamp: number;
}

export interface RateLimitInfo {
  tokens: bigint;
  lastUpdated: number;
  isEnabled: boolean;
  capacity: bigint;
  rate: bigint;
}

export class TokenTransfer {
  private logger: Logger;
  private ccipManager: CCIPManager;
  private providers: Map<number, Provider> = new Map();
  private signers: Map<number, Signer> = new Map();
  private tokenContracts: Map<string, Contract> = new Map();

  // Supported tokens per chain (official CCIP tokens)
  private readonly SUPPORTED_TOKENS: { [chainId: number]: TokenInfo[] } = {
    1: [ // Ethereum
      {
        address: '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8',
        name: 'USD Coin',
        symbol: 'USDC',
        decimals: 6
      },
      {
        address: '0x514910771AF9Ca656af840dff83E8264EcF986CA',
        name: 'Chainlink Token',
        symbol: 'LINK',
        decimals: 18
      }
    ],
    43114: [ // Avalanche
      {
        address: '0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E',
        name: 'USD Coin',
        symbol: 'USDC',
        decimals: 6
      },
      {
        address: '0x5947BB275c521040051D82396192181b413227A3',
        name: 'Chainlink Token',
        symbol: 'LINK',
        decimals: 18
      }
    ],
    137: [ // Polygon
      {
        address: '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',
        name: 'USD Coin',
        symbol: 'USDC',
        decimals: 6
      },
      {
        address: '0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39',
        name: 'Chainlink Token',
        symbol: 'LINK',
        decimals: 18
      }
    ],
    42161: [ // Arbitrum
      {
        address: '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8',
        name: 'USD Coin',
        symbol: 'USDC',
        decimals: 6
      },
      {
        address: '0xf97f4df75117a78c1A5a0DBb814Af92458539FB4',
        name: 'Chainlink Token',
        symbol: 'LINK',
        decimals: 18
      }
    ]
  };

  constructor(ccipManager: CCIPManager) {
    this.ccipManager = ccipManager;
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/token-transfer.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Initialize token transfer with providers and signers
   */
  async initialize(
    providers: Map<number, Provider>,
    signers: Map<number, Signer>
  ): Promise<void> {
    this.providers = providers;
    this.signers = signers;

    // Initialize token contracts
    for (const [chainId, tokens] of Object.entries(this.SUPPORTED_TOKENS)) {
      const provider = providers.get(Number(chainId));
      if (!provider) continue;

      for (const token of tokens) {
        const contract = new Contract(token.address, ERC20_ABI, provider);
        this.tokenContracts.set(`${chainId}-${token.address}`, contract);
      }
    }

    this.logger.info('Token transfer initialized with supported tokens');
  }

  /**
   * Transfer tokens cross-chain
   */
  async transferTokens(
    sourceChainId: number,
    params: TransferParams
  ): Promise<TransferResult> {
    try {
      // Validate parameters
      await this.validateTransfer(sourceChainId, params);

      // Check allowance and approve if needed
      await this.ensureAllowance(sourceChainId, params.token, params.amount);

      // Prepare CCIP message for token transfer
      const message = {
        receiver: params.recipient,
        data: '0x', // Empty data for token-only transfer
        tokenAmounts: [{
          token: params.token,
          amount: params.amount
        }],
        feeToken: ethers.ZeroAddress,
        extraArgs: CCIPManager.createExtraArgs(params.gasLimit || 200000)
      };

      // Calculate fee
      const fee = await this.ccipManager.getFee(
        sourceChainId,
        params.destinationChainId,
        message
      );

      // Send transfer
      const messageId = await this.ccipManager.sendMessage(
        sourceChainId,
        params.destinationChainId,
        message
      );

      const result: TransferResult = {
        messageId,
        txHash: '', // Will be updated by transaction
        fee,
        timestamp: Date.now()
      };

      this.logger.info(`Token transfer initiated`, {
        messageId,
        sourceChain: sourceChainId,
        destinationChain: params.destinationChainId,
        token: params.token,
        amount: params.amount.toString(),
        fee: fee.toString()
      });

      return result;

    } catch (error) {
      this.logger.error('Token transfer failed', {
        sourceChain: sourceChainId,
        params,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get token balance
   */
  async getTokenBalance(
    chainId: number,
    tokenAddress: string,
    account: string
  ): Promise<bigint> {
    const contract = this.tokenContracts.get(`${chainId}-${tokenAddress}`);
    if (!contract) {
      throw new Error(`Token contract not found: ${chainId}-${tokenAddress}`);
    }

    return await contract.balanceOf(account);
  }

  /**
   * Check token allowance
   */
  async getTokenAllowance(
    chainId: number,
    tokenAddress: string,
    owner: string,
    spender: string
  ): Promise<bigint> {
    const contract = this.tokenContracts.get(`${chainId}-${tokenAddress}`);
    if (!contract) {
      throw new Error(`Token contract not found: ${chainId}-${tokenAddress}`);
    }

    return await contract.allowance(owner, spender);
  }

  /**
   * Approve token spending
   */
  async approveToken(
    chainId: number,
    tokenAddress: string,
    spender: string,
    amount: bigint
  ): Promise<string> {
    const signer = this.signers.get(chainId);
    if (!signer) {
      throw new Error(`Signer not found for chain ${chainId}`);
    }

    const contract = this.tokenContracts.get(`${chainId}-${tokenAddress}`);
    if (!contract) {
      throw new Error(`Token contract not found: ${chainId}-${tokenAddress}`);
    }

    const contractWithSigner = contract.connect(signer);
    const tx = await contractWithSigner.approve(spender, amount);
    const receipt = await tx.wait();

    this.logger.info(`Token approval completed`, {
      chainId,
      token: tokenAddress,
      spender,
      amount: amount.toString(),
      txHash: receipt.hash
    });

    return receipt.hash;
  }

  /**
   * Get supported tokens for a chain
   */
  getSupportedTokens(chainId: number): TokenInfo[] {
    return this.SUPPORTED_TOKENS[chainId] || [];
  }

  /**
   * Check if token is supported for cross-chain transfer
   */
  async isTokenSupported(
    sourceChainId: number,
    destinationChainId: number,
    tokenAddress: string
  ): Promise<boolean> {
    try {
      const supportedTokens = await this.ccipManager.getSupportedTokens(
        sourceChainId,
        destinationChainId
      );

      return supportedTokens.includes(tokenAddress);
    } catch (error) {
      this.logger.error('Error checking token support', {
        sourceChain: sourceChainId,
        destinationChain: destinationChainId,
        token: tokenAddress,
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }

  /**
   * Get token transfer fee estimate
   */
  async getTransferFee(
    sourceChainId: number,
    params: TransferParams
  ): Promise<bigint> {
    const message = {
      receiver: params.recipient,
      data: '0x',
      tokenAmounts: [{
        token: params.token,
        amount: params.amount
      }],
      feeToken: ethers.ZeroAddress,
      extraArgs: CCIPManager.createExtraArgs(params.gasLimit || 200000)
    };

    return await this.ccipManager.getFee(
      sourceChainId,
      params.destinationChainId,
      message
    );
  }

  /**
   * Validate transfer parameters
   */
  private async validateTransfer(
    sourceChainId: number,
    params: TransferParams
  ): Promise<void> {
    // Check if chains are supported
    const isChainSupported = await this.ccipManager.isChainSupported(
      sourceChainId,
      params.destinationChainId
    );

    if (!isChainSupported) {
      throw new Error(`Transfer not supported from ${sourceChainId} to ${params.destinationChainId}`);
    }

    // Check if token is supported
    const isTokenSupported = await this.isTokenSupported(
      sourceChainId,
      params.destinationChainId,
      params.token
    );

    if (!isTokenSupported) {
      throw new Error(`Token ${params.token} not supported for cross-chain transfer`);
    }

    // Validate addresses
    if (!ethers.isAddress(params.token)) {
      throw new Error('Invalid token address');
    }

    if (!ethers.isAddress(params.recipient)) {
      throw new Error('Invalid recipient address');
    }

    // Validate amount
    if (params.amount <= 0n) {
      throw new Error('Transfer amount must be greater than 0');
    }

    // Check sender balance
    const signer = this.signers.get(sourceChainId);
    if (signer) {
      const balance = await this.getTokenBalance(
        sourceChainId,
        params.token,
        await signer.getAddress()
      );

      if (balance < params.amount) {
        throw new Error(`Insufficient balance: ${balance} < ${params.amount}`);
      }
    }
  }

  /**
   * Ensure sufficient allowance for CCIP router
   */
  private async ensureAllowance(
    chainId: number,
    tokenAddress: string,
    amount: bigint
  ): Promise<void> {
    const signer = this.signers.get(chainId);
    if (!signer) {
      throw new Error(`Signer not found for chain ${chainId}`);
    }

    const routerAddress = this.ccipManager.getRouterAddress(chainId);
    if (!routerAddress) {
      throw new Error(`Router address not found for chain ${chainId}`);
    }

    const owner = await signer.getAddress();
    const currentAllowance = await this.getTokenAllowance(
      chainId,
      tokenAddress,
      owner,
      routerAddress
    );

    if (currentAllowance < amount) {
      this.logger.info('Approving token for CCIP router', {
        chainId,
        token: tokenAddress,
        amount: amount.toString(),
        currentAllowance: currentAllowance.toString()
      });

      await this.approveToken(chainId, tokenAddress, routerAddress, amount);
    }
  }

  /**
   * Get rate limit information for token transfers
   */
  async getRateLimitInfo(
    sourceChainId: number,
    destinationChainId: number,
    tokenAddress: string
  ): Promise<{ inbound: RateLimitInfo; outbound: RateLimitInfo } | null> {
    try {
      // This would require the token pool address, which varies per token
      // Implementation depends on specific CCIP token pool contracts
      this.logger.debug('Rate limit info not available - requires token pool integration');
      return null;
    } catch (error) {
      this.logger.error('Error getting rate limit info', {
        sourceChain: sourceChainId,
        destinationChain: destinationChainId,
        token: tokenAddress,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  /**
   * Batch transfer multiple tokens
   */
  async batchTransferTokens(
    sourceChainId: number,
    transfers: TransferParams[]
  ): Promise<TransferResult[]> {
    const results: TransferResult[] = [];

    for (const transfer of transfers) {
      try {
        const result = await this.transferTokens(sourceChainId, transfer);
        results.push(result);
      } catch (error) {
        this.logger.error('Batch transfer failed for item', {
          transfer,
          error: error instanceof Error ? error.message : String(error)
        });
        
        if (!transfer.allowFailure) {
          throw error;
        }
      }
    }

    this.logger.info(`Completed batch transfer: ${results.length}/${transfers.length} successful`);
    return results;
  }
}
