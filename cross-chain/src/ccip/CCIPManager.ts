import { ethers, Contract, Provider, Signer } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// CCIP Router ABI (simplified - use full ABI from @chainlink/contracts)
const CCIP_ROUTER_ABI = [
  'function ccipSend(uint64 destinationChainSelector, (bytes receiver, bytes data, (address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external payable returns (bytes32)',
  'function getFee(uint64 destinationChainSelector, (bytes receiver, bytes data, (address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external view returns (uint256 fee)',
  'function getSupportedTokens(uint64 chainSelector) external view returns (address[] memory tokens)',
  'function isChainSupported(uint64 chainSelector) external view returns (bool supported)'
];

export interface CCIPMessage {
  receiver: string;
  data: string;
  tokenAmounts: TokenAmount[];
  feeToken: string;
  extraArgs: string;
}

export interface TokenAmount {
  token: string;
  amount: bigint;
}

export interface CCIPConfig {
  routerAddress: string;
  chainSelector: bigint;
  gasLimit: number;
  confirmations: number;
}

export interface ChainConfig {
  [chainId: number]: CCIPConfig;
}

export class CCIPManager {
  private logger: Logger;
  private routers: Map<number, Contract> = new Map();
  private providers: Map<number, Provider> = new Map();
  private signers: Map<number, Signer> = new Map();

  // Official Chainlink CCIP Router addresses
  private readonly ROUTER_ADDRESSES: { [chainId: number]: string } = {
    1: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D', // Ethereum
    43114: '0xF4c7E640EdA248ef95972845a62bdC74237805dB', // Avalanche
    137: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43', // Polygon
    42161: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8' // Arbitrum
  };

  // Official Chain Selectors
  private readonly CHAIN_SELECTORS: { [chainId: number]: bigint } = {
    1: 5009297550715157269n, // Ethereum
    43114: 6433500567565415381n, // Avalanche
    137: 4051577828743386545n, // Polygon
    42161: 4949039107694359620n // Arbitrum
  };

  constructor() {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/ccip-error.log', level: 'error' }),
        new transports.File({ filename: 'logs/ccip-combined.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Initialize CCIP manager with providers and signers
   */
  async initialize(
    providers: Map<number, Provider>,
    signers: Map<number, Signer>
  ): Promise<void> {
    this.providers = providers;
    this.signers = signers;

    // Initialize router contracts for each chain
    for (const [chainId, provider] of providers.entries()) {
      const routerAddress = this.ROUTER_ADDRESSES[chainId];
      if (!routerAddress) {
        this.logger.warn(`No CCIP router found for chain ${chainId}`);
        continue;
      }

      const router = new Contract(routerAddress, CCIP_ROUTER_ABI, provider);
      this.routers.set(chainId, router);
      
      this.logger.info(`Initialized CCIP router for chain ${chainId}: ${routerAddress}`);
    }
  }

  /**
   * Send cross-chain message using CCIP
   */
  async sendMessage(
    sourceChainId: number,
    destinationChainId: number,
    message: CCIPMessage
  ): Promise<string> {
    const sourceRouter = this.routers.get(sourceChainId);
    const sourceSigner = this.signers.get(sourceChainId);
    
    if (!sourceRouter || !sourceSigner) {
      throw new Error(`Router or signer not found for source chain ${sourceChainId}`);
    }

    const destinationSelector = this.CHAIN_SELECTORS[destinationChainId];
    if (!destinationSelector) {
      throw new Error(`Chain selector not found for destination chain ${destinationChainId}`);
    }

    // Connect router with signer
    const routerWithSigner = sourceRouter.connect(sourceSigner);

    // Calculate fee
    const fee = await this.getFee(sourceChainId, destinationChainId, message);

    // Send message
    const tx = await routerWithSigner.ccipSend(
      destinationSelector,
      {
        receiver: ethers.getBytes(message.receiver),
        data: ethers.getBytes(message.data),
        tokenAmounts: message.tokenAmounts.map(ta => ({
          token: ta.token,
          amount: ta.amount
        })),
        feeToken: message.feeToken,
        extraArgs: message.extraArgs
      },
      { value: message.feeToken === ethers.ZeroAddress ? fee : 0 }
    );

    const receipt = await tx.wait();
    const messageId = receipt.logs[0].topics[1]; // CCIP message ID from logs

    this.logger.info(`CCIP message sent: ${messageId}`, {
      sourceChain: sourceChainId,
      destinationChain: destinationChainId,
      txHash: receipt.hash
    });

    return messageId;
  }

  /**
   * Calculate fee for cross-chain message
   */
  async getFee(
    sourceChainId: number,
    destinationChainId: number,
    message: CCIPMessage
  ): Promise<bigint> {
    const router = this.routers.get(sourceChainId);
    if (!router) {
      throw new Error(`Router not found for chain ${sourceChainId}`);
    }

    const destinationSelector = this.CHAIN_SELECTORS[destinationChainId];
    if (!destinationSelector) {
      throw new Error(`Chain selector not found for destination chain ${destinationChainId}`);
    }

    const fee = await router.getFee(destinationSelector, {
      receiver: ethers.getBytes(message.receiver),
      data: ethers.getBytes(message.data),
      tokenAmounts: message.tokenAmounts.map(ta => ({
        token: ta.token,
        amount: ta.amount
      })),
      feeToken: message.feeToken,
      extraArgs: message.extraArgs
    });

    return fee;
  }

  /**
   * Get supported tokens for a destination chain
   */
  async getSupportedTokens(sourceChainId: number, destinationChainId: number): Promise<string[]> {
    const router = this.routers.get(sourceChainId);
    if (!router) {
      throw new Error(`Router not found for chain ${sourceChainId}`);
    }

    const destinationSelector = this.CHAIN_SELECTORS[destinationChainId];
    if (!destinationSelector) {
      throw new Error(`Chain selector not found for destination chain ${destinationChainId}`);
    }

    return await router.getSupportedTokens(destinationSelector);
  }

  /**
   * Check if chain is supported
   */
  async isChainSupported(sourceChainId: number, destinationChainId: number): Promise<boolean> {
    const router = this.routers.get(sourceChainId);
    if (!router) {
      return false;
    }

    const destinationSelector = this.CHAIN_SELECTORS[destinationChainId];
    if (!destinationSelector) {
      return false;
    }

    return await router.isChainSupported(destinationSelector);
  }

  /**
   * Get all supported chains
   */
  getSupportedChains(): number[] {
    return Object.keys(this.CHAIN_SELECTORS).map(Number);
  }

  /**
   * Get chain selector for chain ID
   */
  getChainSelector(chainId: number): bigint | undefined {
    return this.CHAIN_SELECTORS[chainId];
  }

  /**
   * Get router address for chain
   */
  getRouterAddress(chainId: number): string | undefined {
    return this.ROUTER_ADDRESSES[chainId];
  }

  /**
   * Create CCIP extra args
   */
  static createExtraArgs(gasLimit: number): string {
    // CCIP Extra Args V1 encoding
    const extraArgsV1 = ethers.solidityPacked(
      ['uint256', 'uint256'],
      [200000, gasLimit] // [gasLimit, strict]
    );
    
    return ethers.solidityPacked(
      ['uint256', 'bytes'],
      [1, extraArgsV1] // Version 1
    );
  }

  /**
   * Estimate optimal gas limit for cross-chain call
   */
  async estimateGasLimit(
    destinationChainId: number,
    calldata: string
  ): Promise<number> {
    // Base gas for CCIP overhead
    const baseGas = 200000;
    
    // Estimate gas for calldata
    const calldataGas = ethers.getBytes(calldata).length * 16;
    
    // Chain-specific multipliers
    const chainMultipliers: { [chainId: number]: number } = {
      1: 1.2,    // Ethereum
      43114: 1.1, // Avalanche  
      137: 1.3,   // Polygon
      42161: 1.1  // Arbitrum
    };

    const multiplier = chainMultipliers[destinationChainId] || 1.2;
    const estimatedGas = Math.ceil((baseGas + calldataGas) * multiplier);

    this.logger.debug(`Gas estimate for chain ${destinationChainId}: ${estimatedGas}`);
    
    return Math.min(estimatedGas, 2000000); // Cap at 2M gas
  }
}
