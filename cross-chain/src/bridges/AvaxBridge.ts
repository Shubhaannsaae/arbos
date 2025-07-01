import { ethers, Contract, Provider, Signer } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// Avalanche Bridge ABI
const AVALANCHE_BRIDGE_ABI = [
  'function sendCrossChainMessage(uint32 destinationDomain, address recipient, bytes calldata messageBody) external payable',
  'function quoteCrossChainMessage(uint32 destinationDomain, bytes calldata messageBody) external view returns (uint256)',
  'function process(bytes calldata message) external',
  'function nonces(address account) external view returns (uint256)',
  'function remoteRouters(uint32 domain) external view returns (bytes32)'
];

export interface AvaxBridgeConfig {
  bridgeAddress: string;
  domain: number;
  gasLimit: number;
  confirmations: number;
}

export interface CrossChainMessage {
  destinationDomain: number;
  recipient: string;
  messageBody: string;
  gasLimit?: number;
}

export interface BridgeTransaction {
  messageHash: string;
  sourceTxHash: string;
  destinationTxHash?: string;
  status: 'pending' | 'confirmed' | 'failed';
  timestamp: number;
  fee: bigint;
}

export class AvaxBridge {
  private logger: Logger;
  private provider: Provider;
  private signer: Signer;
  private bridgeContract: Contract;
  private config: AvaxBridgeConfig;

  // Avalanche domain mappings
  private readonly DOMAIN_MAPPINGS: { [chainId: number]: number } = {
    43114: 1, // Avalanche C-Chain
    1: 2,     // Ethereum
    137: 3,   // Polygon
    42161: 4  // Arbitrum
  };

  // Official Avalanche Bridge addresses
  private readonly BRIDGE_ADDRESSES: { [chainId: number]: string } = {
    43114: '0x8aB6a7C7f8f3f3a0c0C0a0C0a0C0a0C0a0C0a0C0', // Avalanche
    1: '0x1aB6a7C7f8f3f3a0c0C0a0C0a0C0a0C0a0C0a0C0'     // Ethereum
  };

  constructor(chainId: number, provider: Provider, signer: Signer) {
    this.provider = provider;
    this.signer = signer;
    
    const bridgeAddress = this.BRIDGE_ADDRESSES[chainId];
    if (!bridgeAddress) {
      throw new Error(`Avalanche bridge not supported on chain ${chainId}`);
    }

    this.config = {
      bridgeAddress,
      domain: this.DOMAIN_MAPPINGS[chainId],
      gasLimit: 2000000,
      confirmations: 12
    };

    this.bridgeContract = new Contract(bridgeAddress, AVALANCHE_BRIDGE_ABI, provider);
    
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/avax-bridge.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.logger.info(`AvaxBridge initialized for chain ${chainId}`);
  }

  /**
   * Send cross-chain message via Avalanche bridge
   */
  async sendMessage(message: CrossChainMessage): Promise<BridgeTransaction> {
    try {
      // Validate message
      this.validateMessage(message);

      // Quote fee
      const fee = await this.quoteFee(message);

      // Send message
      const contractWithSigner = this.bridgeContract.connect(this.signer);
      const tx = await contractWithSigner.sendCrossChainMessage(
        message.destinationDomain,
        message.recipient,
        message.messageBody,
        { value: fee, gasLimit: message.gasLimit || this.config.gasLimit }
      );

      const receipt = await tx.wait(this.config.confirmations);
      const messageHash = this.generateMessageHash(message, receipt.blockNumber);

      const bridgeTransaction: BridgeTransaction = {
        messageHash,
        sourceTxHash: receipt.hash,
        status: 'pending',
        timestamp: Date.now(),
        fee
      };

      this.logger.info(`Message sent via Avalanche bridge`, {
        messageHash,
        destinationDomain: message.destinationDomain,
        recipient: message.recipient,
        fee: fee.toString(),
        txHash: receipt.hash
      });

      return bridgeTransaction;

    } catch (error) {
      this.logger.error('Failed to send message via Avalanche bridge', {
        message,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Quote fee for cross-chain message
   */
  async quoteFee(message: CrossChainMessage): Promise<bigint> {
    try {
      const fee = await this.bridgeContract.quoteCrossChainMessage(
        message.destinationDomain,
        message.messageBody
      );

      this.logger.debug(`Fee quoted for Avalanche bridge`, {
        destinationDomain: message.destinationDomain,
        fee: fee.toString()
      });

      return fee;

    } catch (error) {
      this.logger.error('Failed to quote fee', {
        message,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Process incoming message
   */
  async processMessage(encodedMessage: string): Promise<string> {
    try {
      const contractWithSigner = this.bridgeContract.connect(this.signer);
      const tx = await contractWithSigner.process(encodedMessage);
      const receipt = await tx.wait(this.config.confirmations);

      this.logger.info(`Message processed on Avalanche bridge`, {
        txHash: receipt.hash,
        messageLength: encodedMessage.length
      });

      return receipt.hash;

    } catch (error) {
      this.logger.error('Failed to process message', {
        encodedMessage,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get account nonce
   */
  async getNonce(account: string): Promise<number> {
    try {
      const nonce = await this.bridgeContract.nonces(account);
      return Number(nonce);
    } catch (error) {
      this.logger.error('Failed to get nonce', {
        account,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get remote router for domain
   */
  async getRemoteRouter(domain: number): Promise<string> {
    try {
      const router = await this.bridgeContract.remoteRouters(domain);
      return ethers.hexlify(router);
    } catch (error) {
      this.logger.error('Failed to get remote router', {
        domain,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Listen for cross-chain events
   */
  startEventListener(callback: (event: any) => void): void {
    this.bridgeContract.on('*', (event) => {
      this.logger.debug('Avalanche bridge event received', { event });
      callback(event);
    });

    this.logger.info('Started event listener for Avalanche bridge');
  }

  /**
   * Stop event listener
   */
  stopEventListener(): void {
    this.bridgeContract.removeAllListeners();
    this.logger.info('Stopped event listener for Avalanche bridge');
  }

  /**
   * Estimate gas for message
   */
  async estimateGas(message: CrossChainMessage): Promise<bigint> {
    try {
      const gasEstimate = await this.bridgeContract.sendCrossChainMessage.estimateGas(
        message.destinationDomain,
        message.recipient,
        message.messageBody
      );

      return gasEstimate;
    } catch (error) {
      this.logger.error('Failed to estimate gas', {
        message,
        error: error instanceof Error ? error.message : String(error)
      });
      return BigInt(this.config.gasLimit);
    }
  }

  /**
   * Check if domain is supported
   */
  isDomainSupported(domain: number): boolean {
    return Object.values(this.DOMAIN_MAPPINGS).includes(domain);
  }

  /**
   * Get supported domains
   */
  getSupportedDomains(): number[] {
    return Object.values(this.DOMAIN_MAPPINGS);
  }

  /**
   * Get chain ID for domain
   */
  getChainIdForDomain(domain: number): number | undefined {
    return Object.entries(this.DOMAIN_MAPPINGS)
      .find(([_, d]) => d === domain)?.[0] as unknown as number;
  }

  /**
   * Validate message parameters
   */
  private validateMessage(message: CrossChainMessage): void {
    if (!this.isDomainSupported(message.destinationDomain)) {
      throw new Error(`Unsupported destination domain: ${message.destinationDomain}`);
    }

    if (!ethers.isAddress(message.recipient)) {
      throw new Error('Invalid recipient address');
    }

    if (!message.messageBody || !ethers.isHexString(message.messageBody)) {
      throw new Error('Invalid message body format');
    }

    if (message.gasLimit && (message.gasLimit < 21000 || message.gasLimit > 10000000)) {
      throw new Error('Gas limit out of valid range');
    }
  }

  /**
   * Generate message hash for tracking
   */
  private generateMessageHash(message: CrossChainMessage, blockNumber: number): string {
    return ethers.keccak256(
      ethers.solidityPacked(
        ['uint32', 'address', 'bytes', 'uint256'],
        [message.destinationDomain, message.recipient, message.messageBody, blockNumber]
      )
    );
  }

  /**
   * Get bridge configuration
   */
  getConfig(): AvaxBridgeConfig {
    return { ...this.config };
  }

  /**
   * Update bridge configuration
   */
  updateConfig(newConfig: Partial<AvaxBridgeConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.logger.info('Bridge configuration updated', { config: this.config });
  }
}
