import { ethers, Contract, Provider, Signer } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// Ethereum Bridge ABI (LayerZero based)
const ETHEREUM_BRIDGE_ABI = [
  'function send(uint16 _dstChainId, bytes calldata _destination, bytes calldata _payload, address payable _refundAddress, address _zroPaymentAddress, bytes calldata _adapterParams) external payable',
  'function estimateFees(uint16 _dstChainId, address _userApplication, bytes calldata _payload, bool _payInZRO, bytes calldata _adapterParam) external view returns (uint nativeFee, uint zroFee)',
  'function getConfig(uint16 _version, uint16 _chainId, address _userApplication, uint _configType) external view returns (bytes memory)',
  'function isTrustedRemote(uint16 _srcChainId, bytes calldata _srcAddress) external view returns (bool)'
];

export interface EthBridgeConfig {
  endpointAddress: string;
  chainId: number;
  gasLimit: number;
  confirmations: number;
}

export interface LayerZeroMessage {
  dstChainId: number;
  destination: string;
  payload: string;
  refundAddress: string;
  adapterParams?: string;
}

export interface MessageStatus {
  messageId: string;
  status: 'sent' | 'delivered' | 'failed';
  srcTxHash: string;
  dstTxHash?: string;
  timestamp: number;
  retryCount: number;
}

export class EthBridge {
  private logger: Logger;
  private provider: Provider;
  private signer: Signer;
  private endpointContract: Contract;
  private config: EthBridgeConfig;

  // LayerZero Chain IDs
  private readonly LZ_CHAIN_IDS: { [chainId: number]: number } = {
    1: 101,     // Ethereum
    137: 109,   // Polygon
    43114: 106, // Avalanche
    42161: 110, // Arbitrum
    56: 102,    // BSC
    250: 112    // Fantom
  };

  // LayerZero Endpoint addresses
  private readonly ENDPOINT_ADDRESSES: { [chainId: number]: string } = {
    1: '0x66A71Dcef29A0fFBDBE3c6a460a3B5BC225Cd675',     // Ethereum
    137: '0x3c2269811836af69497E5F486A85D7316753cf62',   // Polygon
    43114: '0x3c2269811836af69497E5F486A85D7316753cf62', // Avalanche
    42161: '0x3c2269811836af69497E5F486A85D7316753cf62', // Arbitrum
    56: '0x3c2269811836af69497E5F486A85D7316753cf62',    // BSC
    250: '0xb6319cC6c8c27A8F5dAF0dD3DF91EA35C4720dd7'   // Fantom
  };

  constructor(chainId: number, provider: Provider, signer: Signer) {
    this.provider = provider;
    this.signer = signer;

    const endpointAddress = this.ENDPOINT_ADDRESSES[chainId];
    if (!endpointAddress) {
      throw new Error(`LayerZero endpoint not available on chain ${chainId}`);
    }

    this.config = {
      endpointAddress,
      chainId,
      gasLimit: 200000,
      confirmations: 12
    };

    this.endpointContract = new Contract(endpointAddress, ETHEREUM_BRIDGE_ABI, provider);

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/eth-bridge.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.logger.info(`EthBridge initialized for chain ${chainId}`);
  }

  /**
   * Send cross-chain message via LayerZero
   */
  async sendMessage(message: LayerZeroMessage): Promise<MessageStatus> {
    try {
      // Validate message
      this.validateMessage(message);

      // Estimate fees
      const { nativeFee } = await this.estimateFees(message);

      // Send message
      const contractWithSigner = this.endpointContract.connect(this.signer);
      const tx = await contractWithSigner.send(
        message.dstChainId,
        message.destination,
        message.payload,
        message.refundAddress,
        ethers.ZeroAddress, // zroPaymentAddress
        message.adapterParams || '0x',
        { 
          value: nativeFee,
          gasLimit: this.config.gasLimit
        }
      );

      const receipt = await tx.wait(this.config.confirmations);
      const messageId = this.generateMessageId(message, receipt.blockNumber);

      const status: MessageStatus = {
        messageId,
        status: 'sent',
        srcTxHash: receipt.hash,
        timestamp: Date.now(),
        retryCount: 0
      };

      this.logger.info(`Message sent via LayerZero`, {
        messageId,
        dstChainId: message.dstChainId,
        fee: nativeFee.toString(),
        txHash: receipt.hash
      });

      return status;

    } catch (error) {
      this.logger.error('Failed to send message via LayerZero', {
        message,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Estimate fees for LayerZero message
   */
  async estimateFees(message: LayerZeroMessage): Promise<{ nativeFee: bigint; zroFee: bigint }> {
    try {
      const [nativeFee, zroFee] = await this.endpointContract.estimateFees(
        message.dstChainId,
        await this.signer.getAddress(),
        message.payload,
        false, // payInZRO
        message.adapterParams || '0x'
      );

      this.logger.debug(`Fees estimated for LayerZero`, {
        dstChainId: message.dstChainId,
        nativeFee: nativeFee.toString(),
        zroFee: zroFee.toString()
      });

      return { nativeFee, zroFee };

    } catch (error) {
      this.logger.error('Failed to estimate fees', {
        message,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get LayerZero configuration
   */
  async getConfig(
    version: number,
    chainId: number,
    configType: number
  ): Promise<string> {
    try {
      const config = await this.endpointContract.getConfig(
        version,
        chainId,
        await this.signer.getAddress(),
        configType
      );

      return ethers.hexlify(config);
    } catch (error) {
      this.logger.error('Failed to get config', {
        version,
        chainId,
        configType,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Check if remote address is trusted
   */
  async isTrustedRemote(srcChainId: number, srcAddress: string): Promise<boolean> {
    try {
      return await this.endpointContract.isTrustedRemote(srcChainId, srcAddress);
    } catch (error) {
      this.logger.error('Failed to check trusted remote', {
        srcChainId,
        srcAddress,
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }

  /**
   * Create adapter parameters for gas settings
   */
  static createAdapterParams(
    version: number,
    gasLimit: number,
    nativeForDst: bigint = 0n,
    dstAddress?: string
  ): string {
    if (version === 1) {
      return ethers.solidityPacked(['uint16', 'uint256'], [version, gasLimit]);
    } else if (version === 2) {
      return ethers.solidityPacked(
        ['uint16', 'uint256', 'uint256', 'address'],
        [version, gasLimit, nativeForDst, dstAddress || ethers.ZeroAddress]
      );
    }
    
    throw new Error(`Unsupported adapter params version: ${version}`);
  }

  /**
   * Get LayerZero chain ID for standard chain ID
   */
  getLzChainId(chainId: number): number | undefined {
    return this.LZ_CHAIN_IDS[chainId];
  }

  /**
   * Get standard chain ID for LayerZero chain ID
   */
  getStandardChainId(lzChainId: number): number | undefined {
    return Object.entries(this.LZ_CHAIN_IDS)
      .find(([_, lzId]) => lzId === lzChainId)?.[0] as unknown as number;
  }

  /**
   * Check if chain is supported
   */
  isChainSupported(chainId: number): boolean {
    return chainId in this.LZ_CHAIN_IDS;
  }

  /**
   * Get supported chains
   */
  getSupportedChains(): number[] {
    return Object.keys(this.LZ_CHAIN_IDS).map(Number);
  }

  /**
   * Listen for LayerZero events
   */
  startEventListener(callback: (event: any) => void): void {
    // Listen for various LayerZero events
    this.endpointContract.on('*', (event) => {
      this.logger.debug('LayerZero event received', { event });
      callback(event);
    });

    this.logger.info('Started event listener for LayerZero bridge');
  }

  /**
   * Stop event listener
   */
  stopEventListener(): void {
    this.endpointContract.removeAllListeners();
    this.logger.info('Stopped event listener for LayerZero bridge');
  }

  /**
   * Retry failed message
   */
  async retryMessage(
    messageStatus: MessageStatus,
    newGasLimit?: number
  ): Promise<MessageStatus> {
    try {
      this.logger.info(`Retrying message ${messageStatus.messageId}`, {
        retryCount: messageStatus.retryCount + 1
      });

      // Implementation would depend on LayerZero retry mechanism
      // This is a simplified version
      
      messageStatus.retryCount++;
      messageStatus.timestamp = Date.now();
      
      return messageStatus;
    } catch (error) {
      this.logger.error('Failed to retry message', {
        messageId: messageStatus.messageId,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Validate message parameters
   */
  private validateMessage(message: LayerZeroMessage): void {
    const lzChainId = this.getLzChainId(message.dstChainId);
    if (!lzChainId) {
      throw new Error(`Unsupported destination chain: ${message.dstChainId}`);
    }

    if (!message.destination || !ethers.isHexString(message.destination)) {
      throw new Error('Invalid destination format');
    }

    if (!message.payload || !ethers.isHexString(message.payload)) {
      throw new Error('Invalid payload format');
    }

    if (!ethers.isAddress(message.refundAddress)) {
      throw new Error('Invalid refund address');
    }
  }

  /**
   * Generate message ID for tracking
   */
  private generateMessageId(message: LayerZeroMessage, blockNumber: number): string {
    return ethers.keccak256(
      ethers.solidityPacked(
        ['uint16', 'bytes', 'bytes', 'uint256'],
        [message.dstChainId, message.destination, message.payload, blockNumber]
      )
    );
  }

  /**
   * Get bridge configuration
   */
  getConfig(): EthBridgeConfig {
    return { ...this.config };
  }

  /**
   * Update bridge configuration
   */
  updateConfig(newConfig: Partial<EthBridgeConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.logger.info('Bridge configuration updated', { config: this.config });
  }
}
