import { ethers, Contract, Provider, Signer } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// Arbitrum Bridge ABI
const ARBITRUM_BRIDGE_ABI = [
  'function depositEth() external payable returns (uint256)',
  'function outboundTransfer(address _token, address _to, uint256 _amount, uint256 _maxGas, uint256 _gasPriceBid, bytes calldata _data) external payable returns (bytes memory)',
  'function calculateL2TokenAddress(address l1ERC20) external view returns (address)',
  'function getWithdrawal(uint256 _l2TxHash) external view returns (uint256)',
  'function executeTransaction(bytes32[] calldata proof, uint256 index, address l2Sender, address to, uint256 l2Block, uint256 l1Block, uint256 timestamp, uint256 amount, bytes calldata data) external'
];

// Arbitrum Inbox ABI
const ARBITRUM_INBOX_ABI = [
  'function createRetryableTicket(address to, uint256 l2CallValue, uint256 maxSubmissionCost, address excessFeeRefundAddress, address callValueRefundAddress, uint256 gasLimit, uint256 maxFeePerGas, bytes calldata data) external payable returns (uint256)',
  'function calculateRetryableSubmissionFee(uint256 dataLength, uint256 baseFee) external view returns (uint256)',
  'function bridge() external view returns (address)'
];

export interface ArbitrumBridgeConfig {
  l1Gateway: string;
  l2Gateway: string;
  inbox: string;
  outbox: string;
  chainId: number;
}

export interface RetryableTicket {
  to: string;
  l2CallValue: bigint;
  data: string;
  gasLimit: number;
  maxFeePerGas: bigint;
  excessFeeRefundAddress: string;
  callValueRefundAddress: string;
}

export interface WithdrawalProof {
  proof: string[];
  index: number;
  l2Sender: string;
  to: string;
  l2Block: number;
  l1Block: number;
  timestamp: number;
  amount: bigint;
  data: string;
}

export interface BridgeTransaction {
  l1TxHash: string;
  l2TxHash?: string;
  status: 'pending' | 'confirmed' | 'executed' | 'failed';
  type: 'deposit' | 'withdrawal';
  amount: bigint;
  token: string;
  timestamp: number;
  retryableTicketId?: string;
}

export class ArbitrumBridge {
  private logger: Logger;
  private provider: Provider;
  private signer: Signer;
  private l1Gateway: Contract;
  private inbox: Contract;
  private config: ArbitrumBridgeConfig;

  // Official Arbitrum bridge addresses
  private readonly MAINNET_CONFIG: ArbitrumBridgeConfig = {
    l1Gateway: '0xa3A7B6F88361F48403514059F1F16C8E78d60EeC',
    l2Gateway: '0x09e9222E96E7B4AE2a407B98d48e330053351EEe',
    inbox: '0x4Dbd4fc535Ac27206064B68FfCf827b0A60BAB3f',
    outbox: '0x0B9857ae2D4A3DBe74ffE1d7DF045bb7F96E4840',
    chainId: 42161
  };

  private readonly TESTNET_CONFIG: ArbitrumBridgeConfig = {
    l1Gateway: '0x70C143928eCfFaf9F5b406f7f4fC28Dc43d68380',
    l2Gateway: '0x6c411aD3E74De3E7Bd422b94A27770f5B86C623B',
    inbox: '0x578BAde599406A8fE3d24Fd7f7211c0911F5B29e',
    outbox: '0x45Af9Ed1D03703e480CE7d328fB684bb67DA5049',
    chainId: 421613
  };

  // Supported tokens
  private readonly SUPPORTED_TOKENS: { [l1Address: string]: string } = {
    '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8': 'USDC',
    '0x514910771AF9Ca656af840dff83E8264EcF986CA': 'LINK',
    '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599': 'WBTC',
    '0xdAC17F958D2ee523a2206206994597C13D831ec7': 'USDT'
  };

  constructor(provider: Provider, signer: Signer, isTestnet: boolean = false) {
    this.provider = provider;
    this.signer = signer;
    
    this.config = isTestnet ? this.TESTNET_CONFIG : this.MAINNET_CONFIG;

    this.l1Gateway = new Contract(this.config.l1Gateway, ARBITRUM_BRIDGE_ABI, provider);
    this.inbox = new Contract(this.config.inbox, ARBITRUM_INBOX_ABI, provider);

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/arbitrum-bridge.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.logger.info(`ArbitrumBridge initialized`, {
      isTestnet,
      config: this.config
    });
  }

  /**
   * Deposit ETH to Arbitrum
   */
  async depositEth(amount: bigint): Promise<BridgeTransaction> {
    try {
      const contractWithSigner = this.l1Gateway.connect(this.signer);
      const tx = await contractWithSigner.depositEth({ value: amount });
      const receipt = await tx.wait();

      const transaction: BridgeTransaction = {
        l1TxHash: receipt.hash,
        status: 'pending',
        type: 'deposit',
        amount,
        token: ethers.ZeroAddress,
        timestamp: Date.now()
      };

      this.logger.info(`ETH deposit initiated`, {
        amount: amount.toString(),
        txHash: receipt.hash
      });

      return transaction;

    } catch (error) {
      this.logger.error('ETH deposit failed', {
        amount: amount.toString(),
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Deposit ERC20 tokens to Arbitrum
   */
  async depositToken(
    token: string,
    to: string,
    amount: bigint,
    maxGas: number = 1000000,
    gasPriceBid: bigint
  ): Promise<BridgeTransaction> {
    try {
      // Validate parameters
      this.validateTokenDeposit(token, to, amount);

      // Calculate L2 token address
      const l2Token = await this.l1Gateway.calculateL2TokenAddress(token);

      // Prepare deposit data
      const data = ethers.solidityPacked(['uint256', 'bytes'], [amount, '0x']);

      // Execute deposit
      const contractWithSigner = this.l1Gateway.connect(this.signer);
      const tx = await contractWithSigner.outboundTransfer(
        token,
        to,
        amount,
        maxGas,
        gasPriceBid,
        data,
        { value: gasPriceBid * BigInt(maxGas) }
      );

      const receipt = await tx.wait();

      const transaction: BridgeTransaction = {
        l1TxHash: receipt.hash,
        status: 'pending',
        type: 'deposit',
        amount,
        token,
        timestamp: Date.now()
      };

      this.logger.info(`Token deposit initiated`, {
        token,
        l2Token,
        amount: amount.toString(),
        txHash: receipt.hash
      });

      return transaction;

    } catch (error) {
      this.logger.error('Token deposit failed', {
        token,
        to,
        amount: amount.toString(),
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Create retryable ticket for L2 transaction
   */
  async createRetryableTicket(ticket: RetryableTicket): Promise<BridgeTransaction> {
    try {
      // Calculate submission cost
      const dataLength = ethers.getBytes(ticket.data).length;
      const submissionCost = await this.calculateSubmissionCost(dataLength);

      // Create retryable ticket
      const contractWithSigner = this.inbox.connect(this.signer);
      const tx = await contractWithSigner.createRetryableTicket(
        ticket.to,
        ticket.l2CallValue,
        submissionCost,
        ticket.excessFeeRefundAddress,
        ticket.callValueRefundAddress,
        ticket.gasLimit,
        ticket.maxFeePerGas,
        ticket.data,
        {
          value: submissionCost + ticket.l2CallValue + (BigInt(ticket.gasLimit) * ticket.maxFeePerGas)
        }
      );

      const receipt = await tx.wait();
      const retryableTicketId = this.extractRetryableTicketId(receipt.logs);

      const transaction: BridgeTransaction = {
        l1TxHash: receipt.hash,
        status: 'pending',
        type: 'deposit',
        amount: ticket.l2CallValue,
        token: ethers.ZeroAddress,
        timestamp: Date.now(),
        retryableTicketId
      };

      this.logger.info(`Retryable ticket created`, {
        retryableTicketId,
        txHash: receipt.hash
      });

      return transaction;

    } catch (error) {
      this.logger.error('Failed to create retryable ticket', {
        ticket,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Execute withdrawal with proof
   */
  async executeWithdrawal(withdrawal: WithdrawalProof): Promise<BridgeTransaction> {
    try {
      // Validate withdrawal proof
      this.validateWithdrawalProof(withdrawal);

      // Execute withdrawal
      const contractWithSigner = this.l1Gateway.connect(this.signer);
      const tx = await contractWithSigner.executeTransaction(
        withdrawal.proof,
        withdrawal.index,
        withdrawal.l2Sender,
        withdrawal.to,
        withdrawal.l2Block,
        withdrawal.l1Block,
        withdrawal.timestamp,
        withdrawal.amount,
        withdrawal.data
      );

      const receipt = await tx.wait();

      const transaction: BridgeTransaction = {
        l1TxHash: receipt.hash,
        status: 'executed',
        type: 'withdrawal',
        amount: withdrawal.amount,
        token: ethers.ZeroAddress, // Extract from data if needed
        timestamp: Date.now()
      };

      this.logger.info(`Withdrawal executed`, {
        amount: withdrawal.amount.toString(),
        txHash: receipt.hash
      });

      return transaction;

    } catch (error) {
      this.logger.error('Withdrawal execution failed', {
        withdrawal,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Calculate submission cost for retryable ticket
   */
  async calculateSubmissionCost(dataLength: number): Promise<bigint> {
    try {
      const baseFee = await this.provider.getFeeData();
      return await this.inbox.calculateRetryableSubmissionFee(
        dataLength,
        baseFee.gasPrice || 0n
      );
    } catch (error) {
      this.logger.error('Failed to calculate submission cost', {
        dataLength,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get L2 token address for L1 token
   */
  async getL2TokenAddress(l1Token: string): Promise<string> {
    try {
      return await this.l1Gateway.calculateL2TokenAddress(l1Token);
    } catch (error) {
      this.logger.error('Failed to get L2 token address', {
        l1Token,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Check withdrawal status
   */
  async getWithdrawalStatus(l2TxHash: string): Promise<bigint> {
    try {
      return await this.l1Gateway.getWithdrawal(l2TxHash);
    } catch (error) {
      this.logger.error('Failed to get withdrawal status', {
        l2TxHash,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get supported tokens
   */
  getSupportedTokens(): { [l1Address: string]: string } {
    return { ...this.SUPPORTED_TOKENS };
  }

  /**
   * Check if token is supported
   */
  isTokenSupported(tokenAddress: string): boolean {
    return tokenAddress in this.SUPPORTED_TOKENS;
  }

  /**
   * Start event listener
   */
  startEventListener(callback: (event: any) => void): void {
    this.l1Gateway.on('*', (event) => {
      this.logger.debug('Arbitrum bridge event received', { event });
      callback(event);
    });

    this.inbox.on('*', (event) => {
      this.logger.debug('Arbitrum inbox event received', { event });
      callback(event);
    });

    this.logger.info('Started event listener for Arbitrum bridge');
  }

  /**
   * Stop event listener
   */
  stopEventListener(): void {
    this.l1Gateway.removeAllListeners();
    this.inbox.removeAllListeners();
    this.logger.info('Stopped event listener for Arbitrum bridge');
  }

  /**
   * Validate token deposit parameters
   */
  private validateTokenDeposit(token: string, to: string, amount: bigint): void {
    if (!ethers.isAddress(token)) {
      throw new Error('Invalid token address');
    }

    if (!ethers.isAddress(to)) {
      throw new Error('Invalid recipient address');
    }

    if (amount <= 0n) {
      throw new Error('Amount must be greater than 0');
    }

    if (!this.isTokenSupported(token)) {
      throw new Error(`Token ${token} not supported`);
    }
  }

  /**
   * Validate withdrawal proof
   */
  private validateWithdrawalProof(withdrawal: WithdrawalProof): void {
    if (!withdrawal.proof || withdrawal.proof.length === 0) {
      throw new Error('Invalid withdrawal proof');
    }

    if (!ethers.isAddress(withdrawal.l2Sender)) {
      throw new Error('Invalid L2 sender address');
    }

    if (!ethers.isAddress(withdrawal.to)) {
      throw new Error('Invalid recipient address');
    }

    if (withdrawal.amount <= 0n) {
      throw new Error('Amount must be greater than 0');
    }
  }

  /**
   * Extract retryable ticket ID from transaction logs
   */
  private extractRetryableTicketId(logs: any[]): string {
    // Simplified extraction - in production, parse actual event logs
    for (const log of logs) {
      if (log.topics[0] === '0x...') { // InboxMessageDelivered event signature
        return log.topics[1]; // Message ID
      }
    }
    return '';
  }

  /**
   * Get bridge configuration
   */
  getConfig(): ArbitrumBridgeConfig {
    return { ...this.config };
  }
}
