import { ethers, Provider, TransactionReceipt, TransactionResponse } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import { EventEmitter } from 'events';

export interface CrossChainTransaction {
  id: string;
  sourceChain: number;
  destinationChain: number;
  protocol: string;
  sourceHash: string;
  destinationHash?: string;
  messageId?: string;
  status: TransactionStatus;
  amount: bigint;
  token: string;
  sender: string;
  recipient: string;
  fee: bigint;
  timestamps: {
    initiated: number;
    confirmed?: number;
    delivered?: number;
    failed?: number;
  };
  attempts: number;
  lastError?: string;
}

export enum TransactionStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  IN_TRANSIT = 'in_transit',
  DELIVERED = 'delivered',
  FAILED = 'failed',
  EXPIRED = 'expired'
}

export interface TrackingConfig {
  confirmations: { [chainId: number]: number };
  timeouts: { [protocol: string]: number };
  retryAttempts: number;
  pollingInterval: number;
}

export class TransactionTracker extends EventEmitter {
  private logger: Logger;
  private providers: Map<number, Provider> = new Map();
  private transactions: Map<string, CrossChainTransaction> = new Map();
  private trackingIntervals: Map<string, NodeJS.Timeout> = new Map();
  private config: TrackingConfig;

  // Protocol-specific event signatures for tracking
  private readonly EVENT_SIGNATURES = {
    ccip: {
      send: '0x35c02761bcd3ef995c6a601a1981f4ed3934dcbe5041e24e286c89f5531d17e4', // CCIPSendRequested
      receive: '0x05665fe9ad095383d018353f4cbcba77e84db27dd215081bbf7cdf9ae6fbe48b'  // CCIPReceive
    },
    layerzero: {
      send: '0xe9bded5f24a4168e4f3bf44e00298c993b22376aad8c58c7dda9718a54cbea82', // MessageSent
      receive: '0x54c8a4f4c8c4f4c8c4f4c8c4f4c8c4f4c8c4f4c8c4f4c8c4f4c8c4f4c8c4f4' // MessageReceived
    }
  };

  constructor(providers: Map<number, Provider>, config?: Partial<TrackingConfig>) {
    super();
    this.providers = providers;
    
    this.config = {
      confirmations: {
        1: 12,     // Ethereum
        43114: 1,  // Avalanche
        137: 128,  // Polygon
        42161: 1   // Arbitrum
      },
      timeouts: {
        ccip: 1800000,      // 30 minutes
        layerzero: 900000,  // 15 minutes
        polygon: 3600000,   // 1 hour
        arbitrum: 900000    // 15 minutes
      },
      retryAttempts: 3,
      pollingInterval: 30000, // 30 seconds
      ...config
    };

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/transaction-tracker.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Track new cross-chain transaction
   */
  async trackTransaction(
    sourceHash: string,
    sourceChain: number,
    destinationChain: number,
    protocol: string,
    metadata: {
      amount: bigint;
      token: string;
      sender: string;
      recipient: string;
      fee: bigint;
      messageId?: string;
    }
  ): Promise<string> {
    const transactionId = this.generateTransactionId(sourceHash, sourceChain, destinationChain);

    const transaction: CrossChainTransaction = {
      id: transactionId,
      sourceChain,
      destinationChain,
      protocol,
      sourceHash,
      messageId: metadata.messageId,
      status: TransactionStatus.PENDING,
      amount: metadata.amount,
      token: metadata.token,
      sender: metadata.sender,
      recipient: metadata.recipient,
      fee: metadata.fee,
      timestamps: {
        initiated: Date.now()
      },
      attempts: 0
    };

    this.transactions.set(transactionId, transaction);

    // Start tracking
    await this.startTracking(transactionId);

    this.logger.info(`Started tracking transaction ${transactionId}`, {
      sourceHash,
      sourceChain,
      destinationChain,
      protocol
    });

    this.emit('transaction:started', transaction);
    return transactionId;
  }

  /**
   * Get transaction status
   */
  getTransaction(transactionId: string): CrossChainTransaction | undefined {
    return this.transactions.get(transactionId);
  }

  /**
   * Get all transactions with optional filters
   */
  getTransactions(filters?: {
    status?: TransactionStatus;
    protocol?: string;
    sourceChain?: number;
    destinationChain?: number;
    sender?: string;
  }): CrossChainTransaction[] {
    let transactions = Array.from(this.transactions.values());

    if (filters) {
      if (filters.status) {
        transactions = transactions.filter(tx => tx.status === filters.status);
      }
      if (filters.protocol) {
        transactions = transactions.filter(tx => tx.protocol === filters.protocol);
      }
      if (filters.sourceChain) {
        transactions = transactions.filter(tx => tx.sourceChain === filters.sourceChain);
      }
      if (filters.destinationChain) {
        transactions = transactions.filter(tx => tx.destinationChain === filters.destinationChain);
      }
      if (filters.sender) {
        transactions = transactions.filter(tx => tx.sender.toLowerCase() === filters.sender!.toLowerCase());
      }
    }

    return transactions;
  }

  /**
   * Stop tracking transaction
   */
  stopTracking(transactionId: string): void {
    const interval = this.trackingIntervals.get(transactionId);
    if (interval) {
      clearInterval(interval);
      this.trackingIntervals.delete(transactionId);
    }

    this.logger.info(`Stopped tracking transaction ${transactionId}`);
  }

  /**
   * Stop tracking all transactions
   */
  stopAllTracking(): void {
    for (const [transactionId, interval] of this.trackingIntervals) {
      clearInterval(interval);
    }
    this.trackingIntervals.clear();
    
    this.logger.info('Stopped tracking all transactions');
  }

  /**
   * Get tracking statistics
   */
  getTrackingStats(): {
    total: number;
    byStatus: { [status: string]: number };
    byProtocol: { [protocol: string]: number };
    averageDeliveryTime: number;
    successRate: number;
  } {
    const transactions = Array.from(this.transactions.values());
    
    const byStatus: { [status: string]: number } = {};
    const byProtocol: { [protocol: string]: number } = {};
    
    let totalDeliveryTime = 0;
    let deliveredCount = 0;
    let successCount = 0;

    for (const tx of transactions) {
      // Count by status
      byStatus[tx.status] = (byStatus[tx.status] || 0) + 1;
      
      // Count by protocol
      byProtocol[tx.protocol] = (byProtocol[tx.protocol] || 0) + 1;
      
      // Calculate delivery time
      if (tx.status === TransactionStatus.DELIVERED && tx.timestamps.delivered) {
        totalDeliveryTime += tx.timestamps.delivered - tx.timestamps.initiated;
        deliveredCount++;
        successCount++;
      } else if (tx.status === TransactionStatus.CONFIRMED) {
        successCount++;
      }
    }

    return {
      total: transactions.length,
      byStatus,
      byProtocol,
      averageDeliveryTime: deliveredCount > 0 ? totalDeliveryTime / deliveredCount : 0,
      successRate: transactions.length > 0 ? (successCount / transactions.length) * 100 : 0
    };
  }

  /**
   * Start tracking specific transaction
   */
  private async startTracking(transactionId: string): Promise<void> {
    const transaction = this.transactions.get(transactionId);
    if (!transaction) {
      throw new Error(`Transaction ${transactionId} not found`);
    }

    // Initial check
    await this.checkTransactionStatus(transactionId);

    // Set up polling interval
    const intervalId = setInterval(
      () => this.checkTransactionStatus(transactionId),
      this.config.pollingInterval
    );
    
    this.trackingIntervals.set(transactionId, intervalId);

    // Set timeout for expiration
    const timeout = this.config.timeouts[transaction.protocol] || 1800000;
    setTimeout(() => {
      this.handleTransactionTimeout(transactionId);
    }, timeout);
  }

  /**
   * Check transaction status
   */
  private async checkTransactionStatus(transactionId: string): Promise<void> {
    const transaction = this.transactions.get(transactionId);
    if (!transaction) {
      return;
    }

    try {
      transaction.attempts++;

      switch (transaction.status) {
        case TransactionStatus.PENDING:
          await this.checkSourceConfirmation(transaction);
          break;
        case TransactionStatus.CONFIRMED:
          await this.checkDestinationDelivery(transaction);
          break;
        case TransactionStatus.IN_TRANSIT:
          await this.checkDestinationDelivery(transaction);
          break;
      }

    } catch (error) {
      this.logger.error(`Error checking transaction ${transactionId}`, {
        error: error instanceof Error ? error.message : String(error),
        attempts: transaction.attempts
      });

      transaction.lastError = error instanceof Error ? error.message : String(error);

      // Mark as failed after max attempts
      if (transaction.attempts >= this.config.retryAttempts) {
        this.updateTransactionStatus(transaction, TransactionStatus.FAILED);
      }
    }
  }

  /**
   * Check source transaction confirmation
   */
  private async checkSourceConfirmation(transaction: CrossChainTransaction): Promise<void> {
    const provider = this.providers.get(transaction.sourceChain);
    if (!provider) {
      throw new Error(`Provider not found for chain ${transaction.sourceChain}`);
    }

    const receipt = await provider.getTransactionReceipt(transaction.sourceHash);
    if (!receipt) {
      return; // Transaction not yet mined
    }

    const currentBlock = await provider.getBlockNumber();
    const confirmations = currentBlock - receipt.blockNumber;
    const requiredConfirmations = this.config.confirmations[transaction.sourceChain] || 12;

    if (confirmations >= requiredConfirmations) {
      transaction.timestamps.confirmed = Date.now();
      
      // Extract message ID from logs if not provided
      if (!transaction.messageId) {
        transaction.messageId = this.extractMessageId(receipt, transaction.protocol);
      }

      this.updateTransactionStatus(transaction, TransactionStatus.CONFIRMED);
    }
  }

  /**
   * Check destination delivery
   */
  private async checkDestinationDelivery(transaction: CrossChainTransaction): Promise<void> {
    if (!transaction.messageId) {
      // Try to find destination transaction by other means
      const destHash = await this.findDestinationTransaction(transaction);
      if (destHash) {
        transaction.destinationHash = destHash;
        transaction.timestamps.delivered = Date.now();
        this.updateTransactionStatus(transaction, TransactionStatus.DELIVERED);
      }
      return;
    }

    const provider = this.providers.get(transaction.destinationChain);
    if (!provider) {
      throw new Error(`Provider not found for chain ${transaction.destinationChain}`);
    }

    // Look for delivery events
    const delivered = await this.checkMessageDelivery(
      provider,
      transaction.messageId,
      transaction.protocol
    );

    if (delivered) {
      transaction.destinationHash = delivered.transactionHash;
      transaction.timestamps.delivered = Date.now();
      this.updateTransactionStatus(transaction, TransactionStatus.DELIVERED);
    } else if (transaction.status === TransactionStatus.CONFIRMED) {
      this.updateTransactionStatus(transaction, TransactionStatus.IN_TRANSIT);
    }
  }

  /**
   * Extract message ID from transaction receipt
   */
  private extractMessageId(receipt: TransactionReceipt, protocol: string): string | undefined {
    const eventSig = this.EVENT_SIGNATURES[protocol as keyof typeof this.EVENT_SIGNATURES]?.send;
    if (!eventSig) {
      return undefined;
    }

    for (const log of receipt.logs) {
      if (log.topics[0] === eventSig) {
        // Message ID is typically the first indexed parameter
        return log.topics[1];
      }
    }

    return undefined;
  }

  /**
   * Check message delivery on destination chain
   */
  private async checkMessageDelivery(
    provider: Provider,
    messageId: string,
    protocol: string
  ): Promise<{ transactionHash: string; blockNumber: number } | null> {
    const eventSig = this.EVENT_SIGNATURES[protocol as keyof typeof this.EVENT_SIGNATURES]?.receive;
    if (!eventSig) {
      return null;
    }

    try {
      // Query recent blocks for delivery events
      const currentBlock = await provider.getBlockNumber();
      const fromBlock = Math.max(currentBlock - 2000, 0); // Last 2000 blocks

      const logs = await provider.getLogs({
        fromBlock,
        toBlock: currentBlock,
        topics: [eventSig, messageId]
      });

      if (logs.length > 0) {
        const log = logs[0];
        return {
          transactionHash: log.transactionHash,
          blockNumber: log.blockNumber
        };
      }

      return null;
    } catch (error) {
      this.logger.error('Error checking message delivery', {
        messageId,
        protocol,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  /**
   * Find destination transaction by other means
   */
  private async findDestinationTransaction(
    transaction: CrossChainTransaction
  ): Promise<string | undefined> {
    // This is a simplified approach - in production, use protocol-specific APIs
    // or maintain a mapping service for cross-chain transaction correlation
    
    const provider = this.providers.get(transaction.destinationChain);
    if (!provider) {
      return undefined;
    }

    try {
      // Look for transactions to the recipient around the expected time
      const currentBlock = await provider.getBlockNumber();
      const searchBlocks = 1000; // Search last 1000 blocks
      
      for (let i = 0; i < searchBlocks; i++) {
        const blockNumber = currentBlock - i;
        if (blockNumber < 0) break;
        
        const block = await provider.getBlock(blockNumber, true);
        if (!block || !block.transactions) continue;
        
        for (const tx of block.transactions) {
          if (typeof tx === 'string') continue;
          
          // Simple heuristic: transaction to recipient with similar value
          if (tx.to?.toLowerCase() === transaction.recipient.toLowerCase()) {
            // Additional checks can be added here
            return tx.hash;
          }
        }
      }
      
      return undefined;
    } catch (error) {
      this.logger.error('Error finding destination transaction', {
        transaction: transaction.id,
        error: error instanceof Error ? error.message : String(error)
      });
      return undefined;
    }
  }

  /**
   * Update transaction status and emit events
   */
  private updateTransactionStatus(
    transaction: CrossChainTransaction,
    status: TransactionStatus
  ): void {
    const previousStatus = transaction.status;
    transaction.status = status;

    this.logger.info(`Transaction status updated`, {
      transactionId: transaction.id,
      previousStatus,
      newStatus: status
    });

    this.emit('transaction:status_changed', {
      transaction,
      previousStatus,
      newStatus: status
    });

    // Stop tracking if final status
    if (status === TransactionStatus.DELIVERED || 
        status === TransactionStatus.FAILED || 
        status === TransactionStatus.EXPIRED) {
      this.stopTracking(transaction.id);
    }
  }

  /**
   * Handle transaction timeout
   */
  private handleTransactionTimeout(transactionId: string): void {
    const transaction = this.transactions.get(transactionId);
    if (!transaction) {
      return;
    }

    if (transaction.status !== TransactionStatus.DELIVERED &&
        transaction.status !== TransactionStatus.FAILED) {
      
      transaction.timestamps.failed = Date.now();
      this.updateTransactionStatus(transaction, TransactionStatus.EXPIRED);
      
      this.logger.warn(`Transaction expired`, {
        transactionId,
        duration: Date.now() - transaction.timestamps.initiated
      });
    }
  }

  /**
   * Generate unique transaction ID
   */
  private generateTransactionId(
    sourceHash: string,
    sourceChain: number,
    destinationChain: number
  ): string {
    return ethers.keccak256(
      ethers.solidityPacked(
        ['string', 'uint256', 'uint256'],
        [sourceHash, sourceChain, destinationChain]
      )
    );
  }

  /**
   * Clean up old completed transactions
   */
  cleanup(maxAgeHours: number = 24): void {
    const cutoffTime = Date.now() - (maxAgeHours * 60 * 60 * 1000);
    let cleanedCount = 0;

    for (const [id, transaction] of this.transactions.entries()) {
      const isCompleted = transaction.status === TransactionStatus.DELIVERED ||
                         transaction.status === TransactionStatus.FAILED ||
                         transaction.status === TransactionStatus.EXPIRED;
      
      if (isCompleted && transaction.timestamps.initiated < cutoffTime) {
        this.transactions.delete(id);
        this.stopTracking(id);
        cleanedCount++;
      }
    }

    this.logger.info(`Cleaned up ${cleanedCount} old transactions`);
  }
}
