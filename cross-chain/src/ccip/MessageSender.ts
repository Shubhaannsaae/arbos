import { ethers } from 'ethers';
import { CCIPManager, CCIPMessage } from './CCIPManager';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

export interface SendMessageParams {
  destinationChainId: number;
  receiver: string;
  data: string;
  tokenTransfers?: TokenTransfer[];
  feeToken?: string;
  gasLimit?: number;
}

export interface TokenTransfer {
  token: string;
  amount: bigint;
}

export interface MessageStatus {
  messageId: string;
  status: 'pending' | 'confirmed' | 'failed';
  sourceChain: number;
  destinationChain: number;
  timestamp: number;
  txHash?: string;
  error?: string;
}

export class MessageSender {
  private logger: Logger;
  private ccipManager: CCIPManager;
  private messageStatuses: Map<string, MessageStatus> = new Map();

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
        new transports.File({ filename: 'logs/message-sender.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Send cross-chain message
   */
  async sendMessage(
    sourceChainId: number,
    params: SendMessageParams
  ): Promise<string> {
    try {
      // Validate inputs
      this.validateSendParams(sourceChainId, params);

      // Check chain support
      const isSupported = await this.ccipManager.isChainSupported(
        sourceChainId,
        params.destinationChainId
      );

      if (!isSupported) {
        throw new Error(
          `Chain ${params.destinationChainId} not supported from ${sourceChainId}`
        );
      }

      // Prepare CCIP message
      const gasLimit = params.gasLimit || 
        await this.ccipManager.estimateGasLimit(
          params.destinationChainId,
          params.data
        );

      const ccipMessage: CCIPMessage = {
        receiver: params.receiver,
        data: params.data,
        tokenAmounts: params.tokenTransfers?.map(transfer => ({
          token: transfer.token,
          amount: transfer.amount
        })) || [],
        feeToken: params.feeToken || ethers.ZeroAddress,
        extraArgs: CCIPManager.createExtraArgs(gasLimit)
      };

      // Send message
      const messageId = await this.ccipManager.sendMessage(
        sourceChainId,
        params.destinationChainId,
        ccipMessage
      );

      // Track message status
      this.messageStatuses.set(messageId, {
        messageId,
        status: 'pending',
        sourceChain: sourceChainId,
        destinationChain: params.destinationChainId,
        timestamp: Date.now()
      });

      this.logger.info(`Message sent with ID: ${messageId}`, {
        sourceChain: sourceChainId,
        destinationChain: params.destinationChainId,
        receiver: params.receiver
      });

      return messageId;

    } catch (error) {
      this.logger.error('Failed to send message', {
        sourceChain: sourceChainId,
        destinationChain: params.destinationChainId,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Send cross-chain token transfer
   */
  async sendTokens(
    sourceChainId: number,
    destinationChainId: number,
    receiver: string,
    tokenTransfers: TokenTransfer[],
    feeToken?: string
  ): Promise<string> {
    // Validate token transfers
    if (!tokenTransfers || tokenTransfers.length === 0) {
      throw new Error('Token transfers cannot be empty');
    }

    // Check supported tokens
    const supportedTokens = await this.ccipManager.getSupportedTokens(
      sourceChainId,
      destinationChainId
    );

    for (const transfer of tokenTransfers) {
      if (!supportedTokens.includes(transfer.token)) {
        throw new Error(`Token ${transfer.token} not supported for cross-chain transfer`);
      }
    }

    return await this.sendMessage(sourceChainId, {
      destinationChainId,
      receiver,
      data: '0x', // Empty data for token-only transfers
      tokenTransfers,
      feeToken
    });
  }

  /**
   * Send cross-chain function call
   */
  async sendFunctionCall(
    sourceChainId: number,
    destinationChainId: number,
    targetContract: string,
    functionData: string,
    tokenTransfers?: TokenTransfer[],
    gasLimit?: number
  ): Promise<string> {
    return await this.sendMessage(sourceChainId, {
      destinationChainId,
      receiver: targetContract,
      data: functionData,
      tokenTransfers,
      gasLimit
    });
  }

  /**
   * Get estimated fee for message
   */
  async getEstimatedFee(
    sourceChainId: number,
    params: SendMessageParams
  ): Promise<bigint> {
    const gasLimit = params.gasLimit || 
      await this.ccipManager.estimateGasLimit(
        params.destinationChainId,
        params.data
      );

    const ccipMessage: CCIPMessage = {
      receiver: params.receiver,
      data: params.data,
      tokenAmounts: params.tokenTransfers?.map(transfer => ({
        token: transfer.token,
        amount: transfer.amount
      })) || [],
      feeToken: params.feeToken || ethers.ZeroAddress,
      extraArgs: CCIPManager.createExtraArgs(gasLimit)
    };

    return await this.ccipManager.getFee(
      sourceChainId,
      params.destinationChainId,
      ccipMessage
    );
  }

  /**
   * Get message status
   */
  getMessageStatus(messageId: string): MessageStatus | undefined {
    return this.messageStatuses.get(messageId);
  }

  /**
   * Update message status
   */
  updateMessageStatus(
    messageId: string,
    status: 'confirmed' | 'failed',
    txHash?: string,
    error?: string
  ): void {
    const messageStatus = this.messageStatuses.get(messageId);
    if (messageStatus) {
      messageStatus.status = status;
      messageStatus.txHash = txHash;
      messageStatus.error = error;
      
      this.logger.info(`Message status updated: ${messageId} -> ${status}`);
    }
  }

  /**
   * Validate send parameters
   */
  private validateSendParams(sourceChainId: number, params: SendMessageParams): void {
    if (!ethers.isAddress(params.receiver)) {
      throw new Error('Invalid receiver address');
    }

    if (!params.data || !ethers.isHexString(params.data)) {
      throw new Error('Invalid data format');
    }

    if (sourceChainId === params.destinationChainId) {
      throw new Error('Source and destination chains cannot be the same');
    }

    if (params.tokenTransfers) {
      for (const transfer of params.tokenTransfers) {
        if (!ethers.isAddress(transfer.token)) {
          throw new Error(`Invalid token address: ${transfer.token}`);
        }
        if (transfer.amount <= 0n) {
          throw new Error('Token amount must be greater than 0');
        }
      }
    }

    if (params.feeToken && !ethers.isAddress(params.feeToken) && params.feeToken !== ethers.ZeroAddress) {
      throw new Error('Invalid fee token address');
    }
  }

  /**
   * Batch send messages
   */
  async batchSendMessages(
    sourceChainId: number,
    messages: SendMessageParams[]
  ): Promise<string[]> {
    const messageIds: string[] = [];
    
    for (const message of messages) {
      try {
        const messageId = await this.sendMessage(sourceChainId, message);
        messageIds.push(messageId);
      } catch (error) {
        this.logger.error(`Failed to send batch message`, {
          message,
          error: error instanceof Error ? error.message : String(error)
        });
        throw error;
      }
    }

    this.logger.info(`Batch sent ${messageIds.length} messages from chain ${sourceChainId}`);
    return messageIds;
  }

  /**
   * Get pending messages
   */
  getPendingMessages(): MessageStatus[] {
    return Array.from(this.messageStatuses.values())
      .filter(status => status.status === 'pending');
  }

  /**
   * Clean up old message statuses
   */
  cleanupOldMessages(maxAgeHours: number = 24): void {
    const cutoffTime = Date.now() - (maxAgeHours * 60 * 60 * 1000);
    
    for (const [messageId, status] of this.messageStatuses.entries()) {
      if (status.timestamp < cutoffTime && status.status !== 'pending') {
        this.messageStatuses.delete(messageId);
      }
    }
  }
}