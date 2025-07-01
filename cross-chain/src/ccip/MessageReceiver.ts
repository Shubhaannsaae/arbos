import { ethers, Contract, Provider } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// CCIP Receiver ABI
const CCIP_RECEIVER_ABI = [
  'event CCIPReceive((bytes32 messageId, uint64 sourceChainSelector, bytes sender, bytes data, (address token, uint256 amount)[] destTokenAmounts))',
  'function ccipReceive((bytes32 messageId, uint64 sourceChainSelector, bytes sender, bytes data, (address token, uint256 amount)[] destTokenAmounts) message) external'
];

export interface ReceivedMessage {
  messageId: string;
  sourceChainSelector: bigint;
  sender: string;
  data: string;
  tokenAmounts: { token: string; amount: bigint }[];
  blockNumber: number;
  txHash: string;
  timestamp: number;
}

export interface MessageHandler {
  (message: ReceivedMessage): Promise<void>;
}

export class MessageReceiver {
  private logger: Logger;
  private providers: Map<number, Provider> = new Map();
  private receivers: Map<number, Contract> = new Map();
  private handlers: Map<string, MessageHandler> = new Map();
  private isListening: Map<number, boolean> = new Map();

  // Chain selector to chain ID mapping
  private readonly SELECTOR_TO_CHAIN: { [selector: string]: number } = {
    '5009297550715157269': 1,     // Ethereum
    '6433500567565415381': 43114, // Avalanche
    '4051577828743386545': 137,   // Polygon
    '4949039107694359620': 42161  // Arbitrum
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
        new transports.File({ filename: 'logs/message-receiver.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Initialize message receiver for a chain
   */
  async initialize(
    chainId: number,
    provider: Provider,
    receiverAddress: string
  ): Promise<void> {
    this.providers.set(chainId, provider);
    
    const receiver = new Contract(receiverAddress, CCIP_RECEIVER_ABI, provider);
    this.receivers.set(chainId, receiver);

    this.logger.info(`Initialized message receiver for chain ${chainId}: ${receiverAddress}`);
  }

  /**
   * Start listening for messages on a chain
   */
  async startListening(chainId: number): Promise<void> {
    const receiver = this.receivers.get(chainId);
    if (!receiver) {
      throw new Error(`Receiver not initialized for chain ${chainId}`);
    }

    if (this.isListening.get(chainId)) {
      this.logger.warn(`Already listening on chain ${chainId}`);
      return;
    }

    // Listen for CCIPReceive events
    receiver.on('CCIPReceive', async (messageData, event) => {
      try {
        const message = await this.parseReceivedMessage(messageData, event);
        await this.handleReceivedMessage(message);
      } catch (error) {
        this.logger.error(`Error handling received message on chain ${chainId}`, {
          error: error instanceof Error ? error.message : String(error),
          event
        });
      }
    });

    this.isListening.set(chainId, true);
    this.logger.info(`Started listening for messages on chain ${chainId}`);
  }

  /**
   * Stop listening for messages on a chain
   */
  async stopListening(chainId: number): Promise<void> {
    const receiver = this.receivers.get(chainId);
    if (receiver) {
      receiver.removeAllListeners('CCIPReceive');
      this.isListening.set(chainId, false);
      this.logger.info(`Stopped listening for messages on chain ${chainId}`);
    }
  }

  /**
   * Register message handler
   */
  registerHandler(messageType: string, handler: MessageHandler): void {
    this.handlers.set(messageType, handler);
    this.logger.info(`Registered handler for message type: ${messageType}`);
  }

  /**
   * Unregister message handler
   */
  unregisterHandler(messageType: string): void {
    this.handlers.delete(messageType);
    this.logger.info(`Unregistered handler for message type: ${messageType}`);
  }

  /**
   * Parse received message from event
   */
  private async parseReceivedMessage(
    messageData: any,
    event: any
  ): Promise<ReceivedMessage> {
    const block = await event.getBlock();
    
    return {
      messageId: messageData.messageId,
      sourceChainSelector: messageData.sourceChainSelector,
      sender: ethers.hexlify(messageData.sender),
      data: ethers.hexlify(messageData.data),
      tokenAmounts: messageData.destTokenAmounts.map((ta: any) => ({
        token: ta.token,
        amount: ta.amount
      })),
      blockNumber: event.blockNumber,
      txHash: event.transactionHash,
      timestamp: block.timestamp
    };
  }

  /**
   * Handle received message
   */
  private async handleReceivedMessage(message: ReceivedMessage): Promise<void> {
    this.logger.info(`Received CCIP message: ${message.messageId}`, {
      sourceChain: this.getChainIdFromSelector(message.sourceChainSelector),
      sender: message.sender,
      dataLength: ethers.getBytes(message.data).length,
      tokenCount: message.tokenAmounts.length
    });

    // Extract message type from data (first 4 bytes as function selector)
    const messageType = message.data.slice(0, 10); // '0x' + 4 bytes

    // Find appropriate handler
    const handler = this.handlers.get(messageType);
    if (handler) {
      try {
        await handler(message);
        this.logger.info(`Successfully handled message ${message.messageId} with type ${messageType}`);
      } catch (error) {
        this.logger.error(`Handler failed for message ${message.messageId}`, {
          messageType,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    } else {
      this.logger.warn(`No handler found for message type: ${messageType}`, {
        messageId: message.messageId
      });
    }
  }

  /**
   * Get historical messages
   */
  async getHistoricalMessages(
    chainId: number,
    fromBlock: number,
    toBlock: number = 'latest' as any
  ): Promise<ReceivedMessage[]> {
    const receiver = this.receivers.get(chainId);
    if (!receiver) {
      throw new Error(`Receiver not initialized for chain ${chainId}`);
    }

    const filter = receiver.filters.CCIPReceive();
    const events = await receiver.queryFilter(filter, fromBlock, toBlock);

    const messages: ReceivedMessage[] = [];
    
    for (const event of events) {
      try {
        const message = await this.parseReceivedMessage(event.args, event);
        messages.push(message);
      } catch (error) {
        this.logger.error(`Error parsing historical message`, {
          chainId,
          blockNumber: event.blockNumber,
          txHash: event.transactionHash,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    this.logger.info(`Retrieved ${messages.length} historical messages for chain ${chainId}`, {
      fromBlock,
      toBlock
    });

    return messages;
  }

  /**
   * Convert chain selector to chain ID
   */
  private getChainIdFromSelector(selector: bigint): number {
    const selectorStr = selector.toString();
    return this.SELECTOR_TO_CHAIN[selectorStr] || 0;
  }

  /**
   * Check if listening on chain
   */
  isListeningOnChain(chainId: number): boolean {
    return this.isListening.get(chainId) || false;
  }

  /**
   * Get all listening chains
   */
  getListeningChains(): number[] {
    return Array.from(this.isListening.entries())
      .filter(([_, isListening]) => isListening)
      .map(([chainId, _]) => chainId);
  }

  /**
   * Process token transfers from message
   */
  async processTokenTransfers(message: ReceivedMessage): Promise<void> {
    if (message.tokenAmounts.length === 0) {
      return;
    }

    this.logger.info(`Processing ${message.tokenAmounts.length} token transfers`, {
      messageId: message.messageId
    });

    for (const tokenAmount of message.tokenAmounts) {
      try {
        // Here you would implement actual token handling logic
        // For example, updating balances, triggering events, etc.
        
        this.logger.info(`Processed token transfer`, {
          messageId: message.messageId,
          token: tokenAmount.token,
          amount: tokenAmount.amount.toString()
        });
      } catch (error) {
        this.logger.error(`Failed to process token transfer`, {
          messageId: message.messageId,
          token: tokenAmount.token,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  /**
   * Decode function call data
   */
  decodeFunctionCall(data: string, abi: any[]): { name: string; args: any[] } | null {
    try {
      const iface = new ethers.Interface(abi);
      const decoded = iface.parseTransaction({ data });
      
      return {
        name: decoded?.name || '',
        args: decoded?.args ? Array.from(decoded.args) : []
      };
    } catch (error) {
      this.logger.debug(`Failed to decode function call data`, {
        data,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  /**
   * Validate message authenticity
   */
  async validateMessage(message: ReceivedMessage): Promise<boolean> {
    try {
      // Verify the message came from a valid CCIP router
      const chainId = this.getChainIdFromSelector(message.sourceChainSelector);
      if (chainId === 0) {
        this.logger.warn(`Unknown source chain selector: ${message.sourceChainSelector}`);
        return false;
      }

      // Additional validation logic can be added here
      // such as checking sender authorization, message format, etc.

      return true;
    } catch (error) {
      this.logger.error(`Message validation failed`, {
        messageId: message.messageId,
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }
}
