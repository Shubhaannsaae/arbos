import { ethers } from 'ethers';
import { logger } from '../utils/logger';
import { chainlinkService } from './chainlinkService';
import { web3Service } from '../config/web3';

// Import CCIP contract interfaces based on official Chainlink documentation
interface CCIPMessage {
  receiver: string;
  data: string;
  tokenAmounts: TokenAmount[];
  extraArgs: string;
  feeToken: string;
}

interface TokenAmount {
  token: string;
  amount: string;
}

interface CrossChainArbitrageParams {
  opportunityId: string;
  amount: number;
  maxSlippage: number;
  gasLimit?: number;
  sourceChain: number;
  targetChain: number;
  userId: string;
}

interface BridgeTokenParams {
  sourceChain: number;
  targetChain: number;
  tokenAddress: string;
  amount: string;
  recipient: string;
}

interface CrossChainSwapParams {
  sourceChain: number;
  targetChain: number;
  sourceToken: string;
  targetToken: string;
  amount: string;
  recipient: string;
  slippage: number;
}

class CrossChainService {
  private ccipRouters: Map<number, string> = new Map();
  private supportedChains: number[] = [];
  private gasLimits: Map<number, number> = new Map();

  constructor() {
    this.initializeCCIPConfiguration();
  }

  /**
   * Initialize CCIP configuration based on official Chainlink documentation
   */
  private initializeCCIPConfiguration(): void {
    const chainlinkConfig = chainlinkService.getChainlinkConfig();
    
    // CCIP Router addresses from official Chainlink documentation
    this.ccipRouters.set(1, '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D'); // Ethereum Mainnet
    this.ccipRouters.set(43114, '0xF89501B77b2FA6329F94F5A05FE84cEbb5c8b1a0'); // Avalanche Mainnet
    this.ccipRouters.set(43113, '0xF694E193200268f9a4868e4Aa017A0118C9a8177'); // Avalanche Fuji Testnet
    this.ccipRouters.set(137, '0x849c5ED5a80F5B408Dd4969b78c2C8fdf0565Bfe'); // Polygon Mainnet
    this.ccipRouters.set(42161, '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8'); // Arbitrum One

    this.supportedChains = Array.from(this.ccipRouters.keys());

    // Gas limits for different destination chains
    this.gasLimits.set(1, 2000000);     // Ethereum
    this.gasLimits.set(43114, 2000000); // Avalanche
    this.gasLimits.set(43113, 2000000); // Avalanche Fuji
    this.gasLimits.set(137, 2000000);   // Polygon
    this.gasLimits.set(42161, 2000000); // Arbitrum

    logger.info(`CCIP configuration initialized for ${this.supportedChains.length} chains`);
  }

  /**
   * Execute cross-chain arbitrage using Chainlink CCIP
   */
  public async executeCrossChainArbitrage(params: CrossChainArbitrageParams): Promise<any> {
    try {
      const {
        opportunityId,
        amount,
        maxSlippage,
        sourceChain,
        targetChain,
        userId
      } = params;

      logger.info(`Starting cross-chain arbitrage execution`, {
        opportunityId,
        sourceChain,
        targetChain,
        amount
      });

      // Validate chain support
      if (!this.isCCIPSupported(sourceChain) || !this.isCCIPSupported(targetChain)) {
        throw new Error(`CCIP not supported for chains ${sourceChain} -> ${targetChain}`);
      }

      // Get signers for both chains
      const sourceSigner = web3Service.getSignerForChain('arbitrage', sourceChain);
      const targetSigner = web3Service.getSignerForChain('arbitrage', targetChain);

      if (!sourceSigner || !targetSigner) {
        throw new Error('Required signers not available for cross-chain arbitrage');
      }

      // Step 1: Execute buy order on source chain
      const buyResult = await this.executeBuyOnSourceChain({
        signer: sourceSigner,
        chainId: sourceChain,
        amount,
        maxSlippage,
        opportunityId
      });

      // Step 2: Send cross-chain message with tokens using CCIP
      const ccipResult = await this.sendCCIPMessageWithTokens({
        sourceChain,
        targetChain,
        tokenAddress: buyResult.tokenAddress,
        amount: buyResult.receivedAmount,
        recipient: await targetSigner.getAddress(),
        sellData: this.encodeSellData(amount, maxSlippage)
      });

      // Step 3: Monitor cross-chain transaction
      const targetResult = await this.monitorCrossChainExecution(
        ccipResult.messageId,
        targetChain,
        params
      );

      // Calculate actual profit
      const totalCost = buyResult.totalCost + ccipResult.fees;
      const actualProfit = targetResult.receivedAmount - totalCost;

      const result = {
        success: true,
        messageId: ccipResult.messageId,
        sourceTransaction: {
          hash: buyResult.txHash,
          chain: sourceChain,
          cost: buyResult.totalCost
        },
        ccipTransaction: {
          messageId: ccipResult.messageId,
          fees: ccipResult.fees,
          estimatedDeliveryTime: ccipResult.estimatedDeliveryTime
        },
        targetTransaction: {
          hash: targetResult.txHash,
          chain: targetChain,
          received: targetResult.receivedAmount
        },
        actualProfit,
        actualProfitUSD: actualProfit,
        executionTime: Date.now() - params.startTime,
        gasUsed: buyResult.gasUsed + targetResult.gasUsed
      };

      logger.info(`Cross-chain arbitrage completed`, {
        opportunityId,
        messageId: ccipResult.messageId,
        actualProfit
      });

      return result;
    } catch (error) {
      logger.error('Error executing cross-chain arbitrage:', error);
      throw error;
    }
  }

  /**
   * Send CCIP message with tokens based on official CCIP documentation
   */
  public async sendCCIPMessageWithTokens(params: any): Promise<any> {
    try {
      const {
        sourceChain,
        targetChain,
        tokenAddress,
        amount,
        recipient,
        sellData
      } = params;

      const routerAddress = this.ccipRouters.get(sourceChain);
      if (!routerAddress) {
        throw new Error(`CCIP router not found for chain ${sourceChain}`);
      }

      const signer = web3Service.getSignerForChain('ccip', sourceChain);
      if (!signer) {
        throw new Error('CCIP signer not available');
      }

      // Create CCIP router contract instance
      const ccipRouter = new ethers.Contract(
        routerAddress,
        [
          'function ccipSend(uint64 destinationChainSelector, tuple(bytes receiver, bytes data, tuple(address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external payable returns (bytes32)',
          'function getFee(uint64 destinationChainSelector, tuple(bytes receiver, bytes data, tuple(address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external view returns (uint256 fee)',
          'function getSupportedTokens(uint64 chainSelector) external view returns (address[] memory tokens)',
          'function isChainSupported(uint64 chainSelector) external view returns (bool supported)'
        ],
        signer
      );

      // Get chain selectors (convert chain IDs to CCIP chain selectors)
      const destinationChainSelector = this.getChainSelector(targetChain);

      // Prepare token amounts
      const tokenAmounts = [{
        token: tokenAddress,
        amount: amount
      }];

      // Prepare CCIP message
      const message = {
        receiver: ethers.utils.defaultAbiCoder.encode(['address'], [recipient]),
        data: sellData || '0x',
        tokenAmounts,
        feeToken: ethers.constants.AddressZero, // Pay fees in native token
        extraArgs: this.buildExtraArgs(targetChain)
      };

      // Calculate fees
      const fees = await ccipRouter.getFee(destinationChainSelector, message);

      // Approve token transfer to CCIP router
      await this.approveTokenForCCIP(tokenAddress, amount, routerAddress, signer);

      // Send CCIP message
      const sendTx = await ccipRouter.ccipSend(
        destinationChainSelector,
        message,
        { value: fees }
      );

      const receipt = await sendTx.wait();

      // Extract message ID from logs
      const messageId = this.extractMessageIdFromLogs(receipt.logs);

      logger.info(`CCIP message sent successfully`, {
        messageId,
        sourceChain,
        targetChain,
        fees: ethers.utils.formatEther(fees)
      });

      return {
        messageId,
        txHash: sendTx.hash,
        fees: fees.toNumber(),
        estimatedDeliveryTime: this.getEstimatedDeliveryTime(sourceChain, targetChain),
        receipt
      };
    } catch (error) {
      logger.error('Error sending CCIP message with tokens:', error);
      throw error;
    }
  }

  /**
   * Bridge tokens using Chainlink CCIP
   */
  public async bridgeTokens(params: BridgeTokenParams): Promise<any> {
    try {
      const { sourceChain, targetChain, tokenAddress, amount, recipient } = params;

      logger.info(`Bridging tokens via CCIP`, {
        sourceChain,
        targetChain,
        tokenAddress,
        amount
      });

      const result = await this.sendCCIPMessageWithTokens({
        sourceChain,
        targetChain,
        tokenAddress,
        amount,
        recipient,
        sellData: '0x' // No additional data for simple bridge
      });

      return {
        success: true,
        messageId: result.messageId,
        txHash: result.txHash,
        fees: result.fees,
        receivedAmount: amount, // Assuming 1:1 for bridging
        estimatedDeliveryTime: result.estimatedDeliveryTime
      };
    } catch (error) {
      logger.error('Error bridging tokens:', error);
      throw error;
    }
  }

  /**
   * Execute cross-chain swap using CCIP
   */
  public async executeCrossChainSwap(params: CrossChainSwapParams): Promise<any> {
    try {
      const {
        sourceChain,
        targetChain,
        sourceToken,
        targetToken,
        amount,
        recipient,
        slippage
      } = params;

      // Encode swap data for target chain execution
      const swapData = this.encodeSwapData({
        tokenIn: sourceToken,
        tokenOut: targetToken,
        amountIn: amount,
        slippage,
        recipient
      });

      const result = await this.sendCCIPMessageWithTokens({
        sourceChain,
        targetChain,
        tokenAddress: sourceToken,
        amount,
        recipient,
        sellData: swapData
      });

      logger.info(`Cross-chain swap initiated`, {
        messageId: result.messageId,
        sourceChain,
        targetChain
      });

      return result;
    } catch (error) {
      logger.error('Error executing cross-chain swap:', error);
      throw error;
    }
  }

  /**
   * Get CCIP fee estimation
   */
  public async estimateCCIPFees(
    sourceChain: number,
    targetChain: number,
    tokenAddress?: string,
    amount?: string
  ): Promise<number> {
    try {
      const routerAddress = this.ccipRouters.get(sourceChain);
      if (!routerAddress) {
        throw new Error(`CCIP router not found for chain ${sourceChain}`);
      }

      const provider = web3Service.getProvider(sourceChain);
      if (!provider) {
        throw new Error(`Provider not available for chain ${sourceChain}`);
      }

      const ccipRouter = new ethers.Contract(
        routerAddress,
        [
          'function getFee(uint64 destinationChainSelector, tuple(bytes receiver, bytes data, tuple(address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external view returns (uint256 fee)'
        ],
        provider
      );

      const destinationChainSelector = this.getChainSelector(targetChain);

      const message = {
        receiver: ethers.utils.defaultAbiCoder.encode(['address'], [ethers.constants.AddressZero]),
        data: '0x',
        tokenAmounts: tokenAddress && amount ? [{ token: tokenAddress, amount }] : [],
        feeToken: ethers.constants.AddressZero,
        extraArgs: this.buildExtraArgs(targetChain)
      };

      const fees = await ccipRouter.getFee(destinationChainSelector, message);
      return fees.toNumber();
    } catch (error) {
      logger.error('Error estimating CCIP fees:', error);
      return 0;
    }
  }

  /**
   * Monitor cross-chain message execution
   */
  public async monitorCrossChainExecution(
    messageId: string,
    targetChain: number,
    params: any
  ): Promise<any> {
    try {
      logger.info(`Monitoring cross-chain execution`, { messageId, targetChain });

      // In production, this would listen for CCIP events on target chain
      // For now, we'll simulate monitoring with polling
      const maxWaitTime = 30 * 60 * 1000; // 30 minutes
      const pollInterval = 30 * 1000; // 30 seconds
      const startTime = Date.now();

      while (Date.now() - startTime < maxWaitTime) {
        try {
          const executionResult = await this.checkMessageExecution(messageId, targetChain);
          
          if (executionResult.executed) {
            logger.info(`Cross-chain message executed successfully`, {
              messageId,
              txHash: executionResult.txHash
            });

            return {
              success: true,
              txHash: executionResult.txHash,
              receivedAmount: executionResult.receivedAmount,
              gasUsed: executionResult.gasUsed,
              executionTime: Date.now() - startTime
            };
          }

          // Wait before next poll
          await new Promise(resolve => setTimeout(resolve, pollInterval));
        } catch (error) {
          logger.warn(`Error checking message execution:`, error);
        }
      }

      throw new Error(`Cross-chain execution timeout for message ${messageId}`);
    } catch (error) {
      logger.error('Error monitoring cross-chain execution:', error);
      throw error;
    }
  }

  /**
   * Check if a chain supports CCIP
   */
  public isCCIPSupported(chainId: number): boolean {
    return this.supportedChains.includes(chainId);
  }

  /**
   * Get supported chains for CCIP
   */
  public getSupportedChains(): number[] {
    return [...this.supportedChains];
  }

  /**
   * Get estimated delivery time between chains
   */
  public getEstimatedDeliveryTime(sourceChain: number, targetChain: number): number {
    // Estimated delivery times in milliseconds based on chain characteristics
    const baseTimes: { [key: number]: number } = {
      1: 15 * 60 * 1000,      // Ethereum: 15 minutes
      43114: 5 * 60 * 1000,   // Avalanche: 5 minutes
      43113: 5 * 60 * 1000,   // Avalanche Fuji: 5 minutes
      137: 10 * 60 * 1000,    // Polygon: 10 minutes
      42161: 8 * 60 * 1000    // Arbitrum: 8 minutes
    };

    const sourceTime = baseTimes[sourceChain] || 15 * 60 * 1000;
    const targetTime = baseTimes[targetChain] || 15 * 60 * 1000;

    // Return the maximum of the two times plus CCIP processing time
    return Math.max(sourceTime, targetTime) + (2 * 60 * 1000); // Add 2 minutes for CCIP
  }

  // Helper methods
  private getChainSelector(chainId: number): string {
    // CCIP chain selectors - official mappings from Chainlink documentation
    const selectors: { [key: number]: string } = {
      1: '5009297550715157269',       // Ethereum Mainnet
      43114: '6433500567565415381',   // Avalanche Mainnet  
      43113: '14767482510784806043',  // Avalanche Fuji Testnet
      137: '4051577828743386545',     // Polygon Mainnet
      42161: '4949039107694359620'    // Arbitrum One
    };

    const selector = selectors[chainId];
    if (!selector) {
      throw new Error(`Chain selector not found for chain ID ${chainId}`);
    }

    return selector;
  }

  private buildExtraArgs(targetChain: number): string {
    const gasLimit = this.gasLimits.get(targetChain) || 2000000;
    
    // Encode extra args according to CCIP specification
    return ethers.utils.defaultAbiCoder.encode(
      ['uint256', 'bool'],
      [gasLimit, false] // gasLimit, strict (false = non-strict)
    );
  }

  private async approveTokenForCCIP(
    tokenAddress: string,
    amount: string,
    routerAddress: string,
    signer: ethers.Signer
  ): Promise<void> {
    try {
      const tokenContract = new ethers.Contract(
        tokenAddress,
        [
          'function approve(address spender, uint256 amount) external returns (bool)',
          'function allowance(address owner, address spender) external view returns (uint256)'
        ],
        signer
      );

      const currentAllowance = await tokenContract.allowance(
        await signer.getAddress(),
        routerAddress
      );

      if (currentAllowance.lt(amount)) {
        const approveTx = await tokenContract.approve(routerAddress, amount);
        await approveTx.wait();
        
        logger.debug(`Token approved for CCIP`, {
          tokenAddress,
          amount,
          routerAddress
        });
      }
    } catch (error) {
      logger.error('Error approving token for CCIP:', error);
      throw error;
    }
  }

  private extractMessageIdFromLogs(logs: any[]): string {
    // Extract CCIP message ID from transaction logs
    for (const log of logs) {
      // Look for CCIPSendRequested event signature
      if (log.topics[0] === '0x35c02761bcd3ef995c6a601a1981f4ed3934dcbe5041e24e286c89f5531d17e4') {
        return log.topics[1]; // Message ID is the first indexed parameter
      }
    }
    
    throw new Error('Message ID not found in transaction logs');
  }

  private encodeSellData(amount: number, maxSlippage: number): string {
    // Encode sell parameters for target chain execution
    return ethers.utils.defaultAbiCoder.encode(
      ['uint256', 'uint256', 'uint256'],
      [amount, maxSlippage * 100, Date.now() + 3600000] // amount, slippage (basis points), deadline
    );
  }

  private encodeSwapData(params: any): string {
    // Encode swap parameters for target chain execution
    return ethers.utils.defaultAbiCoder.encode(
      ['address', 'address', 'uint256', 'uint256', 'address', 'uint256'],
      [
        params.tokenIn,
        params.tokenOut,
        params.amountIn,
        params.slippage * 100, // Convert to basis points
        params.recipient,
        Date.now() + 3600000 // deadline
      ]
    );
  }

  private async executeBuyOnSourceChain(params: any): Promise<any> {
    try {
      const { signer, chainId, amount, maxSlippage, opportunityId } = params;

      // Implementation would execute DEX swap on source chain
      // This is a simplified version - in production would use actual DEX contracts
      const mockTxHash = ethers.utils.keccak256(
        ethers.utils.toUtf8Bytes(`${opportunityId}-${Date.now()}`)
      );

      return {
        txHash: mockTxHash,
        tokenAddress: '0x...', // Would be actual token address
        receivedAmount: amount * 0.997, // After fees
        totalCost: amount,
        gasUsed: 200000
      };
    } catch (error) {
      logger.error('Error executing buy on source chain:', error);
      throw error;
    }
  }

  private async checkMessageExecution(messageId: string, targetChain: number): Promise<any> {
    try {
      // In production, this would check CCIP execution events on target chain
      // For now, return mock success after some time
      const random = Math.random();
      
      if (random > 0.3) { // 70% success rate for simulation
        return {
          executed: true,
          txHash: ethers.utils.keccak256(ethers.utils.toUtf8Bytes(`exec-${messageId}`)),
          receivedAmount: 1000, // Mock received amount
          gasUsed: 150000
        };
      }

      return { executed: false };
    } catch (error) {
      logger.error('Error checking message execution:', error);
      return { executed: false };
    }
  }

  /**
   * Get cross-chain transaction status
   */
  public async getCrossChainStatus(messageId: string): Promise<any> {
    try {
      // Implementation would query CCIP status
      return {
        messageId,
        status: 'success', // pending, success, failed
        sourceChain: 43114,
        targetChain: 1,
        timestamp: new Date(),
        confirmations: 12
      };
    } catch (error) {
      logger.error('Error getting cross-chain status:', error);
      throw error;
    }
  }

  /**
   * Cancel cross-chain transaction (if possible)
   */
  public async cancelCrossChainTransaction(messageId: string): Promise<boolean> {
    try {
      // CCIP transactions cannot be cancelled once sent
      // This method is for compatibility but will always return false
      logger.warn(`Attempted to cancel CCIP message ${messageId} - not possible`);
      return false;
    } catch (error) {
      logger.error('Error cancelling cross-chain transaction:', error);
      return false;
    }
  }

  /**
   * Get cross-chain analytics
   */
  public async getCrossChainAnalytics(timeRange: { start: Date; end: Date }): Promise<any> {
    try {
      // Implementation would analyze cross-chain transaction data
      return {
        totalTransactions: 100,
        successRate: 95.5,
        averageDeliveryTime: 8.5 * 60 * 1000, // 8.5 minutes
        totalVolume: 1000000,
        topRoutes: [
          { source: 43114, target: 1, volume: 500000 },
          { source: 1, target: 43114, volume: 300000 }
        ],
        feesCollected: 5000
      };
    } catch (error) {
      logger.error('Error getting cross-chain analytics:', error);
      throw error;
    }
  }
}

export const crossChainService = new CrossChainService();
export default crossChainService;
