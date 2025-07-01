import { ethers, Contract, Provider, Signer } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';

// Polygon Bridge ABI
const POLYGON_BRIDGE_ABI = [
  'function depositFor(address user, address rootToken, bytes calldata depositData) external',
  'function exit(bytes calldata inputData) external',
  'function processedExits(bytes32 exitHash) external view returns (bool)',
  'function withdrawStart(address token, uint256 amount) external',
  'function withdrawExit(bytes32 txHash, uint logIndex, uint blockNumber, bytes32 rootHash, bytes calldata receipt) external'
];

// Polygon RootChainManager ABI
const ROOT_CHAIN_MANAGER_ABI = [
  'function depositEtherFor(address user) external payable',
  'function depositFor(address user, address rootToken, bytes calldata depositData) external',
  'function exit(bytes calldata inputData) external',
  'function mapToken(address rootToken, address childToken, bytes32 tokenType) external'
];

export interface PolygonBridgeConfig {
  rootChainManager: string;
  checkpointManager: string;
  fxRoot: string;
  fxChild: string;
  chainId: number;
}

export interface DepositParams {
  user: string;
  rootToken: string;
  amount: bigint;
  data?: string;
}

export interface WithdrawParams {
  token: string;
  amount: bigint;
  txHash: string;
  logIndex: number;
  blockNumber: number;
  rootHash: string;
  receipt: string;
}

export interface BridgeStatus {
  txHash: string;
  status: 'pending' | 'completed' | 'failed';
  timestamp: number;
  blockNumber: number;
  amount: bigint;
  token: string;
}

export class PolygonBridge {
  private logger: Logger;
  private provider: Provider;
  private signer: Signer;
  private rootChainManager: Contract;
  private config: PolygonBridgeConfig;

  // Official Polygon Bridge addresses
  private readonly MAINNET_CONFIG: PolygonBridgeConfig = {
    rootChainManager: '0xA0c68C638235ee32657e8f720a23ceC1bFc77C77',
    checkpointManager: '0x86E4Dc95c7FBdBf52e33D563BbDB00823894C287',
    fxRoot: '0xfe5e5D361b2ad62c541bAb87C45a0B9B018389a2',
    fxChild: '0x8397259c983751DAf40400790063935a11afa28a',
    chainId: 1
  };

  private readonly TESTNET_CONFIG: PolygonBridgeConfig = {
    rootChainManager: '0xBbD7cBFA79FaBC85B7d1C2e56007C7E4e9d55b66',
    checkpointManager: '0x2890bA17EfE978480615e330ecB65333b880928e',
    fxRoot: '0x3d1d3E34f7fB6D26245E6640E1c50710eFFf15bA',
    fxChild: '0xCf73231F28B7331BBe3124B907840A94851f9f11',
    chainId: 5
  };

  // Supported tokens on Polygon
  private readonly SUPPORTED_TOKENS: { [address: string]: string } = {
    '0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8': 'USDC',
    '0x514910771AF9Ca656af840dff83E8264EcF986CA': 'LINK',
    '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599': 'WBTC',
    '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2': 'WETH'
  };

  constructor(chainId: number, provider: Provider, signer: Signer, isTestnet: boolean = false) {
    this.provider = provider;
    this.signer = signer;
    
    this.config = isTestnet ? this.TESTNET_CONFIG : this.MAINNET_CONFIG;
    this.config.chainId = chainId;

    this.rootChainManager = new Contract(
      this.config.rootChainManager,
      ROOT_CHAIN_MANAGER_ABI,
      provider
    );

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/polygon-bridge.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.logger.info(`PolygonBridge initialized for chain ${chainId}`, {
      isTestnet,
      config: this.config
    });
  }

  /**
   * Deposit tokens from Ethereum to Polygon
   */
  async deposit(params: DepositParams): Promise<BridgeStatus> {
    try {
      // Validate deposit parameters
      this.validateDepositParams(params);

      // Check if token is supported
      if (!this.isTokenSupported(params.rootToken)) {
        throw new Error(`Token ${params.rootToken} not supported`);
      }

      // Encode deposit data
      const depositData = this.encodeDepositData(params.amount);

      // Prepare transaction
      const contractWithSigner = this.rootChainManager.connect(this.signer);
      
      let tx;
      if (params.rootToken === ethers.ZeroAddress) {
        // ETH deposit
        tx = await contractWithSigner.depositEtherFor(params.user, {
          value: params.amount
        });
      } else {
        // ERC20 deposit
        tx = await contractWithSigner.depositFor(
          params.user,
          params.rootToken,
          depositData
        );
      }

      const receipt = await tx.wait();

      const status: BridgeStatus = {
        txHash: receipt.hash,
        status: 'pending',
        timestamp: Date.now(),
        blockNumber: receipt.blockNumber,
        amount: params.amount,
        token: params.rootToken
      };

      this.logger.info(`Deposit initiated`, {
        user: params.user,
        token: params.rootToken,
        amount: params.amount.toString(),
        txHash: receipt.hash
      });

      return status;

    } catch (error) {
      this.logger.error('Deposit failed', {
        params,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Withdraw tokens from Polygon to Ethereum
   */
  async withdraw(params: WithdrawParams): Promise<BridgeStatus> {
    try {
      // Validate withdraw parameters
      this.validateWithdrawParams(params);

      // Build exit data
      const exitData = this.buildExitData(params);

      // Execute exit
      const contractWithSigner = this.rootChainManager.connect(this.signer);
      const tx = await contractWithSigner.exit(exitData);
      const receipt = await tx.wait();

      const status: BridgeStatus = {
        txHash: receipt.hash,
        status: 'completed',
        timestamp: Date.now(),
        blockNumber: receipt.blockNumber,
        amount: params.amount,
        token: params.token
      };

      this.logger.info(`Withdrawal completed`, {
        token: params.token,
        amount: params.amount.toString(),
        txHash: receipt.hash
      });

      return status;

    } catch (error) {
      this.logger.error('Withdrawal failed', {
        params,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Check if exit has been processed
   */
  async isExitProcessed(exitHash: string): Promise<boolean> {
    try {
      // This would require the bridge contract with processedExits mapping
      const bridgeContract = new Contract(
        this.config.rootChainManager,
        POLYGON_BRIDGE_ABI,
        this.provider
      );

      const exitHashBytes = ethers.getBytes(exitHash);
      return await bridgeContract.processedExits(exitHashBytes);

    } catch (error) {
      this.logger.error('Failed to check exit status', {
        exitHash,
        error: error instanceof Error ? error.message : String(error)
      });
      return false;
    }
  }

  /**
   * Get deposit status by transaction hash
   */
  async getDepositStatus(txHash: string): Promise<BridgeStatus | null> {
    try {
      const receipt = await this.provider.getTransactionReceipt(txHash);
      if (!receipt) {
        return null;
      }

      // Parse logs to extract deposit information
      const depositEvent = this.parseDepositEvent(receipt.logs);
      if (!depositEvent) {
        return null;
      }

      return {
        txHash: receipt.hash,
        status: receipt.status === 1 ? 'completed' : 'failed',
        timestamp: Date.now(), // Would need block timestamp
        blockNumber: receipt.blockNumber,
        amount: depositEvent.amount,
        token: depositEvent.token
      };

    } catch (error) {
      this.logger.error('Failed to get deposit status', {
        txHash,
        error: error instanceof Error ? error.message : String(error)
      });
      return null;
    }
  }

  /**
   * Estimate deposit fee
   */
  async estimateDepositFee(params: DepositParams): Promise<bigint> {
    try {
      const depositData = this.encodeDepositData(params.amount);
      
      let gasEstimate;
      if (params.rootToken === ethers.ZeroAddress) {
        gasEstimate = await this.rootChainManager.depositEtherFor.estimateGas(
          params.user,
          { value: params.amount }
        );
      } else {
        gasEstimate = await this.rootChainManager.depositFor.estimateGas(
          params.user,
          params.rootToken,
          depositData
        );
      }

      // Get current gas price
      const gasPrice = await this.provider.getFeeData();
      const fee = gasEstimate * (gasPrice.gasPrice || 0n);

      this.logger.debug(`Deposit fee estimated`, {
        gasEstimate: gasEstimate.toString(),
        gasPrice: gasPrice.gasPrice?.toString(),
        fee: fee.toString()
      });

      return fee;

    } catch (error) {
      this.logger.error('Failed to estimate deposit fee', {
        params,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get supported tokens
   */
  getSupportedTokens(): { [address: string]: string } {
    return { ...this.SUPPORTED_TOKENS };
  }

  /**
   * Check if token is supported
   */
  isTokenSupported(tokenAddress: string): boolean {
    return tokenAddress === ethers.ZeroAddress || tokenAddress in this.SUPPORTED_TOKENS;
  }

  /**
   * Start event listener for bridge events
   */
  startEventListener(callback: (event: any) => void): void {
    // Listen for deposit events
    this.rootChainManager.on('*', (event) => {
      this.logger.debug('Polygon bridge event received', { event });
      callback(event);
    });

    this.logger.info('Started event listener for Polygon bridge');
  }

  /**
   * Stop event listener
   */
  stopEventListener(): void {
    this.rootChainManager.removeAllListeners();
    this.logger.info('Stopped event listener for Polygon bridge');
  }

  /**
   * Encode deposit data for ERC20 tokens
   */
  private encodeDepositData(amount: bigint): string {
    return ethers.solidityPacked(['uint256'], [amount]);
  }

  /**
   * Build exit data for withdrawal
   */
  private buildExitData(params: WithdrawParams): string {
    // Simplified exit data structure
    return ethers.solidityPacked(
      ['bytes32', 'uint256', 'uint256', 'bytes32', 'bytes'],
      [params.txHash, params.logIndex, params.blockNumber, params.rootHash, params.receipt]
    );
  }

  /**
   * Parse deposit event from transaction logs
   */
  private parseDepositEvent(logs: any[]): { amount: bigint; token: string } | null {
    try {
      // Simplified log parsing - in production, use proper ABI decoding
      for (const log of logs) {
        if (log.topics[0] === '0x...') { // Deposit event signature
          return {
            amount: BigInt(log.data),
            token: log.topics[1]
          };
        }
      }
      return null;
    } catch (error) {
      this.logger.error('Failed to parse deposit event', { logs, error });
      return null;
    }
  }

  /**
   * Validate deposit parameters
   */
  private validateDepositParams(params: DepositParams): void {
    if (!ethers.isAddress(params.user)) {
      throw new Error('Invalid user address');
    }

    if (params.rootToken !== ethers.ZeroAddress && !ethers.isAddress(params.rootToken)) {
      throw new Error('Invalid root token address');
    }

    if (params.amount <= 0n) {
      throw new Error('Amount must be greater than 0');
    }
  }

  /**
   * Validate withdraw parameters
   */
  private validateWithdrawParams(params: WithdrawParams): void {
    if (!ethers.isAddress(params.token)) {
      throw new Error('Invalid token address');
    }

    if (params.amount <= 0n) {
      throw new Error('Amount must be greater than 0');
    }

    if (!ethers.isHexString(params.txHash, 32)) {
      throw new Error('Invalid transaction hash');
    }

    if (params.blockNumber <= 0) {
      throw new Error('Invalid block number');
    }

    if (!ethers.isHexString(params.rootHash, 32)) {
      throw new Error('Invalid root hash');
    }

    if (!ethers.isHexString(params.receipt)) {
      throw new Error('Invalid receipt data');
    }
  }

  /**
   * Get bridge configuration
   */
  getConfig(): PolygonBridgeConfig {
    return { ...this.config };
  }
}
