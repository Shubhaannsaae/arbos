import { ethers, Provider, Contract } from 'ethers';
import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import { EventEmitter } from 'events';

export interface BridgeStatus {
  protocol: string;
  sourceChain: number;
  destinationChain: number;
  isHealthy: boolean;
  lastCheck: number;
  responseTime: number;
  errorRate: number;
  liquidity: bigint;
  queueLength: number;
}

export interface HealthMetrics {
  uptime: number;
  averageResponseTime: number;
  successRate: number;
  totalTransactions: number;
  failedTransactions: number;
  lastFailure?: {
    timestamp: number;
    reason: string;
    txHash: string;
  };
}

export interface AlertConfig {
  maxResponseTime: number;
  maxErrorRate: number;
  minLiquidity: bigint;
  maxQueueLength: number;
  checkInterval: number;
}

export class BridgeMonitor extends EventEmitter {
  private logger: Logger;
  private providers: Map<number, Provider> = new Map();
  private bridgeContracts: Map<string, Contract> = new Map();
  private healthStatus: Map<string, BridgeStatus> = new Map();
  private metrics: Map<string, HealthMetrics> = new Map();
  private monitoringIntervals: Map<string, NodeJS.Timeout> = new Map();
  private alertConfig: AlertConfig;

  // Official bridge addresses for monitoring
  private readonly BRIDGE_ADDRESSES = {
    ccip: {
      ethereum: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D',
      avalanche: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',
      polygon: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43',
      arbitrum: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8'
    },
    layerzero: {
      ethereum: '0x66A71Dcef29A0fFBDBE3c6a460a3B5BC225Cd675',
      avalanche: '0x3c2269811836af69497E5F486A85D7316753cf62',
      polygon: '0x3c2269811836af69497E5F486A85D7316753cf62',
      arbitrum: '0x3c2269811836af69497E5F486A85D7316753cf62'
    }
  };

  constructor(providers: Map<number, Provider>, alertConfig?: Partial<AlertConfig>) {
    super();
    this.providers = providers;
    
    this.alertConfig = {
      maxResponseTime: 30000, // 30 seconds
      maxErrorRate: 0.05, // 5%
      minLiquidity: ethers.parseEther('1000'), // 1000 ETH
      maxQueueLength: 100,
      checkInterval: 60000, // 1 minute
      ...alertConfig
    };

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/bridge-monitor.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.initializeBridgeContracts();
  }

  /**
   * Start monitoring all bridges
   */
  async startMonitoring(): Promise<void> {
    this.logger.info('Starting bridge monitoring');

    for (const [bridgeId, status] of this.healthStatus) {
      const intervalId = setInterval(
        () => this.checkBridgeHealth(bridgeId),
        this.alertConfig.checkInterval
      );
      this.monitoringIntervals.set(bridgeId, intervalId);
    }

    // Initial health check
    await this.performFullHealthCheck();
    
    this.emit('monitoring:started');
  }

  /**
   * Stop monitoring all bridges
   */
  stopMonitoring(): void {
    this.logger.info('Stopping bridge monitoring');

    for (const [bridgeId, interval] of this.monitoringIntervals) {
      clearInterval(interval);
    }
    this.monitoringIntervals.clear();
    
    this.emit('monitoring:stopped');
  }

  /**
   * Check health of specific bridge
   */
  async checkBridgeHealth(bridgeId: string): Promise<BridgeStatus> {
    const startTime = Date.now();
    
    try {
      const status = this.healthStatus.get(bridgeId);
      if (!status) {
        throw new Error(`Bridge ${bridgeId} not found`);
      }

      // Perform health checks
      const healthChecks = await Promise.allSettled([
        this.checkBridgeConnectivity(bridgeId),
        this.checkBridgeLiquidity(bridgeId),
        this.checkBridgeQueue(bridgeId),
        this.checkRecentTransactions(bridgeId)
      ]);

      const responseTime = Date.now() - startTime;
      const isHealthy = healthChecks.every(result => result.status === 'fulfilled');

      // Update status
      const updatedStatus: BridgeStatus = {
        ...status,
        isHealthy,
        lastCheck: Date.now(),
        responseTime,
        errorRate: await this.calculateErrorRate(bridgeId)
      };

      this.healthStatus.set(bridgeId, updatedStatus);
      
      // Check for alerts
      await this.checkAlerts(bridgeId, updatedStatus);

      this.logger.debug(`Health check completed for ${bridgeId}`, {
        isHealthy,
        responseTime,
        errorRate: updatedStatus.errorRate
      });

      return updatedStatus;

    } catch (error) {
      this.logger.error(`Health check failed for ${bridgeId}`, {
        error: error instanceof Error ? error.message : String(error)
      });

      // Mark as unhealthy
      const status = this.healthStatus.get(bridgeId);
      if (status) {
        status.isHealthy = false;
        status.lastCheck = Date.now();
        status.responseTime = Date.now() - startTime;
        this.healthStatus.set(bridgeId, status);
      }

      throw error;
    }
  }

  /**
   * Get current health status for all bridges
   */
  getHealthStatus(): Map<string, BridgeStatus> {
    return new Map(this.healthStatus);
  }

  /**
   * Get health metrics for specific bridge
   */
  getHealthMetrics(bridgeId: string): HealthMetrics | undefined {
    return this.metrics.get(bridgeId);
  }

  /**
   * Get overall system health
   */
  getSystemHealth(): {
    totalBridges: number;
    healthyBridges: number;
    systemUptime: number;
    averageResponseTime: number;
    criticalAlerts: number;
  } {
    const bridges = Array.from(this.healthStatus.values());
    const healthyBridges = bridges.filter(b => b.isHealthy).length;
    const averageResponseTime = bridges.reduce((sum, b) => sum + b.responseTime, 0) / bridges.length;

    return {
      totalBridges: bridges.length,
      healthyBridges,
      systemUptime: this.calculateSystemUptime(),
      averageResponseTime: Math.round(averageResponseTime),
      criticalAlerts: this.getCriticalAlertCount()
    };
  }

  /**
   * Subscribe to bridge events
   */
  subscribeToBridgeEvents(bridgeId: string): void {
    const contract = this.bridgeContracts.get(bridgeId);
    if (!contract) {
      throw new Error(`Bridge contract ${bridgeId} not found`);
    }

    // Listen for relevant bridge events
    contract.on('*', (event) => {
      this.handleBridgeEvent(bridgeId, event);
    });

    this.logger.info(`Subscribed to events for bridge ${bridgeId}`);
  }

  /**
   * Initialize bridge contracts for monitoring
   */
  private initializeBridgeContracts(): void {
    const bridgeABI = [
      'function getOutboundNonce() external view returns (uint256)',
      'function getInboundNonce() external view returns (uint256)',
      'function isChainSupported(uint64 chainSelector) external view returns (bool)',
      'function getPoolBySourceToken(address sourceToken) external view returns (address)',
      'event CCIPSendRequested((bytes32 messageId, uint64 sourceChainSelector, address sender, bytes receiver, uint64 sequenceNumber, uint256 gasLimit) message)',
      'event ExecutionStateChanged(bytes32 indexed messageId, uint8 indexed state, bytes returnData)'
    ];

    // Initialize CCIP routers
    for (const [network, address] of Object.entries(this.BRIDGE_ADDRESSES.ccip)) {
      const chainId = this.getChainIdByNetwork(network);
      const provider = this.providers.get(chainId);
      
      if (provider) {
        const contract = new Contract(address, bridgeABI, provider);
        const bridgeId = `ccip-${network}`;
        
        this.bridgeContracts.set(bridgeId, contract);
        this.healthStatus.set(bridgeId, {
          protocol: 'ccip',
          sourceChain: chainId,
          destinationChain: 0, // Multiple destinations
          isHealthy: true,
          lastCheck: 0,
          responseTime: 0,
          errorRate: 0,
          liquidity: 0n,
          queueLength: 0
        });
        
        this.metrics.set(bridgeId, {
          uptime: 100,
          averageResponseTime: 0,
          successRate: 100,
          totalTransactions: 0,
          failedTransactions: 0
        });
      }
    }

    this.logger.info(`Initialized ${this.bridgeContracts.size} bridge contracts for monitoring`);
  }

  /**
   * Check bridge connectivity
   */
  private async checkBridgeConnectivity(bridgeId: string): Promise<boolean> {
    const contract = this.bridgeContracts.get(bridgeId);
    if (!contract) {
      throw new Error(`Contract not found for ${bridgeId}`);
    }

    try {
      // Simple connectivity check - call a view function
      await contract.getOutboundNonce();
      return true;
    } catch (error) {
      this.logger.warn(`Connectivity check failed for ${bridgeId}`, { error });
      return false;
    }
  }

  /**
   * Check bridge liquidity
   */
  private async checkBridgeLiquidity(bridgeId: string): Promise<bigint> {
    try {
      // Get ETH balance of bridge contract
      const contract = this.bridgeContracts.get(bridgeId);
      if (!contract) {
        return 0n;
      }

      const provider = contract.provider;
      const balance = await provider.getBalance(await contract.getAddress());
      
      // Update status
      const status = this.healthStatus.get(bridgeId);
      if (status) {
        status.liquidity = balance;
      }

      return balance;
    } catch (error) {
      this.logger.warn(`Liquidity check failed for ${bridgeId}`, { error });
      return 0n;
    }
  }

  /**
   * Check bridge transaction queue
   */
  private async checkBridgeQueue(bridgeId: string): Promise<number> {
    try {
      const contract = this.bridgeContracts.get(bridgeId);
      if (!contract) {
        return 0;
      }

      // Get nonce difference as queue proxy
      const outbound = await contract.getOutboundNonce();
      const inbound = await contract.getInboundNonce();
      const queueLength = Number(outbound - inbound);

      // Update status
      const status = this.healthStatus.get(bridgeId);
      if (status) {
        status.queueLength = Math.max(0, queueLength);
      }

      return Math.max(0, queueLength);
    } catch (error) {
      this.logger.warn(`Queue check failed for ${bridgeId}`, { error });
      return 0;
    }
  }

  /**
   * Check recent transactions for errors
   */
  private async checkRecentTransactions(bridgeId: string): Promise<boolean> {
    try {
      const contract = this.bridgeContracts.get(bridgeId);
      if (!contract) {
        return true;
      }

      // Query recent events to check for failures
      const currentBlock = await contract.provider.getBlockNumber();
      const fromBlock = Math.max(currentBlock - 1000, 0); // Last 1000 blocks

      const events = await contract.queryFilter('*', fromBlock, currentBlock);
      
      // Analyze events for failures
      const failureEvents = events.filter(event => 
        event.eventName === 'ExecutionStateChanged' && 
        event.args && 
        event.args[1] === 3 // Failure state
      );

      const errorRate = events.length > 0 ? failureEvents.length / events.length : 0;
      
      // Update metrics
      this.updateMetrics(bridgeId, events.length, failureEvents.length);

      return errorRate < this.alertConfig.maxErrorRate;
    } catch (error) {
      this.logger.warn(`Recent transactions check failed for ${bridgeId}`, { error });
      return true;
    }
  }

  /**
   * Calculate error rate for bridge
   */
  private async calculateErrorRate(bridgeId: string): Promise<number> {
    const metrics = this.metrics.get(bridgeId);
    if (!metrics || metrics.totalTransactions === 0) {
      return 0;
    }

    return metrics.failedTransactions / metrics.totalTransactions;
  }

  /**
   * Check for alert conditions
   */
  private async checkAlerts(bridgeId: string, status: BridgeStatus): Promise<void> {
    const alerts: string[] = [];

    // Response time alert
    if (status.responseTime > this.alertConfig.maxResponseTime) {
      alerts.push(`High response time: ${status.responseTime}ms`);
    }

    // Error rate alert
    if (status.errorRate > this.alertConfig.maxErrorRate) {
      alerts.push(`High error rate: ${(status.errorRate * 100).toFixed(2)}%`);
    }

    // Liquidity alert
    if (status.liquidity < this.alertConfig.minLiquidity) {
      alerts.push(`Low liquidity: ${ethers.formatEther(status.liquidity)} ETH`);
    }

    // Queue length alert
    if (status.queueLength > this.alertConfig.maxQueueLength) {
      alerts.push(`High queue length: ${status.queueLength}`);
    }

    // Health status alert
    if (!status.isHealthy) {
      alerts.push('Bridge is unhealthy');
    }

    // Emit alerts
    if (alerts.length > 0) {
      this.emit('bridge:alert', {
        bridgeId,
        alerts,
        status,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Handle bridge events
   */
  private handleBridgeEvent(bridgeId: string, event: any): void {
    this.emit('bridge:event', {
      bridgeId,
      event,
      timestamp: Date.now()
    });

    // Update metrics based on event type
    if (event.eventName === 'CCIPSendRequested') {
      this.incrementTransactionCount(bridgeId);
    } else if (event.eventName === 'ExecutionStateChanged') {
      if (event.args && event.args[1] === 3) { // Failure state
        this.incrementFailureCount(bridgeId);
      }
    }
  }

  /**
   * Update metrics for bridge
   */
  private updateMetrics(bridgeId: string, totalTx: number, failedTx: number): void {
    const metrics = this.metrics.get(bridgeId);
    if (metrics) {
      metrics.totalTransactions += totalTx;
      metrics.failedTransactions += failedTx;
      metrics.successRate = metrics.totalTransactions > 0 
        ? ((metrics.totalTransactions - metrics.failedTransactions) / metrics.totalTransactions) * 100
        : 100;
    }
  }

  /**
   * Increment transaction count
   */
  private incrementTransactionCount(bridgeId: string): void {
    const metrics = this.metrics.get(bridgeId);
    if (metrics) {
      metrics.totalTransactions++;
    }
  }

  /**
   * Increment failure count
   */
  private incrementFailureCount(bridgeId: string): void {
    const metrics = this.metrics.get(bridgeId);
    if (metrics) {
      metrics.failedTransactions++;
      metrics.successRate = ((metrics.totalTransactions - metrics.failedTransactions) / metrics.totalTransactions) * 100;
    }
  }

  /**
   * Calculate system uptime
   */
  private calculateSystemUptime(): number {
    const bridges = Array.from(this.healthStatus.values());
    if (bridges.length === 0) return 100;

    const healthyCount = bridges.filter(b => b.isHealthy).length;
    return (healthyCount / bridges.length) * 100;
  }

  /**
   * Get critical alert count
   */
  private getCriticalAlertCount(): number {
    let count = 0;
    
    for (const status of this.healthStatus.values()) {
      if (!status.isHealthy || 
          status.errorRate > this.alertConfig.maxErrorRate ||
          status.liquidity < this.alertConfig.minLiquidity) {
        count++;
      }
    }

    return count;
  }

  /**
   * Get chain ID by network name
   */
  private getChainIdByNetwork(network: string): number {
    const chainIds: { [network: string]: number } = {
      ethereum: 1,
      avalanche: 43114,
      polygon: 137,
      arbitrum: 42161
    };

    return chainIds[network] || 0;
  }

  /**
   * Perform full health check
   */
  private async performFullHealthCheck(): Promise<void> {
    const bridgeIds = Array.from(this.healthStatus.keys());
    
    await Promise.allSettled(
      bridgeIds.map(bridgeId => this.checkBridgeHealth(bridgeId))
    );

    this.logger.info('Full health check completed', {
      totalBridges: bridgeIds.length,
      systemHealth: this.getSystemHealth()
    });
  }
}
