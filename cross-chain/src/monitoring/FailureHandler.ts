import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import { EventEmitter } from 'events';
import { CrossChainTransaction, TransactionStatus } from './TransactionTracker';
import { BridgeStatus } from './BridgeMonitor';

export interface FailureEvent {
  id: string;
  type: FailureType;
  severity: FailureSeverity;
  source: string;
  timestamp: number;
  data: any;
  resolved: boolean;
  resolvedAt?: number;
  attempts: number;
  lastAttempt?: number;
}

export enum FailureType {
  TRANSACTION_FAILED = 'transaction_failed',
  TRANSACTION_STUCK = 'transaction_stuck',
  BRIDGE_UNHEALTHY = 'bridge_unhealthy',
  HIGH_LATENCY = 'high_latency',
  LIQUIDITY_LOW = 'liquidity_low',
  NETWORK_ERROR = 'network_error',
  GAS_ESTIMATION_FAILED = 'gas_estimation_failed',
  INSUFFICIENT_FUNDS = 'insufficient_funds'
}

export enum FailureSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface RecoveryAction {
  type: RecoveryType;
  description: string;
  automated: boolean;
  execute: () => Promise<boolean>;
}

export enum RecoveryType {
  RETRY = 'retry',
  REROUTE = 'reroute',
  MANUAL_INTERVENTION = 'manual_intervention',
  REFUND = 'refund',
  ESCALATE = 'escalate'
}

export interface FailureHandlerConfig {
  maxRetries: number;
  retryDelay: number;
  escalationThreshold: number;
  autoRecovery: boolean;
  notificationEndpoints: {
    webhook?: string;
    email?: string;
    slack?: string;
  };
}

export class FailureHandler extends EventEmitter {
  private logger: Logger;
  private failures: Map<string, FailureEvent> = new Map();
  private config: FailureHandlerConfig;
  private recoveryStrategies: Map<FailureType, RecoveryAction[]> = new Map();

  constructor(config?: Partial<FailureHandlerConfig>) {
    super();
    
    this.config = {
      maxRetries: 3,
      retryDelay: 300000, // 5 minutes
      escalationThreshold: 3600000, // 1 hour
      autoRecovery: true,
      notificationEndpoints: {},
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
        new transports.File({ filename: 'logs/failure-handler.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.initializeRecoveryStrategies();
  }

  /**
   * Handle transaction failure
   */
  async handleTransactionFailure(
    transaction: CrossChainTransaction,
    error: Error,
    severity: FailureSeverity = FailureSeverity.MEDIUM
  ): Promise<void> {
    const failureId = this.generateFailureId(transaction.id, FailureType.TRANSACTION_FAILED);
    
    const failure: FailureEvent = {
      id: failureId,
      type: FailureType.TRANSACTION_FAILED,
      severity,
      source: transaction.id,
      timestamp: Date.now(),
      data: {
        transaction,
        error: error.message,
        stack: error.stack
      },
      resolved: false,
      attempts: 0
    };

    this.failures.set(failureId, failure);

    this.logger.error(`Transaction failure detected`, {
      failureId,
      transactionId: transaction.id,
      error: error.message,
      severity
    });

    // Notify stakeholders
    await this.sendNotification(failure);

    // Attempt automatic recovery if enabled
    if (this.config.autoRecovery) {
      await this.attemptRecovery(failureId);
    }

    this.emit('failure:detected', failure);
  }

  /**
   * Handle bridge health failure
   */
  async handleBridgeFailure(
    bridgeId: string,
    status: BridgeStatus,
    severity: FailureSeverity = FailureSeverity.HIGH
  ): Promise<void> {
    const failureId = this.generateFailureId(bridgeId, FailureType.BRIDGE_UNHEALTHY);
    
    const failure: FailureEvent = {
      id: failureId,
      type: FailureType.BRIDGE_UNHEALTHY,
      severity,
      source: bridgeId,
      timestamp: Date.now(),
      data: {
        bridgeStatus: status,
        healthMetrics: {
          responseTime: status.responseTime,
          errorRate: status.errorRate,
          liquidity: status.liquidity.toString(),
          queueLength: status.queueLength
        }
      },
      resolved: false,
      attempts: 0
    };

    this.failures.set(failureId, failure);

    this.logger.error(`Bridge failure detected`, {
      failureId,
      bridgeId,
      status,
      severity
    });

    await this.sendNotification(failure);

    if (this.config.autoRecovery) {
      await this.attemptRecovery(failureId);
    }

    this.emit('failure:detected', failure);
  }

  /**
   * Handle stuck transaction
   */
  async handleStuckTransaction(
    transaction: CrossChainTransaction,
    stuckDuration: number
  ): Promise<void> {
    const severity = stuckDuration > 3600000 ? FailureSeverity.HIGH : FailureSeverity.MEDIUM;
    const failureId = this.generateFailureId(transaction.id, FailureType.TRANSACTION_STUCK);
    
    const failure: FailureEvent = {
      id: failureId,
      type: FailureType.TRANSACTION_STUCK,
      severity,
      source: transaction.id,
      timestamp: Date.now(),
      data: {
        transaction,
        stuckDuration,
        lastStatus: transaction.status
      },
      resolved: false,
      attempts: 0
    };

    this.failures.set(failureId, failure);

    this.logger.warn(`Stuck transaction detected`, {
      failureId,
      transactionId: transaction.id,
      stuckDuration,
      severity
    });

    await this.sendNotification(failure);

    if (this.config.autoRecovery) {
      await this.attemptRecovery(failureId);
    }

    this.emit('failure:detected', failure);
  }

  /**
   * Handle network error
   */
  async handleNetworkError(
    chainId: number,
    error: Error,
    severity: FailureSeverity = FailureSeverity.MEDIUM
  ): Promise<void> {
    const failureId = this.generateFailureId(`chain-${chainId}`, FailureType.NETWORK_ERROR);
    
    const failure: FailureEvent = {
      id: failureId,
      type: FailureType.NETWORK_ERROR,
      severity,
      source: `chain-${chainId}`,
      timestamp: Date.now(),
      data: {
        chainId,
        error: error.message,
        stack: error.stack
      },
      resolved: false,
      attempts: 0
    };

    this.failures.set(failureId, failure);

    this.logger.error(`Network error detected`, {
      failureId,
      chainId,
      error: error.message,
      severity
    });

    await this.sendNotification(failure);
    this.emit('failure:detected', failure);
  }

  /**
   * Attempt automatic recovery
   */
  async attemptRecovery(failureId: string): Promise<boolean> {
    const failure = this.failures.get(failureId);
    if (!failure) {
      return false;
    }

    if (failure.attempts >= this.config.maxRetries) {
      await this.escalateFailure(failure);
      return false;
    }

    const strategies = this.recoveryStrategies.get(failure.type) || [];
    failure.attempts++;
    failure.lastAttempt = Date.now();

    this.logger.info(`Attempting recovery for failure ${failureId}`, {
      attempt: failure.attempts,
      strategies: strategies.length
    });

    for (const strategy of strategies) {
      if (!strategy.automated) {
        continue;
      }

      try {
        const success = await strategy.execute();
        if (success) {
          await this.resolveFailure(failureId);
          this.logger.info(`Recovery successful for failure ${failureId}`, {
            strategy: strategy.type
          });
          return true;
        }
      } catch (error) {
        this.logger.error(`Recovery strategy failed`, {
          failureId,
          strategy: strategy.type,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Schedule retry if not at max attempts
    if (failure.attempts < this.config.maxRetries) {
      setTimeout(() => {
        this.attemptRecovery(failureId);
      }, this.config.retryDelay);
    } else {
      await this.escalateFailure(failure);
    }

    return false;
  }

  /**
   * Resolve failure
   */
  async resolveFailure(failureId: string): Promise<void> {
    const failure = this.failures.get(failureId);
    if (!failure) {
      return;
    }

    failure.resolved = true;
    failure.resolvedAt = Date.now();

    this.logger.info(`Failure resolved`, {
      failureId,
      duration: Date.now() - failure.timestamp,
      attempts: failure.attempts
    });

    this.emit('failure:resolved', failure);
  }

  /**
   * Escalate failure to manual intervention
   */
  async escalateFailure(failure: FailureEvent): Promise<void> {
    failure.severity = FailureSeverity.CRITICAL;

    this.logger.error(`Failure escalated to critical`, {
      failureId: failure.id,
      type: failure.type,
      attempts: failure.attempts,
      duration: Date.now() - failure.timestamp
    });

    // Send critical notification
    await this.sendCriticalNotification(failure);

    this.emit('failure:escalated', failure);
  }

  /**
   * Get failure by ID
   */
  getFailure(failureId: string): FailureEvent | undefined {
    return this.failures.get(failureId);
  }

  /**
   * Get all failures with optional filters
   */
  getFailures(filters?: {
    type?: FailureType;
    severity?: FailureSeverity;
    resolved?: boolean;
    source?: string;
  }): FailureEvent[] {
    let failures = Array.from(this.failures.values());

    if (filters) {
      if (filters.type) {
        failures = failures.filter(f => f.type === filters.type);
      }
      if (filters.severity) {
        failures = failures.filter(f => f.severity === filters.severity);
      }
      if (filters.resolved !== undefined) {
        failures = failures.filter(f => f.resolved === filters.resolved);
      }
      if (filters.source) {
        failures = failures.filter(f => f.source === filters.source);
      }
    }

    return failures.sort((a, b) => b.timestamp - a.timestamp);
  }

  /**
   * Get failure statistics
   */
  getFailureStats(): {
    total: number;
    byType: { [type: string]: number };
    bySeverity: { [severity: string]: number };
    resolved: number;
    escalated: number;
    averageResolutionTime: number;
  } {
    const failures = Array.from(this.failures.values());
    
    const byType: { [type: string]: number } = {};
    const bySeverity: { [severity: string]: number } = {};
    
    let resolved = 0;
    let escalated = 0;
    let totalResolutionTime = 0;
    let resolvedCount = 0;

    for (const failure of failures) {
      byType[failure.type] = (byType[failure.type] || 0) + 1;
      bySeverity[failure.severity] = (bySeverity[failure.severity] || 0) + 1;
      
      if (failure.resolved) {
        resolved++;
        if (failure.resolvedAt) {
          totalResolutionTime += failure.resolvedAt - failure.timestamp;
          resolvedCount++;
        }
      }
      
      if (failure.severity === FailureSeverity.CRITICAL) {
        escalated++;
      }
    }

    return {
      total: failures.length,
      byType,
      bySeverity,
      resolved,
      escalated,
      averageResolutionTime: resolvedCount > 0 ? totalResolutionTime / resolvedCount : 0
    };
  }

  /**
   * Initialize recovery strategies
   */
  private initializeRecoveryStrategies(): void {
    // Transaction failure recovery
    this.recoveryStrategies.set(FailureType.TRANSACTION_FAILED, [
      {
        type: RecoveryType.RETRY,
        description: 'Retry failed transaction with higher gas',
        automated: true,
        execute: async () => {
          // Implementation would retry with higher gas
          return false; // Simplified
        }
      },
      {
        type: RecoveryType.REROUTE,
        description: 'Route through alternative bridge',
        automated: true,
        execute: async () => {
          // Implementation would find alternative route
          return false; // Simplified
        }
      }
    ]);

    // Stuck transaction recovery
    this.recoveryStrategies.set(FailureType.TRANSACTION_STUCK, [
      {
        type: RecoveryType.RETRY,
        description: 'Increase gas price and retry',
        automated: true,
        execute: async () => {
          // Implementation would increase gas and retry
          return false; // Simplified
        }
      }
    ]);

    // Bridge unhealthy recovery
    this.recoveryStrategies.set(FailureType.BRIDGE_UNHEALTHY, [
      {
        type: RecoveryType.REROUTE,
        description: 'Switch to healthy bridge',
        automated: true,
        execute: async () => {
          // Implementation would switch to backup bridge
          return false; // Simplified
        }
      }
    ]);

    // Network error recovery
    this.recoveryStrategies.set(FailureType.NETWORK_ERROR, [
      {
        type: RecoveryType.RETRY,
        description: 'Retry with backup RPC endpoint',
        automated: true,
        execute: async () => {
          // Implementation would switch to backup RPC
          return false; // Simplified
        }
      }
    ]);
  }

  /**
   * Send notification
   */
  private async sendNotification(failure: FailureEvent): Promise<void> {
    const message = this.formatNotificationMessage(failure);

    // Send webhook notification
    if (this.config.notificationEndpoints.webhook) {
      try {
        await this.sendWebhookNotification(
          this.config.notificationEndpoints.webhook,
          failure,
          message
        );
      } catch (error) {
        this.logger.error('Failed to send webhook notification', { error });
      }
    }

    // Additional notification methods can be implemented here
    this.logger.info(`Notification sent for failure ${failure.id}`);
  }

  /**
   * Send critical notification
   */
  private async sendCriticalNotification(failure: FailureEvent): Promise<void> {
    const message = `🚨 CRITICAL FAILURE 🚨\n${this.formatNotificationMessage(failure)}`;
    
    // Send to all configured endpoints for critical failures
    await this.sendNotification(failure);
    
    // Additional escalation logic can be added here
  }

  /**
   * Send webhook notification
   */
  private async sendWebhookNotification(
    webhook: string,
    failure: FailureEvent,
    message: string
  ): Promise<void> {
    const payload = {
      type: 'failure_alert',
      failure: {
        id: failure.id,
        type: failure.type,
        severity: failure.severity,
        source: failure.source,
        timestamp: failure.timestamp
      },
      message
    };

    // In production, use actual HTTP client
    this.logger.debug('Webhook notification payload', { webhook, payload });
  }

  /**
   * Format notification message
   */
  private formatNotificationMessage(failure: FailureEvent): string {
    const duration = Date.now() - failure.timestamp;
    const durationText = this.formatDuration(duration);

    return `
Failure Detected:
ID: ${failure.id}
Type: ${failure.type}
Severity: ${failure.severity}
Source: ${failure.source}
Duration: ${durationText}
Attempts: ${failure.attempts}
    `.trim();
  }

  /**
   * Format duration in human readable format
   */
  private formatDuration(milliseconds: number): string {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  /**
   * Generate unique failure ID
   */
  private generateFailureId(source: string, type: FailureType): string {
    return `${source}-${type}-${Date.now()}`;
  }

  /**
   * Cleanup old resolved failures
   */
  cleanup(maxAgeHours: number = 168): void { // 1 week default
    const cutoffTime = Date.now() - (maxAgeHours * 60 * 60 * 1000);
    let cleanedCount = 0;

    for (const [id, failure] of this.failures.entries()) {
      if (failure.resolved && failure.timestamp < cutoffTime) {
        this.failures.delete(id);
        cleanedCount++;
      }
    }

    this.logger.info(`Cleaned up ${cleanedCount} old resolved failures`);
  }
}
