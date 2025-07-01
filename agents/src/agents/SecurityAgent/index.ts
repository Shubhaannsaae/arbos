import { ethers, BigNumber } from 'ethers';
import { logger } from '../../shared/utils/logger';
import { getAgentConfig } from '../../config/agentConfig';
import { getElizaConfig, invokeBedrockModel } from '../../config/elizaConfig';
import { renderPromptTemplate } from '../../config/modelConfig';
import { SECURITY_THRESHOLDS } from '../../shared/constants/thresholds';
import { AgentContext, AgentDecision, AgentExecution, AgentState } from '../../shared/types/agent';
import { SecurityAlert, SecurityEvent, ThreatLevel } from '../../shared/types/blockchain';

// Import actions
import { monitorTransactions } from './actions/monitorTransactions';
import { detectAnomalies } from './actions/detectAnomalies';
import { triggerAlerts } from './actions/triggerAlerts';

// Import evaluators
import { FraudEvaluator } from './evaluators/fraudEvaluator';
import { SecurityRiskEvaluator } from './evaluators/riskEvaluator';

// Import providers
import { SecurityProvider } from './providers/securityProvider';

export interface SecurityAgentConfig {
  monitoringEnabled: boolean;
  alertThresholds: {
    suspiciousTransaction: number;
    volumeAnomaly: number;
    priceManipulation: number;
    rugPull: number;
    phishing: number;
  };
  monitoredAddresses: string[];
  monitoredContracts: string[];
  supportedChains: number[];
  realTimeMonitoring: boolean;
  emergencyResponse: boolean;
  notificationChannels: string[];
}

export class SecurityAgent {
  private agentId: string;
  private config: SecurityAgentConfig;
  private state: AgentState;
  private providers: {
    security: SecurityProvider;
  };
  private evaluators: {
    fraud: FraudEvaluator;
    risk: SecurityRiskEvaluator;
  };
  private monitoringInterval: NodeJS.Timeout | null = null;
  private activeExecutions: Map<string, AgentExecution> = new Map();
  private securityEvents: Map<string, SecurityEvent[]> = new Map();
  private alertHistory: SecurityAlert[] = [];

  constructor(agentId: string, config: SecurityAgentConfig) {
    this.agentId = agentId;
    this.config = config;

    // Initialize providers
    this.providers = {
      security: new SecurityProvider(config.supportedChains)
    };

    // Initialize evaluators
    this.evaluators = {
      fraud: new FraudEvaluator(),
      risk: new SecurityRiskEvaluator()
    };

    // Initialize state
    this.state = this.initializeState();

    logger.agentStarted(this.agentId, 'security', {
      config: this.config
    });
  }

  private initializeState(): AgentState {
    const agentConfig = getAgentConfig('security');
    
    return {
      agentId: this.agentId,
      status: 'idle',
      healthScore: 100,
      errorCount: 0,
      warningCount: 0,
      memory: {
        shortTerm: {},
        longTerm: {
          suspiciousPatterns: [],
          knownThreats: [],
          whitelistedAddresses: []
        },
        episodic: [],
        semantic: {
          threatIntelligence: {},
          behaviorPatterns: {},
          riskProfiles: {}
        }
      },
      configuration: agentConfig.constraints as any,
      performance: {
        agentId: this.agentId,
        period: {
          start: Date.now(),
          end: Date.now()
        },
        metrics: {
          totalExecutions: 0,
          successfulExecutions: 0,
          failedExecutions: 0,
          successRate: 0,
          averageExecutionTime: 0,
          totalGasUsed: BigNumber.from(0),
          averageGasUsed: BigNumber.from(0),
          totalProfit: BigNumber.from(0),
          totalLoss: BigNumber.from(0),
          netProfit: BigNumber.from(0),
          profitFactor: 0,
          sharpeRatio: 0,
          maxDrawdown: 0,
          winRate: 0
        },
        chainlinkUsage: {
          dataFeedCalls: 0,
          automationExecutions: 0,
          functionsRequests: 0,
          vrfRequests: 0,
          ccipMessages: 0,
          totalCost: BigNumber.from(0)
        }
      },
      resources: {
        cpuUsage: 0,
        memoryUsage: 0,
        networkLatency: 0,
        apiCallsRemaining: 1000,
        gasAllowanceRemaining: BigNumber.from('2000000000000000000') // 2 ETH
      }
    };
  }

  async start(): Promise<void> {
    if (this.state.status !== 'idle') {
      throw new Error('Security agent is already running');
    }

    this.state.status = 'monitoring';

    // Initialize providers
    await this.providers.security.initialize();

    // Load historical security data
    await this.loadSecurityHistory();

    // Start monitoring loop if enabled
    if (this.config.monitoringEnabled) {
      const interval = getAgentConfig('security').executionInterval;
      this.monitoringInterval = setInterval(async () => {
        await this.monitoringLoop();
      }, interval);
    }

    logger.info('SecurityAgent started', {
      agentId: this.agentId,
      monitoringEnabled: this.config.monitoringEnabled,
      monitoredAddresses: this.config.monitoredAddresses.length,
      supportedChains: this.config.supportedChains.length
    });
  }

  async stop(): Promise<void> {
    this.state.status = 'idle';
    
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }

    // Wait for active executions to complete
    const activeCount = this.activeExecutions.size;
    if (activeCount > 0) {
      logger.info(`Waiting for ${activeCount} active executions to complete`);
      
      const timeout = 300000; // 5 minutes
      const startTime = Date.now();
      
      while (this.activeExecutions.size > 0 && (Date.now() - startTime) < timeout) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    logger.agentStopped(this.agentId, 'security');
  }

  private async monitoringLoop(): Promise<void> {
    try {
      this.state.status = 'monitoring';

      // 1. Monitor transactions for suspicious activity
      await this.performTransactionMonitoring();

      // 2. Detect anomalies in market behavior
      await this.performAnomalyDetection();

      // 3. Evaluate new threats and risks
      await this.performThreatEvaluation();

      // 4. Process security alerts
      await this.processSecurityAlerts();

      this.state.status = 'idle';

    } catch (error) {
      this.state.status = 'error';
      this.state.errorCount++;
      
      logger.agentError(this.agentId, 'security', error as Error, {
        monitoringLoop: true
      });

      if (this.state.errorCount >= 5) {
        this.state.status = 'paused';
        logger.error('Security agent paused due to excessive errors', {
          agentId: this.agentId,
          errorCount: this.state.errorCount
        });
      }
    }
  }

  private async loadSecurityHistory(): Promise<void> {
    try {
      // Load historical security events and alerts
      const historicalEvents = await this.providers.security.getHistoricalEvents();
      
      for (const event of historicalEvents) {
        const chainEvents = this.securityEvents.get(event.chainId.toString()) || [];
        chainEvents.push(event);
        this.securityEvents.set(event.chainId.toString(), chainEvents);
      }

      // Load alert history
      this.alertHistory = await this.providers.security.getAlertHistory();

      logger.info('Security history loaded', {
        agentId: this.agentId,
        eventsLoaded: historicalEvents.length,
        alertsLoaded: this.alertHistory.length
      });

    } catch (error) {
      logger.error('Failed to load security history', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async performTransactionMonitoring(): Promise<void> {
    try {
      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'security',
        userId: 'system',
        sessionId: `session_${Date.now()}`,
        networkIds: this.config.supportedChains,
        timestamp: Date.now(),
        gasPrice: BigNumber.from('20000000000'),
        nonce: 0
      };

      const monitoringResult = await monitorTransactions(
        this.providers.security,
        context,
        {
          monitoredAddresses: this.config.monitoredAddresses,
          monitoredContracts: this.config.monitoredContracts,
          alertThresholds: this.config.alertThresholds,
          timeWindow: 300000 // 5 minutes
        }
      );

      if (monitoringResult.suspiciousTransactions.length > 0) {
        await this.processSuspiciousTransactions(monitoringResult.suspiciousTransactions);
      }

    } catch (error) {
      logger.error('Transaction monitoring failed', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async performAnomalyDetection(): Promise<void> {
    try {
      const context: AgentContext = {
        agentId: this.agentId,
        agentType: 'security',
        userId: 'system',
        sessionId: `session_${Date.now()}`,
        networkIds: this.config.supportedChains,
        timestamp: Date.now(),
        gasPrice: BigNumber.from('20000000000'),
        nonce: 0
      };

      const anomalyResult = await detectAnomalies(
        this.providers.security,
        context,
        {
          monitoringWindow: 3600000, // 1 hour
          sensitivityLevel: 'medium',
          anomalyTypes: ['volume', 'price', 'behavior', 'smart_contract'],
          excludeKnownPatterns: true
        }
      );

      if (anomalyResult.anomalies.length > 0) {
        await this.processDetectedAnomalies(anomalyResult.anomalies);
      }

    } catch (error) {
      logger.error('Anomaly detection failed', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async performThreatEvaluation(): Promise<void> {
    try {
      // Evaluate known threats using AI models
      const recentEvents = this.getRecentSecurityEvents();
      
      for (const event of recentEvents) {
        const threatEvaluation = await this.evaluateThreat(event);
        
        if (threatEvaluation.threatLevel === 'critical' || threatEvaluation.threatLevel === 'high') {
          await this.escalateThreat(event, threatEvaluation);
        }
      }

    } catch (error) {
      logger.error('Threat evaluation failed', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async processSecurityAlerts(): Promise<void> {
    try {
      const pendingAlerts = this.alertHistory.filter(alert => 
        alert.status === 'pending' && Date.now() - alert.timestamp < 3600000 // 1 hour
      );

      for (const alert of pendingAlerts) {
        await this.processAlert(alert);
      }

    } catch (error) {
      logger.error('Alert processing failed', {
        agentId: this.agentId,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private async processSuspiciousTransactions(transactions: any[]): Promise<void> {
    for (const transaction of transactions) {
      try {
        // Analyze transaction with fraud evaluator
        const fraudAnalysis = await this.evaluators.fraud.evaluateTransaction(transaction);
        
        if (fraudAnalysis.riskScore > SECURITY_THRESHOLDS.FRAUD_DETECTION) {
          await this.initiateSecurityResponse(transaction, fraudAnalysis);
        }

      } catch (error) {
        logger.error('Failed to process suspicious transaction', {
          transactionHash: transaction.hash,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  private async processDetectedAnomalies(anomalies: any[]): Promise<void> {
    for (const anomaly of anomalies) {
      try {
        // Evaluate anomaly severity
        const riskEvaluation = await this.evaluators.risk.evaluateAnomaly(anomaly);
        
        if (riskEvaluation.severity === 'high' || riskEvaluation.severity === 'critical') {
          await this.initiateAnomalyResponse(anomaly, riskEvaluation);
        }

      } catch (error) {
        logger.error('Failed to process detected anomaly', {
          anomalyId: anomaly.id,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  private async evaluateThreat(event: SecurityEvent): Promise<{
    threatLevel: ThreatLevel;
    confidence: number;
    reasoning: string;
    recommendedActions: string[];
  }> {
    try {
      // Use AI to evaluate threat
      const elizaConfig = getElizaConfig('security');
      
      const prompt = renderPromptTemplate('threat_analysis', {
        eventType: event.type,
        severity: event.severity,
        chainId: event.chainId,
        transactionHash: event.transactionHash || 'N/A',
        contractAddress: event.contractAddress || 'N/A',
        description: event.description,
        metadata: JSON.stringify(event.metadata || {}),
        historicalContext: JSON.stringify(this.getRelatedThreatHistory(event))
      });

      const response = await invokeBedrockModel({
        modelId: elizaConfig.modelId,
        prompt,
        maxTokens: elizaConfig.maxTokens,
        temperature: elizaConfig.temperature
      });

      const aiResponse = JSON.parse(response);
      
      return {
        threatLevel: aiResponse.threatLevel || 'medium',
        confidence: aiResponse.confidence || 0.7,
        reasoning: aiResponse.analysis || 'AI analysis completed',
        recommendedActions: aiResponse.actions || []
      };

    } catch (error) {
      logger.error('Threat evaluation failed', {
        eventId: event.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        threatLevel: 'medium',
        confidence: 0.5,
        reasoning: 'Evaluation failed, defaulting to medium threat level',
        recommendedActions: ['Manual review required']
      };
    }
  }

  private async initiateSecurityResponse(transaction: any, fraudAnalysis: any): Promise<void> {
    const executionId = `security_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const execution: AgentExecution = {
      id: executionId,
      agentId: this.agentId,
      decision: {
        action: 'security_alert',
        confidence: fraudAnalysis.confidence,
        reasoning: `Suspicious transaction detected: ${fraudAnalysis.reasoning}`,
        parameters: { transaction, fraudAnalysis },
        riskScore: fraudAnalysis.riskScore,
        expectedOutcome: null,
        alternatives: []
      },
      startTime: Date.now(),
      status: 'pending',
      transactions: [],
      gasUsed: BigNumber.from(0),
      actualOutcome: null,
      errors: []
    };

    this.activeExecutions.set(executionId, execution);

    logger.executionStarted(this.agentId, executionId, 'security_alert', {
      transactionHash: transaction.hash,
      riskScore: fraudAnalysis.riskScore
    });

    try {
      execution.status = 'executing';
      
      const alertResult = await triggerAlerts(
        this.providers.security,
        {
          agentId: this.agentId,
          agentType: 'security',
          userId: 'system',
          sessionId: `session_${Date.now()}`,
          networkIds: [transaction.chainId],
          timestamp: Date.now(),
          gasPrice: BigNumber.from('20000000000'),
          nonce: 0
        },
        {
          alertType: 'fraud_detected',
          severity: fraudAnalysis.riskScore > 80 ? 'critical' : 'high',
          description: `Fraudulent transaction detected: ${transaction.hash}`,
          affectedAddresses: [transaction.from, transaction.to].filter(Boolean),
          evidenceData: fraudAnalysis,
          recommendedActions: fraudAnalysis.recommendedActions || [],
          emergencyStop: fraudAnalysis.riskScore > 90
        }
      );

      execution.status = 'completed';
      execution.endTime = Date.now();
      execution.actualOutcome = alertResult;

      // Update alert history
      this.alertHistory.push({
        id: executionId,
        type: 'fraud_detected',
        severity: 'high',
        title: 'Fraudulent Transaction Detected',
        description: `Suspicious transaction ${transaction.hash} detected with risk score ${fraudAnalysis.riskScore}`,
        timestamp: Date.now(),
        chainId: transaction.chainId,
        transactionHash: transaction.hash,
        contractAddress: transaction.to,
        affectedAddresses: [transaction.from, transaction.to].filter(Boolean),
        status: 'active',
        source: this.agentId,
        metadata: fraudAnalysis
      });

      logger.executionCompleted(this.agentId, executionId, true, {
        alertsTriggered: alertResult.alertsTriggered,
        notificationsSent: alertResult.notificationsSent,
        duration: execution.endTime - execution.startTime
      });

    } catch (error) {
      execution.status = 'failed';
      execution.endTime = Date.now();
      execution.errors.push(error instanceof Error ? error.message : String(error));

      logger.executionCompleted(this.agentId, executionId, false, {
        error: error instanceof Error ? error.message : String(error),
        duration: execution.endTime! - execution.startTime
      });

    } finally {
      this.activeExecutions.delete(executionId);
    }
  }

  private async initiateAnomalyResponse(anomaly: any, riskEvaluation: any): Promise<void> {
    const executionId = `anomaly_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    logger.info('Initiating anomaly response', {
      executionId,
      anomalyType: anomaly.type,
      severity: riskEvaluation.severity
    });

    // Implementation would trigger appropriate response based on anomaly type
  }

  private async escalateThreat(event: SecurityEvent, threatEvaluation: any): Promise<void> {
    logger.warn('Escalating security threat', {
      agentId: this.agentId,
      eventId: event.id,
      threatLevel: threatEvaluation.threatLevel,
      confidence: threatEvaluation.confidence
    });

    // Implementation would escalate to emergency response protocols
  }

  private async processAlert(alert: SecurityAlert): Promise<void> {
    try {
      // Process pending alert
      logger.info('Processing security alert', {
        alertId: alert.id,
        type: alert.type,
        severity: alert.severity
      });

      // Update alert status
      alert.status = 'processed';
      alert.resolvedAt = Date.now();

    } catch (error) {
      logger.error('Failed to process alert', {
        alertId: alert.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  private getRecentSecurityEvents(): SecurityEvent[] {
    const recentEvents: SecurityEvent[] = [];
    const cutoffTime = Date.now() - 3600000; // 1 hour

    for (const events of this.securityEvents.values()) {
      recentEvents.push(...events.filter(event => event.timestamp > cutoffTime));
    }

    return recentEvents.sort((a, b) => b.timestamp - a.timestamp);
  }

  private getRelatedThreatHistory(event: SecurityEvent): any[] {
    // Get related historical threats for context
    return this.alertHistory
      .filter(alert => 
        alert.type === event.type || 
        alert.contractAddress === event.contractAddress ||
        alert.affectedAddresses?.some(addr => 
          event.affectedAddresses?.includes(addr)
        )
      )
      .slice(0, 5) // Last 5 related events
      .map(alert => ({
        type: alert.type,
        severity: alert.severity,
        timestamp: alert.timestamp,
        description: alert.description
      }));
  }

  // Public methods for external control
  async addMonitoredAddress(address: string): Promise<void> {
    if (!this.config.monitoredAddresses.includes(address)) {
      this.config.monitoredAddresses.push(address);
      
      logger.info('Address added to monitoring', {
        agentId: this.agentId,
        address
      });
    }
  }

  async removeMonitoredAddress(address: string): Promise<void> {
    const index = this.config.monitoredAddresses.indexOf(address);
    if (index > -1) {
      this.config.monitoredAddresses.splice(index, 1);
      
      logger.info('Address removed from monitoring', {
        agentId: this.agentId,
        address
      });
    }
  }

  async addMonitoredContract(contractAddress: string): Promise<void> {
    if (!this.config.monitoredContracts.includes(contractAddress)) {
      this.config.monitoredContracts.push(contractAddress);
      
      logger.info('Contract added to monitoring', {
        agentId: this.agentId,
        contractAddress
      });
    }
  }

  async pauseAgent(): Promise<void> {
    this.state.status = 'paused';
    logger.info('SecurityAgent paused', { agentId: this.agentId });
  }

  async resumeAgent(): Promise<void> {
    if (this.state.status === 'paused') {
      this.state.status = 'idle';
      logger.info('SecurityAgent resumed', { agentId: this.agentId });
    }
  }

  getState(): AgentState {
    return { ...this.state };
  }

  getActiveExecutions(): AgentExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  getAlertHistory(): SecurityAlert[] {
    return [...this.alertHistory];
  }

  getSecurityEvents(chainId?: number): SecurityEvent[] {
    if (chainId) {
      return this.securityEvents.get(chainId.toString()) || [];
    }
    
    const allEvents: SecurityEvent[] = [];
    for (const events of this.securityEvents.values()) {
      allEvents.push(...events);
    }
    
    return allEvents.sort((a, b) => b.timestamp - a.timestamp);
  }

  async updateConfiguration(newConfig: Partial<SecurityAgentConfig>): Promise<void> {
    this.config = { ...this.config, ...newConfig };
    
    logger.info('SecurityAgent configuration updated', {
      agentId: this.agentId,
      newConfig
    });
  }

  async getHealthStatus(): Promise<{
    status: string;
    healthScore: number;
    issues: string[];
    recommendations: string[];
    monitoringStats: {
      eventsProcessed: number;
      alertsTriggered: number;
      threatsDetected: number;
      uptime: number;
    };
  }> {
    const issues: string[] = [];
    const recommendations: string[] = [];
    
    // Check monitoring status
    if (!this.config.monitoringEnabled) {
      issues.push('Real-time monitoring is disabled');
      recommendations.push('Enable monitoring for better threat detection');
    }

    // Check recent error rate
    if (this.state.errorCount > 3) {
      issues.push('High error rate detected');
      recommendations.push('Review agent configuration and logs');
    }

    // Check alert response time
    const recentAlerts = this.alertHistory.filter(alert => 
      Date.now() - alert.timestamp < 3600000 // Last hour
    );
    
    const unprocessedAlerts = recentAlerts.filter(alert => alert.status === 'pending').length;
    if (unprocessedAlerts > 5) {
      issues.push('High number of unprocessed alerts');
      recommendations.push('Increase processing capacity or review alert thresholds');
    }

    const healthScore = Math.max(0, 100 - (issues.length * 20));

    const monitoringStats = {
      eventsProcessed: this.getSecurityEvents().length,
      alertsTriggered: this.alertHistory.length,
      threatsDetected: this.alertHistory.filter(alert => 
        alert.type.includes('fraud') || alert.type.includes('threat')
      ).length,
      uptime: Date.now() - this.state.performance.period.start
    };

    return {
      status: this.state.status,
      healthScore,
      issues,
      recommendations,
      monitoringStats
    };
  }
}

export default SecurityAgent;
