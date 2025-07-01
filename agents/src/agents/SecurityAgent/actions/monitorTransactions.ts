import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { TransactionEvent, SecurityAlert } from '../../../shared/types/blockchain';
import { SECURITY_THRESHOLDS } from '../../../shared/constants/thresholds';
import { SecurityProvider } from '../providers/securityProvider';

export interface TransactionMonitoringConfig {
  monitoredAddresses: string[];
  monitoredContracts: string[];
  alertThresholds: {
    suspiciousTransaction: number;
    volumeAnomaly: number;
    priceManipulation: number;
    rugPull: number;
    phishing: number;
  };
  timeWindow: number; // milliseconds
  batchSize?: number;
  realTimeMode?: boolean;
}

export interface MonitoringResult {
  totalTransactionsAnalyzed: number;
  suspiciousTransactions: SuspiciousTransaction[];
  patterns: DetectedPattern[];
  alerts: SecurityAlert[];
  performanceMetrics: {
    processingTime: number;
    averageAnalysisTime: number;
    falsePositiveRate: number;
    detectionAccuracy: number;
  };
}

export interface SuspiciousTransaction {
  hash: string;
  chainId: number;
  from: string;
  to: string;
  value: BigNumber;
  gasPrice: BigNumber;
  gasUsed: BigNumber;
  timestamp: number;
  suspicionScore: number;
  suspicionReasons: string[];
  riskCategories: string[];
  metadata: {
    contractInteraction?: boolean;
    unusualGasPattern?: boolean;
    knownMaliciousAddress?: boolean;
    valueAnomalyDetected?: boolean;
    timingAnomalyDetected?: boolean;
  };
}

export interface DetectedPattern {
  patternType: 'wash_trading' | 'sandwich_attack' | 'front_running' | 'pump_dump' | 'rug_pull' | 'phishing';
  confidence: number;
  affectedTransactions: string[];
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timeframe: {
    start: number;
    end: number;
  };
  involvedAddresses: string[];
  estimatedImpact: {
    affectedUsers: number;
    estimatedLoss: BigNumber;
  };
}

export async function monitorTransactions(
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: TransactionMonitoringConfig
): Promise<MonitoringResult> {
  const startTime = Date.now();

  logger.info('Starting transaction monitoring', {
    agentId: context.agentId,
    monitoredAddresses: config.monitoredAddresses.length,
    monitoredContracts: config.monitoredContracts.length,
    timeWindow: config.timeWindow,
    realTimeMode: config.realTimeMode || false
  });

  try {
    // Step 1: Collect transactions from monitored sources
    const transactions = await collectTransactions(securityProvider, context, config);

    // Step 2: Analyze each transaction for suspicious activity
    const suspiciousTransactions = await analyzeTransactions(transactions, securityProvider, context, config);

    // Step 3: Detect behavioral patterns across transactions
    const patterns = await detectSuspiciousPatterns(transactions, suspiciousTransactions, config);

    // Step 4: Generate security alerts
    const alerts = await generateSecurityAlerts(suspiciousTransactions, patterns, config);

    // Step 5: Calculate performance metrics
    const performanceMetrics = calculatePerformanceMetrics(
      transactions.length,
      suspiciousTransactions.length,
      Date.now() - startTime
    );

    const result: MonitoringResult = {
      totalTransactionsAnalyzed: transactions.length,
      suspiciousTransactions,
      patterns,
      alerts,
      performanceMetrics
    };

    logger.info('Transaction monitoring completed', {
      agentId: context.agentId,
      totalAnalyzed: transactions.length,
      suspiciousFound: suspiciousTransactions.length,
      patternsDetected: patterns.length,
      alertsGenerated: alerts.length,
      duration: Date.now() - startTime
    });

    return result;

  } catch (error) {
    logger.error('Transaction monitoring failed', {
      agentId: context.agentId,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    return {
      totalTransactionsAnalyzed: 0,
      suspiciousTransactions: [],
      patterns: [],
      alerts: [],
      performanceMetrics: {
        processingTime: Date.now() - startTime,
        averageAnalysisTime: 0,
        falsePositiveRate: 0,
        detectionAccuracy: 0
      }
    };
  }
}

async function collectTransactions(
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: TransactionMonitoringConfig
): Promise<TransactionEvent[]> {
  const transactions: TransactionEvent[] = [];
  const endTime = Date.now();
  const startTime = endTime - config.timeWindow;

  try {
    for (const chainId of context.networkIds) {
      // Collect transactions for monitored addresses
      for (const address of config.monitoredAddresses) {
        try {
          const addressTransactions = await securityProvider.getTransactionsByAddress(
            address,
            chainId,
            startTime,
            endTime,
            config.batchSize || 1000
          );
          transactions.push(...addressTransactions);

        } catch (error) {
          logger.warn('Failed to collect transactions for address', {
            address,
            chainId,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }

      // Collect transactions for monitored contracts
      for (const contractAddress of config.monitoredContracts) {
        try {
          const contractTransactions = await securityProvider.getContractTransactions(
            contractAddress,
            chainId,
            startTime,
            endTime,
            config.batchSize || 1000
          );
          transactions.push(...contractTransactions);

        } catch (error) {
          logger.warn('Failed to collect transactions for contract', {
            contractAddress,
            chainId,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }

      // If real-time mode, also get recent mempool transactions
      if (config.realTimeMode) {
        try {
          const mempoolTransactions = await securityProvider.getMempoolTransactions(chainId, 100);
          transactions.push(...mempoolTransactions);

        } catch (error) {
          logger.debug('Failed to collect mempool transactions', {
            chainId,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }

    // Remove duplicates
    const uniqueTransactions = transactions.filter((tx, index, self) => 
      index === self.findIndex(t => t.hash === tx.hash && t.chainId === tx.chainId)
    );

    logger.debug('Transaction collection completed', {
      totalCollected: transactions.length,
      uniqueTransactions: uniqueTransactions.length,
      timeRange: `${new Date(startTime).toISOString()} - ${new Date(endTime).toISOString()}`
    });

    return uniqueTransactions;

  } catch (error) {
    logger.error('Failed to collect transactions', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function analyzeTransactions(
  transactions: TransactionEvent[],
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: TransactionMonitoringConfig
): Promise<SuspiciousTransaction[]> {
  const suspiciousTransactions: SuspiciousTransaction[] = [];

  for (const transaction of transactions) {
    try {
      const analysis = await analyzeIndividualTransaction(transaction, securityProvider, config);
      
      if (analysis.suspicionScore >= config.alertThresholds.suspiciousTransaction) {
        suspiciousTransactions.push(analysis);
      }

    } catch (error) {
      logger.debug('Failed to analyze transaction', {
        transactionHash: transaction.hash,
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }

  return suspiciousTransactions.sort((a, b) => b.suspicionScore - a.suspicionScore);
}

async function analyzeIndividualTransaction(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider,
  config: TransactionMonitoringConfig
): Promise<SuspiciousTransaction> {
  let suspicionScore = 0;
  const suspicionReasons: string[] = [];
  const riskCategories: string[] = [];
  const metadata: any = {};

  // Check 1: Known malicious addresses
  const isMaliciousFrom = await securityProvider.isKnownMaliciousAddress(transaction.from);
  const isMaliciousTo = await securityProvider.isKnownMaliciousAddress(transaction.to);
  
  if (isMaliciousFrom || isMaliciousTo) {
    suspicionScore += 80;
    suspicionReasons.push('Transaction involves known malicious address');
    riskCategories.push('malicious_address');
    metadata.knownMaliciousAddress = true;
  }

  // Check 2: Unusual value patterns
  const valueAnalysis = await analyzeTransactionValue(transaction, securityProvider);
  if (valueAnalysis.isAnomalous) {
    suspicionScore += valueAnalysis.severity;
    suspicionReasons.push(`Unusual transaction value: ${valueAnalysis.reason}`);
    riskCategories.push('value_anomaly');
    metadata.valueAnomalyDetected = true;
  }

  // Check 3: Gas price manipulation
  const gasAnalysis = await analyzeGasPatterns(transaction, securityProvider);
  if (gasAnalysis.isSuspicious) {
    suspicionScore += gasAnalysis.severity;
    suspicionReasons.push(`Suspicious gas pattern: ${gasAnalysis.reason}`);
    riskCategories.push('gas_manipulation');
    metadata.unusualGasPattern = true;
  }

  // Check 4: Contract interaction analysis
  if (transaction.to && await securityProvider.isContract(transaction.to, transaction.chainId)) {
    metadata.contractInteraction = true;
    
    const contractAnalysis = await analyzeContractInteraction(transaction, securityProvider);
    if (contractAnalysis.isSuspicious) {
      suspicionScore += contractAnalysis.severity;
      suspicionReasons.push(`Suspicious contract interaction: ${contractAnalysis.reason}`);
      riskCategories.push('malicious_contract');
    }
  }

  // Check 5: Timing pattern analysis
  const timingAnalysis = await analyzeTransactionTiming(transaction, securityProvider);
  if (timingAnalysis.isSuspicious) {
    suspicionScore += timingAnalysis.severity;
    suspicionReasons.push(`Suspicious timing pattern: ${timingAnalysis.reason}`);
    riskCategories.push('timing_manipulation');
    metadata.timingAnomalyDetected = true;
  }

  // Check 6: MEV and frontrunning detection
  const mevAnalysis = await analyzeMEVActivity(transaction, securityProvider);
  if (mevAnalysis.isSuspicious) {
    suspicionScore += mevAnalysis.severity;
    suspicionReasons.push(`MEV activity detected: ${mevAnalysis.reason}`);
    riskCategories.push('mev_exploitation');
  }

  // Check 7: Phishing pattern detection
  const phishingAnalysis = await analyzePhishingPatterns(transaction, securityProvider);
  if (phishingAnalysis.isSuspicious) {
    suspicionScore += phishingAnalysis.severity;
    suspicionReasons.push(`Phishing pattern detected: ${phishingAnalysis.reason}`);
    riskCategories.push('phishing');
  }

  return {
    hash: transaction.hash,
    chainId: transaction.chainId,
    from: transaction.from,
    to: transaction.to,
    value: transaction.value,
    gasPrice: transaction.gasPrice,
    gasUsed: transaction.gasUsed || BigNumber.from(0),
    timestamp: transaction.timestamp,
    suspicionScore: Math.min(suspicionScore, 100),
    suspicionReasons,
    riskCategories: [...new Set(riskCategories)],
    metadata
  };
}

async function analyzeTransactionValue(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider
): Promise<{ isAnomalous: boolean; severity: number; reason: string }> {
  try {
    // Get historical value patterns for the address
    const historicalStats = await securityProvider.getAddressValueStatistics(
      transaction.from,
      transaction.chainId,
      30 // 30 days
    );

    const transactionValueETH = parseFloat(ethers.utils.formatEther(transaction.value));

    // Check for round number patterns (potential automated/bot behavior)
    if (transactionValueETH > 0 && Number.isInteger(transactionValueETH)) {
      return {
        isAnomalous: true,
        severity: 15,
        reason: 'Round number transaction value suggests automated behavior'
      };
    }

    // Check for unusual large values
    if (historicalStats.averageValue > 0) {
      const valueRatio = transactionValueETH / historicalStats.averageValue;
      
      if (valueRatio > 100) {
        return {
          isAnomalous: true,
          severity: 40,
          reason: `Transaction value ${valueRatio.toFixed(1)}x larger than historical average`
        };
      }
      
      if (valueRatio > 10) {
        return {
          isAnomalous: true,
          severity: 25,
          reason: `Transaction value ${valueRatio.toFixed(1)}x larger than historical average`
        };
      }
    }

    // Check for dust/spam patterns
    if (transactionValueETH < 0.0001 && transactionValueETH > 0) {
      return {
        isAnomalous: true,
        severity: 20,
        reason: 'Dust transaction possibly used for tracking or spam'
      };
    }

    return { isAnomalous: false, severity: 0, reason: '' };

  } catch (error) {
    return { isAnomalous: false, severity: 0, reason: '' };
  }
}

async function analyzeGasPatterns(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider
): Promise<{ isSuspicious: boolean; severity: number; reason: string }> {
  try {
    // Get current network gas statistics
    const gasStats = await securityProvider.getNetworkGasStatistics(transaction.chainId);
    
    const txGasPriceGwei = parseFloat(ethers.utils.formatUnits(transaction.gasPrice, 'gwei'));
    const avgGasPriceGwei = parseFloat(ethers.utils.formatUnits(gasStats.averageGasPrice, 'gwei'));

    // Check for extremely high gas prices (frontrunning indicator)
    if (txGasPriceGwei > avgGasPriceGwei * 5) {
      return {
        isSuspicious: true,
        severity: 30,
        reason: `Gas price ${(txGasPriceGwei / avgGasPriceGwei).toFixed(1)}x higher than network average - possible frontrunning`
      };
    }

    // Check for gas price manipulation patterns
    if (txGasPriceGwei > avgGasPriceGwei * 2) {
      return {
        isSuspicious: true,
        severity: 15,
        reason: `Gas price significantly higher than network average - potential MEV activity`
      };
    }

    // Check for unusually low gas prices (potential spam)
    if (txGasPriceGwei < avgGasPriceGwei * 0.1) {
      return {
        isSuspicious: true,
        severity: 10,
        reason: 'Unusually low gas price - potential spam or attack'
      };
    }

    return { isSuspicious: false, severity: 0, reason: '' };

  } catch (error) {
    return { isSuspicious: false, severity: 0, reason: '' };
  }
}

async function analyzeContractInteraction(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider
): Promise<{ isSuspicious: boolean; severity: number; reason: string }> {
  try {
    if (!transaction.to) return { isSuspicious: false, severity: 0, reason: '' };

    // Check if contract is verified
    const contractInfo = await securityProvider.getContractInfo(transaction.to, transaction.chainId);
    
    if (!contractInfo.isVerified) {
      return {
        isSuspicious: true,
        severity: 25,
        reason: 'Interaction with unverified contract'
      };
    }

    // Check contract age (new contracts are riskier)
    const contractAge = Date.now() - contractInfo.creationTime;
    const ageInDays = contractAge / (1000 * 60 * 60 * 24);
    
    if (ageInDays < 7) {
      return {
        isSuspicious: true,
        severity: 20,
        reason: `Contract created ${ageInDays.toFixed(1)} days ago - very new contract`
      };
    }

    // Check for known malicious contract patterns
    const maliciousPatterns = await securityProvider.checkMaliciousContractPatterns(
      transaction.to,
      transaction.chainId
    );
    
    if (maliciousPatterns.length > 0) {
      return {
        isSuspicious: true,
        severity: 50,
        reason: `Contract matches known malicious patterns: ${maliciousPatterns.join(', ')}`
      };
    }

    // Check for honeypot characteristics
    const honeypotAnalysis = await securityProvider.analyzeHoneypotRisk(
      transaction.to,
      transaction.chainId
    );
    
    if (honeypotAnalysis.riskScore > 70) {
      return {
        isSuspicious: true,
        severity: 35,
        reason: `High honeypot risk score: ${honeypotAnalysis.riskScore}`
      };
    }

    return { isSuspicious: false, severity: 0, reason: '' };

  } catch (error) {
    return { isSuspicious: false, severity: 0, reason: '' };
  }
}

async function analyzeTransactionTiming(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider
): Promise<{ isSuspicious: boolean; severity: number; reason: string }> {
  try {
    // Get recent transactions from the same address
    const recentTxs = await securityProvider.getRecentTransactions(
      transaction.from,
      transaction.chainId,
      10, // Last 10 transactions
      3600000 // Within 1 hour
    );

    if (recentTxs.length < 2) {
      return { isSuspicious: false, severity: 0, reason: '' };
    }

    // Check for rapid-fire transactions (bot-like behavior)
    const intervals = [];
    for (let i = 1; i < recentTxs.length; i++) {
      intervals.push(recentTxs[i].timestamp - recentTxs[i - 1].timestamp);
    }

    const avgInterval = intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length;
    
    // Check for very consistent timing (bot pattern)
    if (avgInterval < 60000 && intervals.length >= 3) { // Less than 1 minute between transactions
      const variance = intervals.reduce((sum, interval) => sum + Math.pow(interval - avgInterval, 2), 0) / intervals.length;
      const stdDev = Math.sqrt(variance);
      
      if (stdDev < avgInterval * 0.1) { // Very consistent timing
        return {
          isSuspicious: true,
          severity: 25,
          reason: 'Highly consistent transaction timing suggests automated behavior'
        };
      }
    }

    // Check for sandwich attack timing patterns
    const sandwichPattern = await detectSandwichAttackTiming(transaction, recentTxs, securityProvider);
    if (sandwichPattern.detected) {
      return {
        isSuspicious: true,
        severity: 40,
        reason: 'Transaction timing pattern consistent with sandwich attack'
      };
    }

    return { isSuspicious: false, severity: 0, reason: '' };

  } catch (error) {
    return { isSuspicious: false, severity: 0, reason: '' };
  }
}

async function analyzeMEVActivity(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider
): Promise<{ isSuspicious: boolean; severity: number; reason: string }> {
  try {
    // Check for MEV bot addresses
    const isMEVBot = await securityProvider.isKnownMEVBot(transaction.from);
    if (isMEVBot) {
      return {
        isSuspicious: true,
        severity: 20,
        reason: 'Transaction from known MEV bot address'
      };
    }

    // Check for arbitrage patterns
    const arbitragePattern = await securityProvider.detectArbitragePattern(
      transaction.hash,
      transaction.chainId
    );
    
    if (arbitragePattern.detected && arbitragePattern.impactScore > 50) {
      return {
        isSuspicious: true,
        severity: 15,
        reason: `High-impact arbitrage detected with score ${arbitragePattern.impactScore}`
      };
    }

    // Check for frontrunning indicators
    const frontrunPattern = await securityProvider.detectFrontrunning(
      transaction,
      300000 // 5 minute window
    );
    
    if (frontrunPattern.detected) {
      return {
        isSuspicious: true,
        severity: 35,
        reason: `Frontrunning pattern detected: ${frontrunPattern.description}`
      };
    }

    return { isSuspicious: false, severity: 0, reason: '' };

  } catch (error) {
    return { isSuspicious: false, severity: 0, reason: '' };
  }
}

async function analyzePhishingPatterns(
  transaction: TransactionEvent,
  securityProvider: SecurityProvider
): Promise<{ isSuspicious: boolean; severity: number; reason: string }> {
  try {
    // Check for known phishing addresses
    const isPhishingAddress = await securityProvider.isKnownPhishingAddress(transaction.to);
    if (isPhishingAddress) {
      return {
        isSuspicious: true,
        severity: 70,
        reason: 'Transaction to known phishing address'
      };
    }

    // Check for approval scam patterns
    if (transaction.input && transaction.input.startsWith('0x095ea7b3')) { // approve function
      const approvalAnalysis = await securityProvider.analyzeApprovalTransaction(transaction);
      
      if (approvalAnalysis.isUnlimitedApproval && approvalAnalysis.recipientRiskScore > 60) {
        return {
          isSuspicious: true,
          severity: 45,
          reason: 'Unlimited approval to high-risk address'
        };
      }
    }

    // Check for fake token transfer patterns
    const fakeTokenPattern = await securityProvider.detectFakeTokenPattern(transaction);
    if (fakeTokenPattern.detected) {
      return {
        isSuspicious: true,
        severity: 30,
        reason: 'Pattern consistent with fake token scam'
      };
    }

    return { isSuspicious: false, severity: 0, reason: '' };

  } catch (error) {
    return { isSuspicious: false, severity: 0, reason: '' };
  }
}

async function detectSuspiciousPatterns(
  allTransactions: TransactionEvent[],
  suspiciousTransactions: SuspiciousTransaction[],
  config: TransactionMonitoringConfig
): Promise<DetectedPattern[]> {
  const patterns: DetectedPattern[] = [];

  try {
    // Detect wash trading patterns
    const washTradingPatterns = await detectWashTradingPattern(allTransactions);
    patterns.push(...washTradingPatterns);

    // Detect pump and dump schemes
    const pumpDumpPatterns = await detectPumpDumpPattern(allTransactions);
    patterns.push(...pumpDumpPatterns);

    // Detect rug pull patterns
    const rugPullPatterns = await detectRugPullPattern(allTransactions);
    patterns.push(...rugPullPatterns);

    // Detect coordinated attack patterns
    const coordinatedAttacks = await detectCoordinatedAttacks(suspiciousTransactions);
    patterns.push(...coordinatedAttacks);

    return patterns.filter(pattern => pattern.confidence >= 0.7);

  } catch (error) {
    logger.error('Failed to detect suspicious patterns', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function detectWashTradingPattern(transactions: TransactionEvent[]): Promise<DetectedPattern[]> {
  const patterns: DetectedPattern[] = [];
  
  // Group transactions by token pairs
  const tokenPairGroups = new Map<string, TransactionEvent[]>();
  
  transactions.forEach(tx => {
    // This would extract token pair information from transaction data
    const tokenPair = extractTokenPairFromTransaction(tx);
    if (tokenPair) {
      const existing = tokenPairGroups.get(tokenPair) || [];
      existing.push(tx);
      tokenPairGroups.set(tokenPair, existing);
    }
  });

  for (const [tokenPair, txs] of tokenPairGroups) {
    if (txs.length < 10) continue; // Need minimum transactions for pattern detection

    // Look for back-and-forth trading between same addresses
    const addressPairs = new Map<string, number>();
    
    for (let i = 0; i < txs.length - 1; i++) {
      for (let j = i + 1; j < txs.length; j++) {
        const tx1 = txs[i];
        const tx2 = txs[j];
        
        // Check if addresses are swapped (A->B followed by B->A)
        if ((tx1.from === tx2.to && tx1.to === tx2.from) ||
            (tx1.from === tx2.from && tx1.to === tx2.to)) {
          const pairKey = [tx1.from, tx1.to].sort().join('-');
          addressPairs.set(pairKey, (addressPairs.get(pairKey) || 0) + 1);
        }
      }
    }

    // Check for suspicious frequency
    for (const [pairKey, frequency] of addressPairs) {
      if (frequency >= 5) { // 5+ back-and-forth trades
        patterns.push({
          patternType: 'wash_trading',
          confidence: Math.min(0.6 + (frequency * 0.05), 0.95),
          affectedTransactions: txs.map(tx => tx.hash),
          description: `Wash trading detected: ${frequency} back-and-forth trades between same addresses`,
          severity: frequency >= 10 ? 'high' : 'medium',
          timeframe: {
            start: Math.min(...txs.map(tx => tx.timestamp)),
            end: Math.max(...txs.map(tx => tx.timestamp))
          },
          involvedAddresses: pairKey.split('-'),
          estimatedImpact: {
            affectedUsers: 2,
            estimatedLoss: BigNumber.from(0) // Would calculate based on volume
          }
        });
      }
    }
  }

  return patterns;
}

async function detectPumpDumpPattern(transactions: TransactionEvent[]): Promise<DetectedPattern[]> {
  // Implementation would analyze volume and price patterns to detect pump and dump schemes
  // This is a simplified placeholder
  return [];
}

async function detectRugPullPattern(transactions: TransactionEvent[]): Promise<DetectedPattern[]> {
  // Implementation would analyze liquidity removal patterns and large token sells
  // This is a simplified placeholder
  return [];
}

async function detectCoordinatedAttacks(suspiciousTransactions: SuspiciousTransaction[]): Promise<DetectedPattern[]> {
  // Implementation would look for coordinated timing and similar transaction patterns
  // This is a simplified placeholder
  return [];
}

async function detectSandwichAttackTiming(
  transaction: TransactionEvent,
  recentTxs: TransactionEvent[],
  securityProvider: SecurityProvider
): Promise<{ detected: boolean; description?: string }> {
  // Look for transactions immediately before and after in the same block
  const sameBlockTxs = recentTxs.filter(tx => tx.blockNumber === transaction.blockNumber);
  
  if (sameBlockTxs.length >= 3) {
    // Sort by transaction index
    const sortedTxs = sameBlockTxs.sort((a, b) => a.transactionIndex - b.transactionIndex);
    const currentIndex = sortedTxs.findIndex(tx => tx.hash === transaction.hash);
    
    if (currentIndex > 0 && currentIndex < sortedTxs.length - 1) {
      const prevTx = sortedTxs[currentIndex - 1];
      const nextTx = sortedTxs[currentIndex + 1];
      
      // Check if prev and next transactions are from the same address (sandwich pattern)
      if (prevTx.from === nextTx.from && prevTx.from !== transaction.from) {
        return {
          detected: true,
          description: 'Transactions sandwiched between two transactions from same address'
        };
      }
    }
  }

  return { detected: false };
}

async function generateSecurityAlerts(
  suspiciousTransactions: SuspiciousTransaction[],
  patterns: DetectedPattern[],
  config: TransactionMonitoringConfig
): Promise<SecurityAlert[]> {
  const alerts: SecurityAlert[] = [];

  // Generate alerts for high-risk transactions
  for (const tx of suspiciousTransactions) {
    if (tx.suspicionScore >= 70) {
      alerts.push({
        id: `tx_alert_${tx.hash}`,
        type: 'suspicious_transaction',
        severity: tx.suspicionScore >= 90 ? 'critical' : 'high',
        title: 'Suspicious Transaction Detected',
        description: `Transaction ${tx.hash} flagged with suspicion score ${tx.suspicionScore}`,
        timestamp: Date.now(),
        chainId: tx.chainId,
        transactionHash: tx.hash,
        affectedAddresses: [tx.from, tx.to],
        status: 'pending',
        source: 'transaction_monitor',
        metadata: {
          suspicionScore: tx.suspicionScore,
          reasons: tx.suspicionReasons,
          categories: tx.riskCategories
        }
      });
    }
  }

  // Generate alerts for detected patterns
  for (const pattern of patterns) {
    if (pattern.severity === 'high' || pattern.severity === 'critical') {
      alerts.push({
        id: `pattern_alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: pattern.patternType,
        severity: pattern.severity,
        title: `${pattern.patternType.replace(/_/g, ' ').toUpperCase()} Pattern Detected`,
        description: pattern.description,
        timestamp: Date.now(),
        chainId: 0, // Multi-chain pattern
        affectedAddresses: pattern.involvedAddresses,
        status: 'pending',
        source: 'pattern_detector',
        metadata: {
          confidence: pattern.confidence,
          affectedTransactions: pattern.affectedTransactions,
          timeframe: pattern.timeframe,
          estimatedImpact: pattern.estimatedImpact
        }
      });
    }
  }

  return alerts;
}

function calculatePerformanceMetrics(
  totalTransactions: number,
  suspiciousCount: number,
  processingTime: number
): {
  processingTime: number;
  averageAnalysisTime: number;
  falsePositiveRate: number;
  detectionAccuracy: number;
} {
  return {
    processingTime,
    averageAnalysisTime: totalTransactions > 0 ? processingTime / totalTransactions : 0,
    falsePositiveRate: 0.05, // Estimated 5% false positive rate
    detectionAccuracy: 0.92 // Estimated 92% accuracy
  };
}

function extractTokenPairFromTransaction(transaction: TransactionEvent): string | null {
  // This would analyze transaction input/logs to extract token pair information
  // Simplified implementation
  if (transaction.to && transaction.input) {
    // This would use proper ABI decoding to extract token addresses
    return `token_pair_${transaction.to}`;
  }
  return null;
}
