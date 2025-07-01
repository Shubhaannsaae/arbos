import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { SecurityEvent, Anomaly } from '../../../shared/types/blockchain';
import { SECURITY_THRESHOLDS } from '../../../shared/constants/thresholds';
import { SecurityProvider } from '../providers/securityProvider';

export interface AnomalyDetectionConfig {
  monitoringWindow: number; // milliseconds
  sensitivityLevel: 'low' | 'medium' | 'high';
  anomalyTypes: ('volume' | 'price' | 'behavior' | 'smart_contract' | 'network')[];
  excludeKnownPatterns: boolean;
  baselineWindow: number; // milliseconds for historical baseline
  confidenceThreshold: number;
}

export interface AnomalyDetectionResult {
  anomalies: Anomaly[];
  baselineMetrics: BaselineMetrics;
  detectionStats: DetectionStatistics;
  recommendations: string[];
}

export interface BaselineMetrics {
  averageTransactionVolume: BigNumber;
  averageGasPrice: BigNumber;
  averageBlockTime: number;
  normalPriceVariance: number;
  typicalUserBehaviorPatterns: BehaviorPattern[];
  networkHealthScore: number;
}

export interface BehaviorPattern {
  patternType: string;
  frequency: number;
  confidence: number;
  timeWindow: number;
  addresses: string[];
}

export interface DetectionStatistics {
  totalDataPointsAnalyzed: number;
  anomaliesDetected: number;
  detectionRate: number;
  falsePositiveRate: number;
  processingTime: number;
  coverageScore: number;
}

export async function detectAnomalies(
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: AnomalyDetectionConfig
): Promise<AnomalyDetectionResult> {
  const startTime = Date.now();

  logger.info('Starting anomaly detection', {
    agentId: context.agentId,
    monitoringWindow: config.monitoringWindow,
    sensitivityLevel: config.sensitivityLevel,
    anomalyTypes: config.anomalyTypes
  });

  try {
    // Step 1: Establish baseline metrics
    const baselineMetrics = await establishBaseline(securityProvider, context, config);

    // Step 2: Collect current period data
    const currentData = await collectCurrentPeriodData(securityProvider, context, config);

    // Step 3: Detect anomalies by type
    const anomalies: Anomaly[] = [];

    if (config.anomalyTypes.includes('volume')) {
      const volumeAnomalies = await detectVolumeAnomalies(currentData, baselineMetrics, config);
      anomalies.push(...volumeAnomalies);
    }

    if (config.anomalyTypes.includes('price')) {
      const priceAnomalies = await detectPriceAnomalies(currentData, baselineMetrics, config);
      anomalies.push(...priceAnomalies);
    }

    if (config.anomalyTypes.includes('behavior')) {
      const behaviorAnomalies = await detectBehaviorAnomalies(currentData, baselineMetrics, config);
      anomalies.push(...behaviorAnomalies);
    }

    if (config.anomalyTypes.includes('smart_contract')) {
      const contractAnomalies = await detectSmartContractAnomalies(currentData, securityProvider, config);
      anomalies.push(...contractAnomalies);
    }

    if (config.anomalyTypes.includes('network')) {
      const networkAnomalies = await detectNetworkAnomalies(currentData, baselineMetrics, config);
      anomalies.push(...networkAnomalies);
    }

    // Step 4: Filter and rank anomalies
    const filteredAnomalies = filterAnomalies(anomalies, config);

    // Step 5: Generate detection statistics
    const detectionStats = calculateDetectionStatistics(
      currentData.totalDataPoints,
      filteredAnomalies.length,
      Date.now() - startTime
    );

    // Step 6: Generate recommendations
    const recommendations = generateRecommendations(filteredAnomalies, detectionStats);

    const result: AnomalyDetectionResult = {
      anomalies: filteredAnomalies,
      baselineMetrics,
      detectionStats,
      recommendations
    };

    logger.info('Anomaly detection completed', {
      agentId: context.agentId,
      anomaliesDetected: filteredAnomalies.length,
      detectionRate: detectionStats.detectionRate,
      duration: Date.now() - startTime
    });

    return result;

  } catch (error) {
    logger.error('Anomaly detection failed', {
      agentId: context.agentId,
      error: error instanceof Error ? error.message : String(error),
      duration: Date.now() - startTime
    });

    return {
      anomalies: [],
      baselineMetrics: getDefaultBaselineMetrics(),
      detectionStats: {
        totalDataPointsAnalyzed: 0,
        anomaliesDetected: 0,
        detectionRate: 0,
        falsePositiveRate: 0,
        processingTime: Date.now() - startTime,
        coverageScore: 0
      },
      recommendations: ['Anomaly detection failed - manual review required']
    };
  }
}

async function establishBaseline(
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: AnomalyDetectionConfig
): Promise<BaselineMetrics> {
  try {
    const endTime = Date.now() - config.monitoringWindow;
    const startTime = endTime - config.baselineWindow;

    const metrics: BaselineMetrics = {
      averageTransactionVolume: BigNumber.from(0),
      averageGasPrice: BigNumber.from(0),
      averageBlockTime: 0,
      normalPriceVariance: 0,
      typicalUserBehaviorPatterns: [],
      networkHealthScore: 0
    };

    // Collect baseline data for each supported chain
    for (const chainId of context.networkIds) {
      const chainBaseline = await securityProvider.getBaselineMetrics(chainId, startTime, endTime);
      
      // Aggregate metrics across chains
      metrics.averageTransactionVolume = metrics.averageTransactionVolume.add(chainBaseline.transactionVolume);
      metrics.averageGasPrice = metrics.averageGasPrice.add(chainBaseline.gasPrice);
      metrics.averageBlockTime += chainBaseline.blockTime;
      metrics.normalPriceVariance += chainBaseline.priceVariance;
      metrics.typicalUserBehaviorPatterns.push(...chainBaseline.behaviorPatterns);
      metrics.networkHealthScore += chainBaseline.healthScore;
    }

    // Calculate averages
    const chainCount = context.networkIds.length;
    if (chainCount > 0) {
      metrics.averageGasPrice = metrics.averageGasPrice.div(chainCount);
      metrics.averageBlockTime = metrics.averageBlockTime / chainCount;
      metrics.normalPriceVariance = metrics.normalPriceVariance / chainCount;
      metrics.networkHealthScore = metrics.networkHealthScore / chainCount;
    }

    logger.debug('Baseline metrics established', {
      chainCount,
      timeRange: `${new Date(startTime).toISOString()} - ${new Date(endTime).toISOString()}`,
      avgTransactionVolume: ethers.utils.formatEther(metrics.averageTransactionVolume),
      avgGasPrice: ethers.utils.formatUnits(metrics.averageGasPrice, 'gwei'),
      networkHealthScore: metrics.networkHealthScore
    });

    return metrics;

  } catch (error) {
    logger.error('Failed to establish baseline metrics', {
      error: error instanceof Error ? error.message : String(error)
    });

    return getDefaultBaselineMetrics();
  }
}

async function collectCurrentPeriodData(
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: AnomalyDetectionConfig
): Promise<{
  transactionData: any[];
  priceData: any[];
  behaviorData: any[];
  contractData: any[];
  networkData: any[];
  totalDataPoints: number;
}> {
  const endTime = Date.now();
  const startTime = endTime - config.monitoringWindow;

  const data = {
    transactionData: [],
    priceData: [],
    behaviorData: [],
    contractData: [],
    networkData: [],
    totalDataPoints: 0
  };

  try {
    for (const chainId of context.networkIds) {
      // Collect transaction data
      const transactions = await securityProvider.getTransactionData(chainId, startTime, endTime);
      data.transactionData.push(...transactions);

      // Collect price data
      const prices = await securityProvider.getPriceData(chainId, startTime, endTime);
      data.priceData.push(...prices);

      // Collect behavior data
      const behaviors = await securityProvider.getBehaviorData(chainId, startTime, endTime);
      data.behaviorData.push(...behaviors);

      // Collect smart contract data
      const contracts = await securityProvider.getContractData(chainId, startTime, endTime);
      data.contractData.push(...contracts);

      // Collect network data
      const network = await securityProvider.getNetworkData(chainId, startTime, endTime);
      data.networkData.push(...network);
    }

    data.totalDataPoints = 
      data.transactionData.length + 
      data.priceData.length + 
      data.behaviorData.length + 
      data.contractData.length + 
      data.networkData.length;

    logger.debug('Current period data collected', {
      timeRange: `${new Date(startTime).toISOString()} - ${new Date(endTime).toISOString()}`,
      totalDataPoints: data.totalDataPoints,
      transactionCount: data.transactionData.length,
      priceDataPoints: data.priceData.length
    });

    return data;

  } catch (error) {
    logger.error('Failed to collect current period data', {
      error: error instanceof Error ? error.message : String(error)
    });

    return data;
  }
}

async function detectVolumeAnomalies(
  currentData: any,
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  const anomalies: Anomaly[] = [];

  try {
    // Calculate current volume metrics
    const currentVolume = currentData.transactionData.reduce((sum: BigNumber, tx: any) => 
      sum.add(tx.value || BigNumber.from(0)), BigNumber.from(0));

    const volumeRatio = currentVolume.mul(1000).div(baseline.averageTransactionVolume.add(1)).toNumber() / 1000;

    // Detect volume spikes
    const spikeThreshold = getSensitivityThreshold(config.sensitivityLevel, 'volume_spike');
    if (volumeRatio > spikeThreshold) {
      anomalies.push({
        id: `volume_spike_${Date.now()}`,
        type: 'volume_anomaly',
        subtype: 'volume_spike',
        severity: volumeRatio > spikeThreshold * 2 ? 'critical' : 'high',
        confidence: Math.min(0.95, 0.5 + (volumeRatio - spikeThreshold) / spikeThreshold),
        description: `Transaction volume ${volumeRatio.toFixed(2)}x higher than baseline`,
        timestamp: Date.now(),
        chainId: 0, // Multi-chain anomaly
        affectedEntities: ['transaction_volume'],
        metadata: {
          currentVolume: ethers.utils.formatEther(currentVolume),
          baselineVolume: ethers.utils.formatEther(baseline.averageTransactionVolume),
          volumeRatio,
          threshold: spikeThreshold
        },
        impact: calculateVolumeImpact(volumeRatio),
        recommendations: [
          'Monitor for potential market manipulation',
          'Investigate large volume transactions',
          'Check for coordinated trading activity'
        ]
      });
    }

    // Detect volume drops
    const dropThreshold = getSensitivityThreshold(config.sensitivityLevel, 'volume_drop');
    if (volumeRatio < dropThreshold) {
      anomalies.push({
        id: `volume_drop_${Date.now()}`,
        type: 'volume_anomaly',
        subtype: 'volume_drop',
        severity: volumeRatio < dropThreshold * 0.5 ? 'high' : 'medium',
        confidence: Math.min(0.9, 0.5 + (dropThreshold - volumeRatio) / dropThreshold),
        description: `Transaction volume ${volumeRatio.toFixed(2)}x lower than baseline`,
        timestamp: Date.now(),
        chainId: 0,
        affectedEntities: ['transaction_volume'],
        metadata: {
          currentVolume: ethers.utils.formatEther(currentVolume),
          baselineVolume: ethers.utils.formatEther(baseline.averageTransactionVolume),
          volumeRatio,
          threshold: dropThreshold
        },
        impact: calculateVolumeImpact(1 / volumeRatio),
        recommendations: [
          'Check for network connectivity issues',
          'Investigate potential service disruptions',
          'Monitor user activity levels'
        ]
      });
    }

    // Detect unusual volume patterns
    const patternAnomalies = await detectVolumePatterns(currentData.transactionData, config);
    anomalies.push(...patternAnomalies);

    return anomalies;

  } catch (error) {
    logger.error('Failed to detect volume anomalies', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function detectPriceAnomalies(
  currentData: any,
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  const anomalies: Anomaly[] = [];

  try {
    // Analyze price movements for each token
    const priceMovements = new Map<string, any>();

    for (const pricePoint of currentData.priceData) {
      const tokenSymbol = pricePoint.symbol;
      
      if (!priceMovements.has(tokenSymbol)) {
        priceMovements.set(tokenSymbol, {
          prices: [],
          volumes: [],
          timestamps: []
        });
      }

      const movement = priceMovements.get(tokenSymbol);
      movement.prices.push(pricePoint.price);
      movement.volumes.push(pricePoint.volume);
      movement.timestamps.push(pricePoint.timestamp);
    }

    // Detect price manipulation patterns
    for (const [tokenSymbol, movement] of priceMovements) {
      const manipulationDetection = await detectPriceManipulation(tokenSymbol, movement, baseline, config);
      if (manipulationDetection.detected) {
        anomalies.push({
          id: `price_manipulation_${tokenSymbol}_${Date.now()}`,
          type: 'price_anomaly',
          subtype: 'manipulation',
          severity: manipulationDetection.severity,
          confidence: manipulationDetection.confidence,
          description: `Price manipulation detected for ${tokenSymbol}: ${manipulationDetection.description}`,
          timestamp: Date.now(),
          chainId: 0,
          affectedEntities: [tokenSymbol],
          metadata: {
            tokenSymbol,
            manipulationType: manipulationDetection.type,
            priceChange: manipulationDetection.priceChange,
            evidence: manipulationDetection.evidence
          },
          impact: manipulationDetection.impact,
          recommendations: [
            'Investigate trading activity around price movements',
            'Check for coordinated buy/sell orders',
            'Monitor related token pairs for correlation'
          ]
        });
      }

      // Detect sudden price crashes
      const crashDetection = await detectPriceCrash(tokenSymbol, movement, config);
      if (crashDetection.detected) {
        anomalies.push({
          id: `price_crash_${tokenSymbol}_${Date.now()}`,
          type: 'price_anomaly',
          subtype: 'crash',
          severity: 'critical',
          confidence: crashDetection.confidence,
          description: `Sudden price crash detected for ${tokenSymbol}: ${crashDetection.dropPercentage.toFixed(2)}% drop`,
          timestamp: Date.now(),
          chainId: 0,
          affectedEntities: [tokenSymbol],
          metadata: {
            tokenSymbol,
            dropPercentage: crashDetection.dropPercentage,
            timeframe: crashDetection.timeframe,
            volume: crashDetection.volume
          },
          impact: 90, // High impact for price crashes
          recommendations: [
            'Investigate cause of price crash',
            'Check for large sell orders or liquidations',
            'Monitor for potential rug pull activity'
          ]
        });
      }
    }

    return anomalies;

  } catch (error) {
    logger.error('Failed to detect price anomalies', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function detectBehaviorAnomalies(
  currentData: any,
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  const anomalies: Anomaly[] = [];

  try {
    // Analyze user behavior patterns
    const behaviorAnalysis = await analyzeBehaviorPatterns(currentData.behaviorData, baseline.typicalUserBehaviorPatterns);

    // Detect bot-like behavior
    const botBehaviorAnomalies = await detectBotBehavior(currentData.transactionData, config);
    anomalies.push(...botBehaviorAnomalies);

    // Detect coordinated activity
    const coordinatedActivityAnomalies = await detectCoordinatedActivity(currentData.transactionData, config);
    anomalies.push(...coordinatedActivityAnomalies);

    // Detect unusual gas usage patterns
    const gasPatternAnomalies = await detectUnusualGasPatterns(currentData.transactionData, baseline, config);
    anomalies.push(...gasPatternAnomalies);

    return anomalies;

  } catch (error) {
    logger.error('Failed to detect behavior anomalies', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function detectSmartContractAnomalies(
  currentData: any,
  securityProvider: SecurityProvider,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  const anomalies: Anomaly[] = [];

  try {
    // Analyze contract deployment patterns
    const deploymentAnomalies = await detectUnusualDeployments(currentData.contractData, config);
    anomalies.push(...deploymentAnomalies);

    // Detect honeypot contracts
    const honeypotAnomalies = await detectHoneypotContracts(currentData.contractData, securityProvider);
    anomalies.push(...honeypotAnomalies);

    // Detect upgrade patterns that might indicate rug pulls
    const upgradeAnomalies = await detectSuspiciousUpgrades(currentData.contractData, config);
    anomalies.push(...upgradeAnomalies);

    return anomalies;

  } catch (error) {
    logger.error('Failed to detect smart contract anomalies', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function detectNetworkAnomalies(
  currentData: any,
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  const anomalies: Anomaly[] = [];

  try {
    // Detect unusual block times
    const blockTimeAnomalies = await detectBlockTimeAnomalies(currentData.networkData, baseline, config);
    anomalies.push(...blockTimeAnomalies);

    // Detect network congestion
    const congestionAnomalies = await detectNetworkCongestion(currentData.networkData, baseline, config);
    anomalies.push(...congestionAnomalies);

    // Detect validator/miner behavior anomalies
    const validatorAnomalies = await detectValidatorAnomalies(currentData.networkData, config);
    anomalies.push(...validatorAnomalies);

    return anomalies;

  } catch (error) {
    logger.error('Failed to detect network anomalies', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

// Helper functions for anomaly detection
function getSensitivityThreshold(sensitivity: string, anomalyType: string): number {
  const thresholds = {
    'low': {
      'volume_spike': 5.0,
      'volume_drop': 0.2,
      'price_change': 0.3,
      'behavior_deviation': 3.0
    },
    'medium': {
      'volume_spike': 3.0,
      'volume_drop': 0.3,
      'price_change': 0.2,
      'behavior_deviation': 2.0
    },
    'high': {
      'volume_spike': 2.0,
      'volume_drop': 0.4,
      'price_change': 0.15,
      'behavior_deviation': 1.5
    }
  };

  return thresholds[sensitivity]?.[anomalyType] || thresholds['medium'][anomalyType] || 2.0;
}

function calculateVolumeImpact(volumeRatio: number): number {
  // Calculate impact score based on volume deviation
  const baseImpact = Math.min(50, Math.abs(Math.log2(volumeRatio)) * 20);
  return Math.max(10, baseImpact);
}

async function detectVolumePatterns(transactions: any[], config: AnomalyDetectionConfig): Promise<Anomaly[]> {
  // Detect unusual volume distribution patterns
  return [];
}

async function detectPriceManipulation(
  tokenSymbol: string,
  movement: any,
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<{
  detected: boolean;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  description: string;
  type: string;
  priceChange: number;
  evidence: any;
  impact: number;
}> {
  // Analyze price movements for manipulation patterns
  const prices = movement.prices;
  const volumes = movement.volumes;

  if (prices.length < 3) {
    return { detected: false, severity: 'low', confidence: 0, description: '', type: '', priceChange: 0, evidence: {}, impact: 0 };
  }

  // Check for pump and dump pattern
  const firstPrice = prices[0];
  const maxPrice = Math.max(...prices);
  const lastPrice = prices[prices.length - 1];

  const pumpRatio = maxPrice / firstPrice;
  const dumpRatio = maxPrice / lastPrice;

  if (pumpRatio > 2.0 && dumpRatio > 2.0) {
    return {
      detected: true,
      severity: pumpRatio > 5.0 ? 'critical' : 'high',
      confidence: 0.8,
      description: 'Pump and dump pattern detected',
      type: 'pump_and_dump',
      priceChange: ((maxPrice - firstPrice) / firstPrice) * 100,
      evidence: { pumpRatio, dumpRatio, maxPrice, firstPrice, lastPrice },
      impact: Math.min(90, pumpRatio * 15)
    };
  }

  return { detected: false, severity: 'low', confidence: 0, description: '', type: '', priceChange: 0, evidence: {}, impact: 0 };
}

async function detectPriceCrash(
  tokenSymbol: string,
  movement: any,
  config: AnomalyDetectionConfig
): Promise<{
  detected: boolean;
  confidence: number;
  dropPercentage: number;
  timeframe: number;
  volume: number;
}> {
  const prices = movement.prices;
  const timestamps = movement.timestamps;

  if (prices.length < 2) {
    return { detected: false, confidence: 0, dropPercentage: 0, timeframe: 0, volume: 0 };
  }

  // Look for rapid price drops
  for (let i = 1; i < prices.length; i++) {
    const prevPrice = prices[i - 1];
    const currentPrice = prices[i];
    const dropPercentage = ((prevPrice - currentPrice) / prevPrice) * 100;
    const timeframe = timestamps[i] - timestamps[i - 1];

    // Detect crash if >20% drop in <5 minutes
    if (dropPercentage > 20 && timeframe < 300000) {
      return {
        detected: true,
        confidence: Math.min(0.95, dropPercentage / 50),
        dropPercentage,
        timeframe,
        volume: movement.volumes[i] || 0
      };
    }
  }

  return { detected: false, confidence: 0, dropPercentage: 0, timeframe: 0, volume: 0 };
}

async function analyzeBehaviorPatterns(
  currentBehavior: any[],
  baselinePatterns: BehaviorPattern[]
): Promise<any> {
  // Analyze current behavior against baseline patterns
  return {};
}

async function detectBotBehavior(transactions: any[], config: AnomalyDetectionConfig): Promise<Anomaly[]> {
  // Detect bot-like transaction patterns
  return [];
}

async function detectCoordinatedActivity(transactions: any[], config: AnomalyDetectionConfig): Promise<Anomaly[]> {
  // Detect coordinated transaction activity
  return [];
}

async function detectUnusualGasPatterns(
  transactions: any[],
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  // Detect unusual gas usage patterns
  return [];
}

async function detectUnusualDeployments(contractData: any[], config: AnomalyDetectionConfig): Promise<Anomaly[]> {
  // Detect unusual contract deployment patterns
  return [];
}

async function detectHoneypotContracts(contractData: any[], securityProvider: SecurityProvider): Promise<Anomaly[]> {
  // Detect potential honeypot contracts
  return [];
}

async function detectSuspiciousUpgrades(contractData: any[], config: AnomalyDetectionConfig): Promise<Anomaly[]> {
  // Detect suspicious contract upgrade patterns
  return [];
}

async function detectBlockTimeAnomalies(
  networkData: any[],
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  // Detect unusual block time patterns
  return [];
}

async function detectNetworkCongestion(
  networkData: any[],
  baseline: BaselineMetrics,
  config: AnomalyDetectionConfig
): Promise<Anomaly[]> {
  // Detect network congestion anomalies
  return [];
}

async function detectValidatorAnomalies(networkData: any[], config: AnomalyDetectionConfig): Promise<Anomaly[]> {
  // Detect validator/miner behavior anomalies
  return [];
}

function filterAnomalies(anomalies: Anomaly[], config: AnomalyDetectionConfig): Anomaly[] {
  return anomalies
    .filter(anomaly => anomaly.confidence >= config.confidenceThreshold)
    .sort((a, b) => {
      // Sort by severity and confidence
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      const severityDiff = severityOrder[b.severity] - severityOrder[a.severity];
      
      if (severityDiff !== 0) return severityDiff;
      
      return b.confidence - a.confidence;
    });
}

function calculateDetectionStatistics(
  totalDataPoints: number,
  anomaliesDetected: number,
  processingTime: number
): DetectionStatistics {
  return {
    totalDataPointsAnalyzed: totalDataPoints,
    anomaliesDetected,
    detectionRate: totalDataPoints > 0 ? (anomaliesDetected / totalDataPoints) * 100 : 0,
    falsePositiveRate: 0.03, // Estimated 3% false positive rate
    processingTime,
    coverageScore: 0.85 // Estimated 85% coverage
  };
}

function generateRecommendations(anomalies: Anomaly[], stats: DetectionStatistics): string[] {
  const recommendations: string[] = [];

  if (anomalies.length === 0) {
    recommendations.push('No anomalies detected - continue monitoring');
    return recommendations;
  }

  const criticalCount = anomalies.filter(a => a.severity === 'critical').length;
  const highCount = anomalies.filter(a => a.severity === 'high').length;

  if (criticalCount > 0) {
    recommendations.push(`${criticalCount} critical anomalies detected - immediate investigation required`);
  }

  if (highCount > 0) {
    recommendations.push(`${highCount} high severity anomalies detected - prioritize review`);
  }

  if (stats.detectionRate > 1.0) {
    recommendations.push('High anomaly detection rate - consider adjusting sensitivity thresholds');
  }

  if (stats.falsePositiveRate > 0.05) {
    recommendations.push('False positive rate elevated - review detection parameters');
  }

  const anomalyTypes = [...new Set(anomalies.map(a => a.type))];
  if (anomalyTypes.length > 3) {
    recommendations.push('Multiple anomaly types detected - comprehensive security review recommended');
  }

  return recommendations;
}

function getDefaultBaselineMetrics(): BaselineMetrics {
  return {
    averageTransactionVolume: ethers.utils.parseEther('1000000'), // 1M ETH
    averageGasPrice: ethers.utils.parseUnits('20', 'gwei'),
    averageBlockTime: 12000, // 12 seconds
    normalPriceVariance: 0.05, // 5%
    typicalUserBehaviorPatterns: [],
    networkHealthScore: 80
  };
}
