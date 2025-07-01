import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { Anomaly, SecurityEvent } from '../../../shared/types/blockchain';
import { SECURITY_THRESHOLDS } from '../../../shared/constants/thresholds';

export interface SecurityRiskAssessment {
  overallRiskScore: number;
  riskLevel: 'very_low' | 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  riskCategories: RiskCategory[];
  mitigationStrategies: MitigationStrategy[];
  timeToAction: number; // milliseconds
  impactAnalysis: ImpactAnalysis;
  recommendations: string[];
  escalationRequired: boolean;
}

export interface RiskCategory {
  category: string;
  score: number;
  weight: number;
  contributing_factors: string[];
  evidence: any[];
  trend: 'increasing' | 'stable' | 'decreasing';
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface MitigationStrategy {
  strategy: string;
  priority: 'immediate' | 'urgent' | 'normal' | 'low';
  effectiveness: number; // 0-100
  implementationComplexity: 'low' | 'medium' | 'high';
  estimatedCost: string;
  timeToImplement: number; // hours
  prerequisites: string[];
  expectedOutcome: string;
}

export interface ImpactAnalysis {
  financialImpact: {
    estimatedLoss: BigNumber;
    affectedValue: BigNumber;
    recoveryLikelihood: number;
  };
  operationalImpact: {
    systemsAffected: string[];
    downtimeRisk: number;
    userImpact: number;
  };
  reputationalImpact: {
    severityLevel: number;
    stakeholdersAffected: string[];
    recoveryTime: number;
  };
  legalImpact: {
    complianceViolations: string[];
    regulatoryRisk: number;
    liabilityExposure: number;
  };
}

export interface AnomalyRiskAnalysis {
  anomalyId: string;
  baselineDeviation: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  falsePositiveProbability: number;
  potentialImpact: number;
  urgency: number;
  correlatedAnomalies: string[];
  rootCauseAnalysis: RootCauseAnalysis;
}

export interface RootCauseAnalysis {
  primaryCause: string;
  contributingFactors: string[];
  systemicIssues: string[];
  preventionMeasures: string[];
  detectionGaps: string[];
}

export interface ThreatContext {
  threatActors: string[];
  attackVectors: string[];
  motivations: string[];
  capabilities: string[];
  targetedAssets: string[];
  geopoliticalFactors: string[];
}

export class SecurityRiskEvaluator {
  private riskModelWeights: Map<string, number> = new Map();
  private historicalRiskData: Map<string, any[]> = new Map();
  private threatIntelligence: Map<string, any> = new Map();

  constructor() {
    this.initializeRiskModel();
  }

  async evaluateAnomaly(anomaly: Anomaly): Promise<AnomalyRiskAnalysis> {
    const startTime = Date.now();

    try {
      logger.debug('Starting anomaly risk evaluation', {
        anomalyId: anomaly.id,
        anomalyType: anomaly.type,
        severity: anomaly.severity
      });

      // Step 1: Calculate baseline deviation
      const baselineDeviation = await this.calculateBaselineDeviation(anomaly);

      // Step 2: Assess anomaly severity and confidence
      const { severity, confidence } = this.assessAnomalySeverity(anomaly, baselineDeviation);

      // Step 3: Calculate false positive probability
      const falsePositiveProbability = await this.calculateFalsePositiveProbability(anomaly);

      // Step 4: Assess potential impact
      const potentialImpact = await this.assessPotentialImpact(anomaly);

      // Step 5: Calculate urgency
      const urgency = this.calculateUrgency(anomaly, severity, potentialImpact);

      // Step 6: Find correlated anomalies
      const correlatedAnomalies = await this.findCorrelatedAnomalies(anomaly);

      // Step 7: Perform root cause analysis
      const rootCauseAnalysis = await this.performRootCauseAnalysis(anomaly, correlatedAnomalies);

      const analysis: AnomalyRiskAnalysis = {
        anomalyId: anomaly.id,
        baselineDeviation,
        severity,
        confidence,
        falsePositiveProbability,
        potentialImpact,
        urgency,
        correlatedAnomalies,
        rootCauseAnalysis
      };

      logger.debug('Anomaly risk evaluation completed', {
        anomalyId: anomaly.id,
        severity,
        confidence,
        potentialImpact,
        duration: Date.now() - startTime
      });

      return analysis;

    } catch (error) {
      logger.error('Anomaly risk evaluation failed', {
        anomalyId: anomaly.id,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime
      });

      return {
        anomalyId: anomaly.id,
        baselineDeviation: 0,
        severity: 'low',
        confidence: 0,
        falsePositiveProbability: 1.0,
        potentialImpact: 0,
        urgency: 0,
        correlatedAnomalies: [],
        rootCauseAnalysis: {
          primaryCause: 'Analysis failed',
          contributingFactors: [],
          systemicIssues: [],
          preventionMeasures: [],
          detectionGaps: ['Risk evaluation failure']
        }
      };
    }
  }

  async evaluateSecurityEvent(event: SecurityEvent): Promise<SecurityRiskAssessment> {
    const startTime = Date.now();

    try {
      logger.debug('Starting security event risk evaluation', {
        eventId: event.id,
        eventType: event.type,
        severity: event.severity
      });

      // Step 1: Analyze risk categories
      const riskCategories = await this.analyzeRiskCategories(event);

      // Step 2: Calculate overall risk score
      const overallRiskScore = this.calculateOverallRiskScore(riskCategories);

      // Step 3: Determine risk level
      const riskLevel = this.determineRiskLevel(overallRiskScore);

      // Step 4: Calculate confidence
      const confidence = this.calculateConfidence(riskCategories, event);

      // Step 5: Generate mitigation strategies
      const mitigationStrategies = await this.generateMitigationStrategies(event, riskCategories);

      // Step 6: Calculate time to action
      const timeToAction = this.calculateTimeToAction(riskLevel, event.severity);

      // Step 7: Perform impact analysis
      const impactAnalysis = await this.performImpactAnalysis(event, riskCategories);

      // Step 8: Generate recommendations
      const recommendations = this.generateRecommendations(event, riskCategories, impactAnalysis);

      // Step 9: Determine escalation requirement
      const escalationRequired = this.shouldEscalate(riskLevel, impactAnalysis, event);

      const assessment: SecurityRiskAssessment = {
        overallRiskScore,
        riskLevel,
        confidence,
        riskCategories,
        mitigationStrategies,
        timeToAction,
        impactAnalysis,
        recommendations,
        escalationRequired
      };

      logger.debug('Security event risk evaluation completed', {
        eventId: event.id,
        overallRiskScore,
        riskLevel,
        escalationRequired,
        duration: Date.now() - startTime
      });

      return assessment;

    } catch (error) {
      logger.error('Security event risk evaluation failed', {
        eventId: event.id,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime
      });

      return this.getDefaultRiskAssessment();
    }
  }

  async evaluateThreatContext(events: SecurityEvent[]): Promise<ThreatContext> {
    try {
      const threatActors = await this.identifyThreatActors(events);
      const attackVectors = await this.identifyAttackVectors(events);
      const motivations = await this.analyzeThreatMotivations(events);
      const capabilities = await this.assessThreatCapabilities(events);
      const targetedAssets = await this.identifyTargetedAssets(events);
      const geopoliticalFactors = await this.analyzeGeopoliticalFactors(events);

      return {
        threatActors,
        attackVectors,
        motivations,
        capabilities,
        targetedAssets,
        geopoliticalFactors
      };

    } catch (error) {
      logger.error('Threat context evaluation failed', {
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        threatActors: [],
        attackVectors: [],
        motivations: [],
        capabilities: [],
        targetedAssets: [],
        geopoliticalFactors: []
      };
    }
  }

  private async calculateBaselineDeviation(anomaly: Anomaly): Promise<number> {
    try {
      const historicalData = this.historicalRiskData.get(anomaly.type) || [];
      
      if (historicalData.length < 10) {
        return 50; // Default deviation when insufficient data
      }

      // Calculate statistical deviation from baseline
      const values = historicalData.map(data => data.value);
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      const stdDev = Math.sqrt(variance);

      const currentValue = this.extractAnomalyValue(anomaly);
      const deviation = Math.abs(currentValue - mean) / (stdDev || 1);

      // Convert to percentage
      return Math.min(deviation * 10, 100);

    } catch (error) {
      logger.warn('Failed to calculate baseline deviation', {
        anomalyId: anomaly.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return 50; // Default deviation
    }
  }

  private assessAnomalySeverity(anomaly: Anomaly, baselineDeviation: number): {
    severity: 'low' | 'medium' | 'high' | 'critical';
    confidence: number;
  } {
    let severity: 'low' | 'medium' | 'high' | 'critical' = 'low';
    let confidence = 0.5;

    // Base severity from anomaly
    const baseSeverity = anomaly.severity;
    
    // Adjust based on baseline deviation
    if (baselineDeviation > 80) {
      severity = 'critical';
      confidence = 0.9;
    } else if (baselineDeviation > 60) {
      severity = 'high';
      confidence = 0.8;
    } else if (baselineDeviation > 40) {
      severity = 'medium';
      confidence = 0.7;
    } else {
      severity = 'low';
      confidence = 0.6;
    }

    // Consider original anomaly confidence
    confidence = Math.max(confidence, anomaly.confidence);

    return { severity, confidence };
  }

  private async calculateFalsePositiveProbability(anomaly: Anomaly): Promise<number> {
    try {
      // Base false positive rate by anomaly type
      const baseRates: Record<string, number> = {
        'volume_anomaly': 0.05,
        'price_anomaly': 0.08,
        'behavior_anomaly': 0.12,
        'network_anomaly': 0.15,
        'contract_anomaly': 0.10
      };

      let falsePositiveRate = baseRates[anomaly.type] || 0.10;

      // Adjust based on confidence
      falsePositiveRate *= (1 - anomaly.confidence);

      // Adjust based on impact
      if (anomaly.impact > 80) {
        falsePositiveRate *= 0.5; // Lower FP rate for high impact
      }

      // Historical correction
      const historicalAccuracy = await this.getHistoricalAccuracy(anomaly.type);
      falsePositiveRate *= (1 - historicalAccuracy);

      return Math.max(0.01, Math.min(0.5, falsePositiveRate));

    } catch (error) {
      return 0.15; // Default 15% false positive rate
    }
  }

  private async assessPotentialImpact(anomaly: Anomaly): Promise<number> {
    try {
      let impact = anomaly.impact || 50;

      // Adjust based on affected entities
      const entityCount = anomaly.affectedEntities?.length || 1;
      impact += Math.min(entityCount * 5, 30);

      // Adjust based on anomaly type criticality
      const criticalTypes = ['price_anomaly', 'contract_anomaly'];
      if (criticalTypes.includes(anomaly.type)) {
        impact += 20;
      }

      // Consider temporal factors
      const recentSimilarAnomalies = await this.getRecentSimilarAnomalies(anomaly);
      if (recentSimilarAnomalies.length > 2) {
        impact += 15; // Pattern indicates escalating threat
      }

      return Math.min(impact, 100);

    } catch (error) {
      return anomaly.impact || 50;
    }
  }

  private calculateUrgency(
    anomaly: Anomaly, 
    severity: string, 
    potentialImpact: number
  ): number {
    const severityScores = {
      'low': 20,
      'medium': 40,
      'high': 70,
      'critical': 90
    };

    const baseSeverityScore = severityScores[severity] || 20;
    const impactScore = potentialImpact;
    const confidenceBonus = anomaly.confidence * 10;

    const urgency = (baseSeverityScore * 0.5) + (impactScore * 0.4) + (confidenceBonus * 0.1);

    return Math.min(urgency, 100);
  }

  private async findCorrelatedAnomalies(anomaly: Anomaly): Promise<string[]> {
    try {
      // Find anomalies with similar characteristics, timing, or affected entities
      const correlatedIds: string[] = [];

      // Check temporal correlation (within 1 hour)
      const timeWindow = 3600000; // 1 hour
      const similarTimeAnomalies = await this.getAnomaliesInTimeWindow(
        anomaly.timestamp - timeWindow,
        anomaly.timestamp + timeWindow
      );

      // Check entity correlation
      const entityCorrelatedAnomalies = await this.getAnomaliesByEntities(
        anomaly.affectedEntities || []
      );

      // Check type correlation
      const typeCorrelatedAnomalies = await this.getAnomaliesByType(anomaly.type);

      const allCorrelated = [
        ...similarTimeAnomalies,
        ...entityCorrelatedAnomalies,
        ...typeCorrelatedAnomalies
      ];

      // Remove duplicates and self
      const uniqueCorrelated = [...new Set(allCorrelated)]
        .filter(id => id !== anomaly.id);

      return uniqueCorrelated.slice(0, 10); // Limit to top 10 correlations

    } catch (error) {
      logger.warn('Failed to find correlated anomalies', {
        anomalyId: anomaly.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  private async performRootCauseAnalysis(
    anomaly: Anomaly,
    correlatedAnomalies: string[]
  ): Promise<RootCauseAnalysis> {
    try {
      const primaryCause = await this.identifyPrimaryCause(anomaly);
      const contributingFactors = await this.identifyContributingFactors(anomaly, correlatedAnomalies);
      const systemicIssues = await this.identifySystemicIssues(anomaly, correlatedAnomalies);
      const preventionMeasures = this.generatePreventionMeasures(primaryCause, contributingFactors);
      const detectionGaps = await this.identifyDetectionGaps(anomaly);

      return {
        primaryCause,
        contributingFactors,
        systemicIssues,
        preventionMeasures,
        detectionGaps
      };

    } catch (error) {
      logger.error('Root cause analysis failed', {
        anomalyId: anomaly.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        primaryCause: 'Unknown - analysis failed',
        contributingFactors: [],
        systemicIssues: [],
        preventionMeasures: ['Improve detection capabilities'],
        detectionGaps: ['Root cause analysis failure']
      };
    }
  }

  private async analyzeRiskCategories(event: SecurityEvent): Promise<RiskCategory[]> {
    const categories: RiskCategory[] = [];

    try {
      // Technical Risk
      const technicalRisk = await this.assessTechnicalRisk(event);
      categories.push({
        category: 'technical',
        score: technicalRisk.score,
        weight: 0.3,
        contributing_factors: technicalRisk.factors,
        evidence: technicalRisk.evidence,
        trend: technicalRisk.trend,
        severity: this.scoresToSeverity(technicalRisk.score)
      });

      // Operational Risk
      const operationalRisk = await this.assessOperationalRisk(event);
      categories.push({
        category: 'operational',
        score: operationalRisk.score,
        weight: 0.25,
        contributing_factors: operationalRisk.factors,
        evidence: operationalRisk.evidence,
        trend: operationalRisk.trend,
        severity: this.scoresToSeverity(operationalRisk.score)
      });

      // Financial Risk
      const financialRisk = await this.assessFinancialRisk(event);
      categories.push({
        category: 'financial',
        score: financialRisk.score,
        weight: 0.25,
        contributing_factors: financialRisk.factors,
        evidence: financialRisk.evidence,
        trend: financialRisk.trend,
        severity: this.scoresToSeverity(financialRisk.score)
      });

      // Reputational Risk
      const reputationalRisk = await this.assessReputationalRisk(event);
      categories.push({
        category: 'reputational',
        score: reputationalRisk.score,
        weight: 0.1,
        contributing_factors: reputationalRisk.factors,
        evidence: reputationalRisk.evidence,
        trend: reputationalRisk.trend,
        severity: this.scoresToSeverity(reputationalRisk.score)
      });

      // Compliance Risk
      const complianceRisk = await this.assessComplianceRisk(event);
      categories.push({
        category: 'compliance',
        score: complianceRisk.score,
        weight: 0.1,
        contributing_factors: complianceRisk.factors,
        evidence: complianceRisk.evidence,
        trend: complianceRisk.trend,
        severity: this.scoresToSeverity(complianceRisk.score)
      });

    } catch (error) {
      logger.error('Risk category analysis failed', {
        eventId: event.id,
        error: error instanceof Error ? error.message : String(error)
      });
    }

    return categories;
  }

  private calculateOverallRiskScore(riskCategories: RiskCategory[]): number {
    if (riskCategories.length === 0) return 0;

    const weightedScore = riskCategories.reduce((sum, category) => {
      return sum + (category.score * category.weight);
    }, 0);

    const totalWeight = riskCategories.reduce((sum, category) => sum + category.weight, 0);

    return totalWeight > 0 ? weightedScore / totalWeight : 0;
  }

  private determineRiskLevel(riskScore: number): 'very_low' | 'low' | 'medium' | 'high' | 'critical' {
    if (riskScore >= 80) return 'critical';
    if (riskScore >= 60) return 'high';
    if (riskScore >= 40) return 'medium';
    if (riskScore >= 20) return 'low';
    return 'very_low';
  }

  private calculateConfidence(riskCategories: RiskCategory[], event: SecurityEvent): number {
    const evidenceStrength = riskCategories.reduce((sum, category) => {
      return sum + (category.evidence.length * category.weight);
    }, 0);

    const categoryConsistency = this.calculateCategoryConsistency(riskCategories);
    const eventReliability = this.assessEventReliability(event);

    return Math.min(
      (evidenceStrength * 0.4) + (categoryConsistency * 0.3) + (eventReliability * 0.3),
      0.95
    );
  }

  private async generateMitigationStrategies(
    event: SecurityEvent,
    riskCategories: RiskCategory[]
  ): Promise<MitigationStrategy[]> {
    const strategies: MitigationStrategy[] = [];

    try {
      // Generate strategies based on event type and risk categories
      for (const category of riskCategories) {
        if (category.score > 50) {
          const categoryStrategies = await this.generateCategoryMitigationStrategies(
            category,
            event
          );
          strategies.push(...categoryStrategies);
        }
      }

      // Add general strategies based on event type
      const generalStrategies = await this.generateGeneralMitigationStrategies(event);
      strategies.push(...generalStrategies);

      // Sort by priority and effectiveness
      return strategies
        .sort((a, b) => {
          const priorityOrder = { immediate: 4, urgent: 3, normal: 2, low: 1 };
          const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
          
          if (priorityDiff !== 0) return priorityDiff;
          
          return b.effectiveness - a.effectiveness;
        })
        .slice(0, 10); // Limit to top 10 strategies

    } catch (error) {
      logger.error('Mitigation strategy generation failed', {
        eventId: event.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  private calculateTimeToAction(
    riskLevel: string,
    eventSeverity: string
  ): number {
    const urgencyMatrix = {
      'critical': { 'critical': 0, 'high': 300000, 'medium': 900000, 'low': 1800000 }, // 0min, 5min, 15min, 30min
      'high': { 'critical': 300000, 'high': 900000, 'medium': 3600000, 'low': 7200000 }, // 5min, 15min, 1hr, 2hr
      'medium': { 'critical': 900000, 'high': 3600000, 'medium': 14400000, 'low': 86400000 }, // 15min, 1hr, 4hr, 24hr
      'low': { 'critical': 3600000, 'high': 14400000, 'medium': 86400000, 'low': 604800000 }, // 1hr, 4hr, 24hr, 7days
      'very_low': { 'critical': 14400000, 'high': 86400000, 'medium': 604800000, 'low': 2592000000 } // 4hr, 24hr, 7days, 30days
    };

    return urgencyMatrix[riskLevel]?.[eventSeverity] || 3600000; // Default 1 hour
  }

  private async performImpactAnalysis(
    event: SecurityEvent,
    riskCategories: RiskCategory[]
  ): Promise<ImpactAnalysis> {
    try {
      const financialImpact = await this.assessFinancialImpact(event, riskCategories);
      const operationalImpact = await this.assessOperationalImpact(event, riskCategories);
      const reputationalImpact = await this.assessReputationalImpact(event, riskCategories);
      const legalImpact = await this.assessLegalImpact(event, riskCategories);

      return {
        financialImpact,
        operationalImpact,
        reputationalImpact,
        legalImpact
      };

    } catch (error) {
      logger.error('Impact analysis failed', {
        eventId: event.id,
        error: error instanceof Error ? error.message : String(error)
      });

      return this.getDefaultImpactAnalysis();
    }
  }

  private generateRecommendations(
    event: SecurityEvent,
    riskCategories: RiskCategory[],
    impactAnalysis: ImpactAnalysis
  ): string[] {
    const recommendations: string[] = [];

    // Risk level based recommendations
    const highRiskCategories = riskCategories.filter(cat => cat.score > 70);
    if (highRiskCategories.length > 0) {
      recommendations.push('Immediate escalation to security team required');
      recommendations.push('Activate incident response procedures');
    }

    // Financial impact recommendations
    if (impactAnalysis.financialImpact.estimatedLoss.gt(ethers.utils.parseEther('100'))) {
      recommendations.push('Consider emergency fund allocation for incident response');
      recommendations.push('Engage financial risk management team');
    }

    // Operational impact recommendations
    if (impactAnalysis.operationalImpact.downtimeRisk > 50) {
      recommendations.push('Prepare backup systems and failover procedures');
      recommendations.push('Brief operations team on potential service disruptions');
    }

    // Event type specific recommendations
    if (event.type.includes('fraud')) {
      recommendations.push('Report to relevant authorities if required');
      recommendations.push('Preserve evidence for potential legal proceedings');
    }

    if (event.type.includes('anomaly')) {
      recommendations.push('Investigate root cause to prevent recurrence');
      recommendations.push('Update detection algorithms based on findings');
    }

    // General recommendations
    recommendations.push('Document all response actions for post-incident review');
    recommendations.push('Monitor for similar events in the coming 48 hours');

    return [...new Set(recommendations)]; // Remove duplicates
  }

  private shouldEscalate(
    riskLevel: string,
    impactAnalysis: ImpactAnalysis,
    event: SecurityEvent
  ): boolean {
    // Always escalate critical risks
    if (riskLevel === 'critical') return true;

    // Escalate high financial impact
    if (impactAnalysis.financialImpact.estimatedLoss.gt(ethers.utils.parseEther('50'))) {
      return true;
    }

    // Escalate high operational impact
    if (impactAnalysis.operationalImpact.downtimeRisk > 70) {
      return true;
    }

    // Escalate compliance violations
    if (impactAnalysis.legalImpact.complianceViolations.length > 0) {
      return true;
    }

    // Escalate certain event types
    const escalationEventTypes = ['fraud_detected', 'security_breach', 'data_leak', 'system_compromise'];
    if (escalationEventTypes.some(type => event.type.includes(type))) {
      return true;
    }

    return false;
  }

  // Helper methods for specific risk assessments
  private async assessTechnicalRisk(event: SecurityEvent): Promise<{
    score: number;
    factors: string[];
    evidence: any[];
    trend: 'increasing' | 'stable' | 'decreasing';
  }> {
    const factors: string[] = [];
    const evidence: any[] = [];
    let score = 30; // Base technical risk

    // Assess based on event type
    if (event.type.includes('contract')) {
      score += 25;
      factors.push('Smart contract involvement');
    }

    if (event.type.includes('network')) {
      score += 20;
      factors.push('Network level impact');
    }

    // Assess system complexity
    const affectedSystems = event.affectedAddresses?.length || 1;
    if (affectedSystems > 5) {
      score += 15;
      factors.push('Multiple systems affected');
    }

    // Historical trend analysis
    const trend = await this.analyzeTechnicalRiskTrend(event.type);

    return {
      score: Math.min(score, 100),
      factors,
      evidence,
      trend
    };
  }

  private async assessOperationalRisk(event: SecurityEvent): Promise<{
    score: number;
    factors: string[];
    evidence: any[];
    trend: 'increasing' | 'stable' | 'decreasing';
  }> {
    const factors: string[] = [];
    const evidence: any[] = [];
    let score = 25; // Base operational risk

    // Service disruption potential
    if (event.severity === 'critical' || event.severity === 'high') {
      score += 30;
      factors.push('High potential for service disruption');
    }

    // Time of occurrence (outside business hours = higher operational risk)
    const eventHour = new Date(event.timestamp).getHours();
    if (eventHour < 6 || eventHour > 22) {
      score += 10;
      factors.push('Occurred outside business hours');
    }

    const trend = await this.analyzeOperationalRiskTrend(event.type);

    return {
      score: Math.min(score, 100),
      factors,
      evidence,
      trend
    };
  }

  private async assessFinancialRisk(event: SecurityEvent): Promise<{
    score: number;
    factors: string[];
    evidence: any[];
    trend: 'increasing' | 'stable' | 'decreasing';
  }> {
    const factors: string[] = [];
    const evidence: any[] = [];
    let score = 20; // Base financial risk

    // Direct financial events
    if (event.type.includes('fraud') || event.type.includes('theft')) {
      score += 40;
      factors.push('Direct financial threat detected');
    }

    // Market impact events
    if (event.type.includes('price') || event.type.includes('market')) {
      score += 25;
      factors.push('Market impact potential');
    }

    const trend = await this.analyzeFinancialRiskTrend(event.type);

    return {
      score: Math.min(score, 100),
      factors,
      evidence,
      trend
    };
  }

  private async assessReputationalRisk(event: SecurityEvent): Promise<{
    score: number;
    factors: string[];
    evidence: any[];
    trend: 'increasing' | 'stable' | 'decreasing';
  }> {
    const factors: string[] = [];
    const evidence: any[] = [];
    let score = 15; // Base reputational risk

    // Public visibility
    if (event.type.includes('breach') || event.type.includes('hack')) {
      score += 35;
      factors.push('High public visibility risk');
    }

    // User impact
    if (event.affectedAddresses && event.affectedAddresses.length > 100) {
      score += 20;
      factors.push('Large number of users affected');
    }

    const trend = await this.analyzeReputationalRiskTrend(event.type);

    return {
      score: Math.min(score, 100),
      factors,
      evidence,
      trend
    };
  }

  private async assessComplianceRisk(event: SecurityEvent): Promise<{
    score: number;
    factors: string[];
    evidence: any[];
    trend: 'increasing' | 'stable' | 'decreasing';
  }> {
    const factors: string[] = [];
    const evidence: any[] = [];
    let score = 10; // Base compliance risk

    // Regulatory reporting requirements
    if (event.type.includes('breach') || event.type.includes('fraud')) {
      score += 25;
      factors.push('May require regulatory reporting');
    }

    // Data protection implications
    if (event.type.includes('data') || event.type.includes('privacy')) {
      score += 20;
      factors.push('Data protection regulation implications');
    }

    const trend = await this.analyzeComplianceRiskTrend(event.type);

    return {
      score: Math.min(score, 100),
      factors,
      evidence,
      trend
    };
  }

  // Additional helper methods
  private extractAnomalyValue(anomaly: Anomaly): number {
    // Extract numerical value from anomaly for statistical analysis
    if (anomaly.metadata?.value) return anomaly.metadata.value;
    if (anomaly.impact) return anomaly.impact;
    return 50; // Default value
  }

  private async getHistoricalAccuracy(anomalyType: string): Promise<number> {
    // Get historical accuracy for this anomaly type
    return 0.85; // Default 85% accuracy
  }

  private async getRecentSimilarAnomalies(anomaly: Anomaly): Promise<any[]> {
    // Get similar anomalies from recent history
    return [];
  }

  private async getAnomaliesInTimeWindow(startTime: number, endTime: number): Promise<string[]> {
    // Get anomalies within time window
    return [];
  }

  private async getAnomaliesByEntities(entities: string[]): Promise<string[]> {
    // Get anomalies affecting same entities
    return [];
  }

  private async getAnomaliesByType(type: string): Promise<string[]> {
    // Get anomalies of same type
    return [];
  }

  private async identifyPrimaryCause(anomaly: Anomaly): Promise<string> {
    // Identify primary cause of anomaly
    const causeMap: Record<string, string> = {
      'volume_anomaly': 'Unusual trading volume pattern',
      'price_anomaly': 'Market price manipulation or natural volatility',
      'behavior_anomaly': 'Abnormal user or system behavior',
      'network_anomaly': 'Network congestion or infrastructure issue',
      'contract_anomaly': 'Smart contract vulnerability or exploit'
    };

    return causeMap[anomaly.type] || 'Unknown cause - requires investigation';
  }

  private async identifyContributingFactors(anomaly: Anomaly, correlatedAnomalies: string[]): Promise<string[]> {
    const factors: string[] = [];

    if (correlatedAnomalies.length > 0) {
      factors.push('Correlated with other security events');
    }

    if (anomaly.confidence < 0.7) {
      factors.push('Low confidence in detection');
    }

    if (anomaly.impact > 70) {
      factors.push('High potential impact');
    }

    return factors;
  }

  private async identifySystemicIssues(anomaly: Anomaly, correlatedAnomalies: string[]): Promise<string[]> {
    const issues: string[] = [];

    if (correlatedAnomalies.length > 3) {
      issues.push('Pattern suggests systemic vulnerability');
    }

    // Add more systemic issue detection logic

    return issues;
  }

  private generatePreventionMeasures(primaryCause: string, contributingFactors: string[]): string[] {
    const measures: string[] = [];

    if (primaryCause.includes('manipulation')) {
      measures.push('Implement additional price oracle validation');
      measures.push('Add circuit breakers for extreme price movements');
    }

    if (primaryCause.includes('volume')) {
      measures.push('Enhance volume anomaly detection algorithms');
      measures.push('Implement real-time volume monitoring');
    }

    if (contributingFactors.includes('Low confidence')) {
      measures.push('Improve detection model training');
      measures.push('Add additional data sources for verification');
    }

    return measures;
  }

  private async identifyDetectionGaps(anomaly: Anomaly): Promise<string[]> {
    const gaps: string[] = [];

    if (anomaly.confidence < 0.6) {
      gaps.push('Low detection confidence indicates model limitations');
    }

    // Add more detection gap analysis

    return gaps;
  }

  private scoresToSeverity(score: number): 'low' | 'medium' | 'high' | 'critical' {
    if (score >= 80) return 'critical';
    if (score >= 60) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  }

  private calculateCategoryConsistency(riskCategories: RiskCategory[]): number {
    if (riskCategories.length < 2) return 0.5;

    const scores = riskCategories.map(cat => cat.score);
    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);

    // Lower standard deviation = higher consistency
    return Math.max(0, 1 - (stdDev / 50));
  }

  private assessEventReliability(event: SecurityEvent): number {
    let reliability = 0.7; // Base reliability

    if (event.source === 'automated_detection') reliability += 0.1;
    if (event.metadata?.confidence) reliability = Math.max(reliability, event.metadata.confidence);

    return Math.min(reliability, 0.95);
  }

  private async generateCategoryMitigationStrategies(
    category: RiskCategory,
    event: SecurityEvent
  ): Promise<MitigationStrategy[]> {
    const strategies: MitigationStrategy[] = [];

    switch (category.category) {
      case 'technical':
        strategies.push({
          strategy: 'Implement additional monitoring for affected systems',
          priority: 'urgent',
          effectiveness: 75,
          implementationComplexity: 'medium',
          estimatedCost: 'Medium ($1K-10K)',
          timeToImplement: 4,
          prerequisites: ['System access', 'Monitoring tools'],
          expectedOutcome: 'Improved detection of similar issues'
        });
        break;

      case 'financial':
        strategies.push({
          strategy: 'Activate financial risk controls',
          priority: 'immediate',
          effectiveness: 85,
          implementationComplexity: 'low',
          estimatedCost: 'Low (<$1K)',
          timeToImplement: 1,
          prerequisites: ['Risk management system'],
          expectedOutcome: 'Reduced financial exposure'
        });
        break;

      // Add more category-specific strategies
    }

    return strategies;
  }

  private async generateGeneralMitigationStrategies(event: SecurityEvent): Promise<MitigationStrategy[]> {
    return [
      {
        strategy: 'Document incident for future analysis',
        priority: 'normal',
        effectiveness: 60,
        implementationComplexity: 'low',
        estimatedCost: 'Low (<$1K)',
        timeToImplement: 0.5,
        prerequisites: ['Documentation system'],
        expectedOutcome: 'Improved incident response knowledge base'
      }
    ];
  }

  private async assessFinancialImpact(event: SecurityEvent, riskCategories: RiskCategory[]): Promise<{
    estimatedLoss: BigNumber;
    affectedValue: BigNumber;
    recoveryLikelihood: number;
  }> {
    // Estimate financial impact based on event type and risk categories
    let estimatedLoss = BigNumber.from(0);
    let affectedValue = BigNumber.from(0);
    let recoveryLikelihood = 0.8;

    if (event.type.includes('fraud')) {
      estimatedLoss = ethers.utils.parseEther('10'); // $10 default
      affectedValue = ethers.utils.parseEther('100');
      recoveryLikelihood = 0.3;
    }

    return { estimatedLoss, affectedValue, recoveryLikelihood };
  }

  private async assessOperationalImpact(event: SecurityEvent, riskCategories: RiskCategory[]): Promise<{
    systemsAffected: string[];
    downtimeRisk: number;
    userImpact: number;
  }> {
    return {
      systemsAffected: event.affectedAddresses || [],
      downtimeRisk: event.severity === 'critical' ? 80 : 30,
      userImpact: 40
    };
  }

  private async assessReputationalImpact(event: SecurityEvent, riskCategories: RiskCategory[]): Promise<{
    severityLevel: number;
    stakeholdersAffected: string[];
    recoveryTime: number;
  }> {
    return {
      severityLevel: event.severity === 'critical' ? 80 : 40,
      stakeholdersAffected: ['users', 'partners'],
      recoveryTime: 168 // hours (1 week)
    };
  }

  private async assessLegalImpact(event: SecurityEvent, riskCategories: RiskCategory[]): Promise<{
    complianceViolations: string[];
    regulatoryRisk: number;
    liabilityExposure: number;
  }> {
    const violations: string[] = [];
    
    if (event.type.includes('breach')) {
      violations.push('Data protection regulation');
    }

    return {
      complianceViolations: violations,
      regulatoryRisk: violations.length > 0 ? 60 : 20,
      liabilityExposure: 30
    };
  }

  // Trend analysis methods
  private async analyzeTechnicalRiskTrend(eventType: string): Promise<'increasing' | 'stable' | 'decreasing'> {
    // Analyze historical trend for technical risks
    return 'stable';
  }

  private async analyzeOperationalRiskTrend(eventType: string): Promise<'increasing' | 'stable' | 'decreasing'> {
    return 'stable';
  }

  private async analyzeFinancialRiskTrend(eventType: string): Promise<'increasing' | 'stable' | 'decreasing'> {
    return 'stable';
  }

  private async analyzeReputationalRiskTrend(eventType: string): Promise<'increasing' | 'stable' | 'decreasing'> {
    return 'stable';
  }

  private async analyzeComplianceRiskTrend(eventType: string): Promise<'increasing' | 'stable' | 'decreasing'> {
    return 'stable';
  }

  // Threat context analysis methods
  private async identifyThreatActors(events: SecurityEvent[]): Promise<string[]> {
    const actors: string[] = [];
    
    events.forEach(event => {
      if (event.type.includes('fraud')) actors.push('Fraudsters');
      if (event.type.includes('hack')) actors.push('Hackers');
      if (event.type.includes('bot')) actors.push('Automated attackers');
    });

    return [...new Set(actors)];
  }

  private async identifyAttackVectors(events: SecurityEvent[]): Promise<string[]> {
    const vectors: string[] = [];
    
    events.forEach(event => {
      if (event.type.includes('phishing')) vectors.push('Social engineering');
      if (event.type.includes('contract')) vectors.push('Smart contract exploitation');
      if (event.type.includes('network')) vectors.push('Network attacks');
    });

    return [...new Set(vectors)];
  }

  private async analyzeThreatMotivations(events: SecurityEvent[]): Promise<string[]> {
    return ['Financial gain', 'Market manipulation', 'Disruption'];
  }

  private async assessThreatCapabilities(events: SecurityEvent[]): Promise<string[]> {
    return ['Technical expertise', 'Financial resources', 'Social engineering skills'];
  }

  private async identifyTargetedAssets(events: SecurityEvent[]): Promise<string[]> {
    const assets: string[] = [];
    
    events.forEach(event => {
      if (event.affectedAddresses) {
        assets.push(...event.affectedAddresses);
      }
    });

    return [...new Set(assets)];
  }

  private async analyzeGeopoliticalFactors(events: SecurityEvent[]): Promise<string[]> {
    // Analyze geopolitical factors affecting security
    return [];
  }

  private getDefaultRiskAssessment(): SecurityRiskAssessment {
    return {
      overallRiskScore: 50,
      riskLevel: 'medium',
      confidence: 0.5,
      riskCategories: [],
      mitigationStrategies: [],
      timeToAction: 3600000, // 1 hour
      impactAnalysis: this.getDefaultImpactAnalysis(),
      recommendations: ['Manual assessment required'],
      escalationRequired: false
    };
  }

  private getDefaultImpactAnalysis(): ImpactAnalysis {
    return {
      financialImpact: {
        estimatedLoss: BigNumber.from(0),
        affectedValue: BigNumber.from(0),
        recoveryLikelihood: 0.8
      },
      operationalImpact: {
        systemsAffected: [],
        downtimeRisk: 30,
        userImpact: 20
      },
      reputationalImpact: {
        severityLevel: 30,
        stakeholdersAffected: [],
        recoveryTime: 24
      },
      legalImpact: {
        complianceViolations: [],
        regulatoryRisk: 20,
        liabilityExposure: 10
      }
    };
  }

  private initializeRiskModel(): void {
    // Initialize risk model weights
    this.riskModelWeights.set('technical', 0.3);
    this.riskModelWeights.set('operational', 0.25);
    this.riskModelWeights.set('financial', 0.25);
    this.riskModelWeights.set('reputational', 0.1);
    this.riskModelWeights.set('compliance', 0.1);
  }

  // Public utility methods
  updateRiskModelWeights(weights: Map<string, number>): void {
    this.riskModelWeights = new Map(weights);
  }

  addHistoricalRiskData(type: string, data: any[]): void {
    this.historicalRiskData.set(type, data);
  }

  updateThreatIntelligence(threatData: Map<string, any>): void {
    this.threatIntelligence = new Map(threatData);
  }

  clearCache(): void {
    this.historicalRiskData.clear();
  }
}
