import { Request, Response, NextFunction } from 'express';
import { securityService } from '../services/securityService';
import { chainlinkService } from '../services/chainlinkService';
import { logger } from '../utils/logger';
import { validateRequest } from '../utils/validators';
import { ApiResponse, PaginationParams } from '../types/api';

interface SecurityAlert {
  id: string;
  userId: string;
  type: SecurityAlertType;
  severity: SecuritySeverity;
  title: string;
  description: string;
  riskScore: number;
  status: SecurityStatus;
  metadata: any;
  createdAt: Date;
  resolvedAt?: Date;
}

enum SecurityAlertType {
  SUSPICIOUS_TRANSACTION = 'suspicious_transaction',
  UNUSUAL_ACTIVITY = 'unusual_activity',
  HIGH_RISK_CONTRACT = 'high_risk_contract',
  PRICE_MANIPULATION = 'price_manipulation',
  FRONT_RUNNING = 'front_running',
  MEV_ATTACK = 'mev_attack',
  PHISHING_ATTEMPT = 'phishing_attempt',
  WALLET_COMPROMISE = 'wallet_compromise',
  SMART_CONTRACT_VULNERABILITY = 'smart_contract_vulnerability'
}

enum SecuritySeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

enum SecurityStatus {
  ACTIVE = 'active',
  INVESTIGATING = 'investigating',
  RESOLVED = 'resolved',
  FALSE_POSITIVE = 'false_positive'
}

class SecurityController {
  /**
   * Get security dashboard overview
   */
  public async getSecurityDashboard(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const timeRange = {
        start: req.query.start ? new Date(req.query.start as string) : new Date(Date.now() - 24 * 60 * 60 * 1000), // 24 hours
        end: req.query.end ? new Date(req.query.end as string) : new Date()
      };

      const dashboard = await securityService.getSecurityDashboard(userId, timeRange);

      // Enrich with real-time risk assessment
      const riskAssessment = await securityService.calculateCurrentRiskScore(userId);
      const chainlinkHealthStatus = await chainlinkService.getHealthStatus();

      const enrichedDashboard = {
        ...dashboard,
        currentRiskScore: riskAssessment.score,
        riskFactors: riskAssessment.factors,
        chainlinkStatus: chainlinkHealthStatus,
        recommendations: await securityService.getSecurityRecommendations(userId, riskAssessment.score)
      };

      const response: ApiResponse<typeof enrichedDashboard> = {
        success: true,
        data: enrichedDashboard
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting security dashboard:', error);
      next(error);
    }
  }

  /**
   * Get security alerts
   */
  public async getSecurityAlerts(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const pagination: PaginationParams = {
        page: parseInt(req.query.page as string) || 1,
        limit: parseInt(req.query.limit as string) || 20,
        sortBy: req.query.sortBy as string || 'createdAt',
        sortOrder: (req.query.sortOrder as 'asc' | 'desc') || 'desc'
      };

      const filters = {
        type: req.query.type as SecurityAlertType,
        severity: req.query.severity as SecuritySeverity,
        status: req.query.status as SecurityStatus,
        minRiskScore: req.query.minRiskScore ? parseInt(req.query.minRiskScore as string) : undefined,
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined
      };

      const result = await securityService.getSecurityAlerts(userId, filters, pagination);

      const response: ApiResponse<typeof result.alerts> = {
        success: true,
        data: result.alerts,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: result.total,
          totalPages: Math.ceil(result.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting security alerts:', error);
      next(error);
    }
  }

  /**
   * Analyze transaction for security risks
   */
  public async analyzeTransaction(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const { txHash, chainId } = req.body;

      // Validate input
      const validation = validateRequest(req.body, {
        txHash: { required: true, type: 'string', pattern: /^0x[a-fA-F0-9]{64}$/ },
        chainId: { required: true, type: 'number' }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid transaction data',
          details: validation.errors
        });
        return;
      }

      // Perform comprehensive security analysis
      const analysis = await securityService.analyzeTransaction({
        txHash,
        chainId,
        userId
      });

      // Get additional context from Chainlink
      const priceContext = await chainlinkService.getTransactionPriceContext(txHash, chainId);
      
      const enrichedAnalysis = {
        ...analysis,
        priceContext,
        recommendations: await securityService.generateRecommendations(analysis.riskScore, analysis.riskFactors)
      };

      logger.info(`Transaction analyzed`, {
        userId,
        txHash,
        chainId,
        riskScore: analysis.riskScore,
        alertsGenerated: analysis.alerts.length
      });

      const response: ApiResponse<typeof enrichedAnalysis> = {
        success: true,
        data: enrichedAnalysis
      };

      res.json(response);
    } catch (error) {
      logger.error('Error analyzing transaction:', error);
      next(error);
    }
  }

  /**
   * Monitor wallet for suspicious activity
   */
  public async monitorWallet(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const { walletAddress, chainId, monitoringLevel } = req.body;

      // Validate input
      const validation = validateRequest(req.body, {
        walletAddress: { required: true, type: 'string', pattern: /^0x[a-fA-F0-9]{40}$/ },
        chainId: { required: true, type: 'number' },
        monitoringLevel: { required: false, type: 'string', enum: ['basic', 'standard', 'enhanced'] }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid wallet monitoring request',
          details: validation.errors
        });
        return;
      }

      // Setup wallet monitoring using Chainlink Automation
      const monitoringSetup = await securityService.setupWalletMonitoring({
        userId,
        walletAddress,
        chainId,
        monitoringLevel: monitoringLevel || 'standard'
      });

      logger.info(`Wallet monitoring setup`, {
        userId,
        walletAddress,
        chainId,
        monitoringId: monitoringSetup.monitoringId
      });

      const response: ApiResponse<typeof monitoringSetup> = {
        success: true,
        data: monitoringSetup,
        message: 'Wallet monitoring setup successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error setting up wallet monitoring:', error);
      next(error);
    }
  }

  /**
   * Get real-time risk assessment
   */
  public async getRiskAssessment(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const scope = req.query.scope as string || 'overall'; // overall, portfolio, agents, transactions
      const timeWindow = parseInt(req.query.timeWindow as string) || 24; // hours

      const riskAssessment = await securityService.calculateRiskAssessment({
        userId,
        scope,
        timeWindow
      });

      // Get market risk context from Chainlink
      const marketRisk = await chainlinkService.getMarketRiskIndicators();

      const enrichedAssessment = {
        ...riskAssessment,
        marketContext: marketRisk,
        recommendations: await securityService.generateRiskMitigationStrategies(riskAssessment)
      };

      const response: ApiResponse<typeof enrichedAssessment> = {
        success: true,
        data: enrichedAssessment
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting risk assessment:', error);
      next(error);
    }
  }

  /**
   * Report security incident
   */
  public async reportIncident(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const { 
        type, 
        severity, 
        description, 
        txHash, 
        chainId, 
        metadata 
      } = req.body;

      // Validate input
      const validation = validateRequest(req.body, {
        type: { required: true, type: 'string', enum: Object.values(SecurityAlertType) },
        severity: { required: true, type: 'string', enum: Object.values(SecuritySeverity) },
        description: { required: true, type: 'string', minLength: 10, maxLength: 1000 },
        txHash: { required: false, type: 'string', pattern: /^0x[a-fA-F0-9]{64}$/ },
        chainId: { required: false, type: 'number' }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid incident report',
          details: validation.errors
        });
        return;
      }

      const incident = await securityService.createSecurityIncident({
        userId,
        type,
        severity,
        description,
        txHash,
        chainId,
        metadata: metadata || {}
      });

      // Auto-trigger investigation for high severity incidents
      if (severity === SecuritySeverity.HIGH || severity === SecuritySeverity.CRITICAL) {
        await securityService.initiateIncidentInvestigation(incident.id);
        
        // Trigger emergency protocols if critical
        if (severity === SecuritySeverity.CRITICAL) {
          await securityService.triggerEmergencyProtocols(userId, incident.id);
        }
      }

      logger.info(`Security incident reported`, {
        userId,
        incidentId: incident.id,
        type,
        severity,
        txHash
      });

      const response: ApiResponse<typeof incident> = {
        success: true,
        data: incident,
        message: 'Security incident reported successfully'
      };

      res.status(201).json(response);
    } catch (error) {
      logger.error('Error reporting security incident:', error);
      next(error);
    }
  }

  /**
   * Resolve security alert
   */
  public async resolveAlert(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const alertId = req.params.alertId;
      const { resolution, notes } = req.body;

      if (!alertId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Alert ID is required'
        });
        return;
      }

      const validResolutions = [SecurityStatus.RESOLVED, SecurityStatus.FALSE_POSITIVE];
      if (!validResolutions.includes(resolution)) {
        res.status(400).json({
          success: false,
          error: 'Invalid Resolution',
          message: 'Resolution must be either resolved or false_positive'
        });
        return;
      }

      const resolvedAlert = await securityService.resolveSecurityAlert({
        alertId,
        userId,
        resolution,
        notes
      });

      logger.info(`Security alert resolved`, {
        userId,
        alertId,
        resolution,
        hasNotes: !!notes
      });

      const response: ApiResponse<typeof resolvedAlert> = {
        success: true,
        data: resolvedAlert,
        message: 'Security alert resolved successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error resolving security alert:', error);
      next(error);
    }
  }

  /**
   * Get security metrics
   */
  public async getSecurityMetrics(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const timeRange = {
        start: req.query.start ? new Date(req.query.start as string) : new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days
        end: req.query.end ? new Date(req.query.end as string) : new Date()
      };

      const granularity = req.query.granularity as string || 'daily'; // hourly, daily, weekly

      const metrics = await securityService.getSecurityMetrics(userId, timeRange, granularity);

      const response: ApiResponse<typeof metrics> = {
        success: true,
        data: metrics
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting security metrics:', error);
      next(error);
    }
  }

  /**
   * Emergency stop all operations
   */
  public async emergencyStop(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const { reason, scope } = req.body;

      // Validate input
      const validation = validateRequest(req.body, {
        reason: { required: true, type: 'string', minLength: 10, maxLength: 500 },
        scope: { required: false, type: 'string', enum: ['all', 'agents', 'portfolios', 'arbitrage'] }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid emergency stop request',
          details: validation.errors
        });
        return;
      }

      const emergencyResult = await securityService.triggerEmergencyStop({
        userId,
        reason,
        scope: scope || 'all'
      });

      logger.critical(`Emergency stop triggered`, {
        userId,
        reason,
        scope,
        affectedOperations: emergencyResult.stoppedOperations.length
      });

      const response: ApiResponse<typeof emergencyResult> = {
        success: true,
        data: emergencyResult,
        message: 'Emergency stop executed successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error executing emergency stop:', error);
      next(error);
    }
  }

  /**
   * Get threat intelligence feed
   */
  public async getThreatIntelligence(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const category = req.query.category as string; // contracts, addresses, patterns, vulnerabilities
      const severity = req.query.severity as SecuritySeverity;
      const limit = parseInt(req.query.limit as string) || 50;

      const threatIntel = await securityService.getThreatIntelligence({
        userId,
        category,
        severity,
        limit
      });

      const response: ApiResponse<typeof threatIntel> = {
        success: true,
        data: threatIntel
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting threat intelligence:', error);
      next(error);
    }
  }
}

export const securityController = new SecurityController();
export default securityController;
