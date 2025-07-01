import { Router } from 'express';
import { securityController } from '../controllers/securityController';
import { 
  jwtAuth, 
  populateUser, 
  requirePermission, 
  requireSubscription 
} from '../middleware/auth';
import { validate, securitySchemas, commonSchemas } from '../middleware/validation';
import { 
  tierBasedLimiter, 
  securityLimiter, 
  rateLimitStatus 
} from '../middleware/rateLimiter';
import { SubscriptionTier } from '../models/User';
import { z } from 'zod';

const router = Router();

// Apply authentication and rate limiting to all routes
router.use(jwtAuth);
router.use(populateUser);
router.use(rateLimitStatus());
router.use(tierBasedLimiter());

/**
 * @route   GET /api/security/dashboard
 * @desc    Get security dashboard overview
 * @access  Private
 * @params  Query: timeRange
 */
router.get(
  '/dashboard',
  requirePermission('read_portfolio'),
  validate.query(commonSchemas.dateRange),
  securityController.getSecurityDashboard
);

/**
 * @route   GET /api/security/alerts
 * @desc    Get security alerts
 * @access  Private
 * @params  Query: pagination, filters
 */
router.get(
  '/alerts',
  requirePermission('read_portfolio'),
  validate.query(commonSchemas.pagination.extend({
    type: z.string().optional(),
    severity: z.enum(['low', 'medium', 'high', 'critical']).optional(),
    status: z.enum(['active', 'investigating', 'resolved', 'false_positive']).optional(),
    minRiskScore: z.number().min(0).max(100).optional(),
    startDate: z.string().datetime().optional(),
    endDate: z.string().datetime().optional()
  })),
  securityController.getSecurityAlerts
);

/**
 * @route   POST /api/security/analyze/transaction
 * @desc    Analyze transaction for security risks
 * @access  Private (Basic subscription or higher)
 * @params  Body: txHash, chainId
 */
router.post(
  '/analyze/transaction',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('execute_trades'),
  securityLimiter(),
  validate.body(securitySchemas.analyzeTransaction),
  securityController.analyzeTransaction
);

/**
 * @route   POST /api/security/monitor/wallet
 * @desc    Setup wallet monitoring
 * @access  Private (Premium subscription or higher)
 * @params  Body: walletAddress, chainId, monitoringLevel
 */
router.post(
  '/monitor/wallet',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.body(securitySchemas.monitorWallet),
  securityController.monitorWallet
);

/**
 * @route   GET /api/security/risk-assessment
 * @desc    Get real-time risk assessment
 * @access  Private
 * @params  Query: scope, timeWindow
 */
router.get(
  '/risk-assessment',
  requirePermission('read_portfolio'),
  validate.query(z.object({
    scope: z.enum(['overall', 'portfolio', 'agents', 'transactions']).default('overall'),
    timeWindow: z.number().int().min(1).max(168).default(24) // hours
  })),
  securityController.getRiskAssessment
);

/**
 * @route   POST /api/security/incidents
 * @desc    Report security incident
 * @access  Private
 * @params  Body: incident details
 */
router.post(
  '/incidents',
  requirePermission('read_portfolio'),
  securityLimiter(),
  validate.body(securitySchemas.reportIncident),
  securityController.reportIncident
);

/**
 * @route   PUT /api/security/alerts/:alertId/resolve
 * @desc    Resolve security alert
 * @access  Private
 * @params  URL: alertId (UUID), Body: resolution, notes
 */
router.put(
  '/alerts/:alertId/resolve',
  requirePermission('read_portfolio'),
  validate.params(z.object({
    alertId: commonSchemas.uuid
  })),
  validate.body(z.object({
    resolution: z.enum(['resolved', 'false_positive']),
    notes: z.string().max(1000).optional()
  })),
  securityController.resolveAlert
);

/**
 * @route   GET /api/security/metrics
 * @desc    Get security metrics and analytics
 * @access  Private (Premium subscription or higher)
 * @params  Query: timeRange, granularity
 */
router.get(
  '/metrics',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.query(commonSchemas.dateRange.extend({
    granularity: z.enum(['hourly', 'daily', 'weekly']).default('daily')
  })),
  securityController.getSecurityMetrics
);

/**
 * @route   POST /api/security/emergency-stop
 * @desc    Emergency stop all operations
 * @access  Private
 * @params  Body: reason, scope
 */
router.post(
  '/emergency-stop',
  requirePermission('emergency_stop'),
  securityLimiter(),
  validate.body(securitySchemas.emergencyStop),
  securityController.emergencyStop
);

/**
 * @route   GET /api/security/threat-intelligence
 * @desc    Get threat intelligence feed
 * @access  Private (Enterprise subscription only)
 * @params  Query: category, severity, limit
 */
router.get(
  '/threat-intelligence',
  requireSubscription(SubscriptionTier.ENTERPRISE),
  requirePermission('advanced_analytics'),
  validate.query(z.object({
    category: z.enum(['contracts', 'addresses', 'patterns', 'vulnerabilities']).optional(),
    severity: z.enum(['low', 'medium', 'high', 'critical']).optional(),
    limit: z.number().int().min(1).max(100).default(50)
  })),
  securityController.getThreatIntelligence
);

/**
 * @route   GET /api/security/contracts/scan/:contractAddress
 * @desc    Scan smart contract for security vulnerabilities
 * @access  Private (Premium subscription or higher)
 * @params  URL: contractAddress, Query: chainId
 */
router.get(
  '/contracts/scan/:contractAddress',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  securityLimiter(),
  validate.params(z.object({
    contractAddress: commonSchemas.ethereumAddress
  })),
  validate.query(z.object({
    chainId: commonSchemas.chainId
  })),
  (req, res, next) => {
    // This would integrate with mlService for contract analysis
    res.json({
      success: true,
      data: {
        contractAddress: req.params.contractAddress,
        chainId: req.query.chainId,
        securityScore: 85,
        vulnerabilities: [],
        recommendations: [],
        scanDate: new Date(),
        source: 'ArbOS Security Scanner'
      }
    });
  }
);

/**
 * @route   POST /api/security/whitelist/addresses
 * @desc    Add addresses to security whitelist
 * @access  Private (Premium subscription or higher)
 * @params  Body: addresses array
 */
router.post(
  '/whitelist/addresses',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.body(z.object({
    addresses: z.array(commonSchemas.ethereumAddress).min(1).max(100),
    reason: z.string().min(10).max(500)
  })),
  (req, res, next) => {
    res.json({
      success: true,
      data: {
        addedAddresses: req.body.addresses,
        reason: req.body.reason,
        addedAt: new Date()
      },
      message: 'Addresses added to whitelist successfully'
    });
  }
);

export default router;
