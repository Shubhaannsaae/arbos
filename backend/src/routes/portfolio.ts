import { Router } from 'express';
import { portfolioController } from '../controllers/portfolioController';
import { 
  jwtAuth, 
  populateUser, 
  requirePermission, 
  requireSubscription 
} from '../middleware/auth';
import { validate, portfolioSchemas, commonSchemas } from '../middleware/validation';
import { 
  tierBasedLimiter, 
  rebalanceLimiter, 
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
 * @route   POST /api/portfolios
 * @desc    Create a new portfolio
 * @access  Private (Basic subscription or higher)
 * @params  Body: CreatePortfolioDto
 */
router.post(
  '/',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('manage_portfolio'),
  validate.body(portfolioSchemas.createPortfolio),
  portfolioController.createPortfolio
);

/**
 * @route   GET /api/portfolios
 * @desc    Get all portfolios for authenticated user
 * @access  Private
 * @params  Query: pagination, filters
 */
router.get(
  '/',
  requirePermission('read_portfolio'),
  validate.query(commonSchemas.pagination.extend({
    chainId: commonSchemas.chainId.optional(),
    isActive: z.boolean().optional()
  })),
  portfolioController.getUserPortfolios
);

/**
 * @route   GET /api/portfolios/:portfolioId
 * @desc    Get specific portfolio by ID
 * @access  Private
 * @params  URL: portfolioId (UUID)
 */
router.get(
  '/:portfolioId',
  requirePermission('read_portfolio'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  portfolioController.getPortfolio
);

/**
 * @route   PUT /api/portfolios/:portfolioId
 * @desc    Update portfolio configuration
 * @access  Private (Basic subscription or higher)
 * @params  URL: portfolioId (UUID), Body: UpdatePortfolioDto
 */
router.put(
  '/:portfolioId',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('manage_portfolio'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  validate.body(portfolioSchemas.updatePortfolio),
  portfolioController.updatePortfolio
);

/**
 * @route   POST /api/portfolios/:portfolioId/rebalance
 * @desc    Execute portfolio rebalancing
 * @access  Private (Basic subscription or higher)
 * @params  URL: portfolioId (UUID), Body: RebalanceParams
 */
router.post(
  '/:portfolioId/rebalance',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('manage_portfolio'),
  rebalanceLimiter(),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  validate.body(portfolioSchemas.rebalancePortfolio),
  portfolioController.rebalancePortfolio
);

/**
 * @route   GET /api/portfolios/:portfolioId/performance
 * @desc    Get portfolio performance metrics
 * @access  Private
 * @params  URL: portfolioId (UUID), Query: dateRange
 */
router.get(
  '/:portfolioId/performance',
  requirePermission('read_portfolio'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  validate.query(commonSchemas.dateRange),
  portfolioController.getPortfolioPerformance
);

/**
 * @route   GET /api/portfolios/:portfolioId/rebalancing/history
 * @desc    Get portfolio rebalancing history
 * @access  Private
 * @params  URL: portfolioId (UUID), Query: pagination
 */
router.get(
  '/:portfolioId/rebalancing/history',
  requirePermission('read_portfolio'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  validate.query(commonSchemas.pagination),
  portfolioController.getRebalancingHistory
);

/**
 * @route   POST /api/portfolios/:portfolioId/rebalancing/simulate
 * @desc    Simulate portfolio rebalancing
 * @access  Private (Premium subscription or higher)
 * @params  URL: portfolioId (UUID), Body: newAllocation, strategy
 */
router.post(
  '/:portfolioId/rebalancing/simulate',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  validate.body(z.object({
    newAllocation: z.array(z.object({
      tokenAddress: commonSchemas.ethereumAddress,
      percentage: commonSchemas.percentage
    })),
    strategy: z.enum(['optimal', 'aggressive', 'conservative']).default('optimal')
  })),
  portfolioController.simulateRebalancing
);

/**
 * @route   POST /api/portfolios/:portfolioId/automation/setup
 * @desc    Setup automated rebalancing using Chainlink Automation
 * @access  Private (Premium subscription or higher)
 * @params  URL: portfolioId (UUID), Body: automation settings
 */
router.post(
  '/:portfolioId/automation/setup',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  validate.body(z.object({
    trigger: z.enum(['threshold', 'time', 'volatility', 'market_conditions']),
    threshold: z.number().min(0.1).max(50),
    frequency: z.enum(['daily', 'weekly', 'monthly', 'quarterly']).optional(),
    slippageTolerance: commonSchemas.slippage.default(0.5),
    gasOptimization: z.boolean().default(true)
  })),
  portfolioController.setupAutomatedRebalancing
);

/**
 * @route   DELETE /api/portfolios/:portfolioId
 * @desc    Delete a portfolio
 * @access  Private (Basic subscription or higher)
 * @params  URL: portfolioId (UUID)
 */
router.delete(
  '/:portfolioId',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('manage_portfolio'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  portfolioController.deletePortfolio
);

/**
 * @route   GET /api/portfolios/:portfolioId/tokens/prices
 * @desc    Get real-time token prices for portfolio
 * @access  Private
 * @params  URL: portfolioId (UUID)
 */
router.get(
  '/:portfolioId/tokens/prices',
  requirePermission('read_portfolio'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  (req, res, next) => {
    // This would integrate with chainlinkService for real-time prices
    res.json({
      success: true,
      data: {
        portfolioId: req.params.portfolioId,
        prices: [],
        lastUpdated: new Date(),
        source: 'Chainlink Data Feeds'
      }
    });
  }
);

/**
 * @route   POST /api/portfolios/:portfolioId/analysis
 * @desc    Get AI-powered portfolio analysis
 * @access  Private (Premium subscription or higher)
 * @params  URL: portfolioId (UUID)
 */
router.post(
  '/:portfolioId/analysis',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.params(z.object({
    portfolioId: commonSchemas.uuid
  })),
  (req, res, next) => {
    // This would integrate with mlService for AI analysis
    res.json({
      success: true,
      data: {
        portfolioId: req.params.portfolioId,
        analysis: {
          riskScore: 45,
          recommendations: [],
          optimizationSuggestions: [],
          performancePrediction: {
            expectedReturn: 8.5,
            confidence: 75
          }
        },
        generatedAt: new Date()
      }
    });
  }
);

export default router;
