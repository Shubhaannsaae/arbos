import { Router } from 'express';
import { arbitrageController } from '../controllers/arbitrageController';
import { 
  jwtAuth, 
  populateUser, 
  requirePermission, 
  requireSubscription 
} from '../middleware/auth';
import { validate, arbitrageSchemas, commonSchemas } from '../middleware/validation';
import { 
  tierBasedLimiter, 
  arbitrageLimiter, 
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
 * @route   GET /api/arbitrage/opportunities
 * @desc    Get available arbitrage opportunities
 * @access  Private
 * @params  Query: filters, pagination
 */
router.get(
  '/opportunities',
  requirePermission('read_arbitrage'),
  validate.query(commonSchemas.pagination.extend({
    minProfit: z.number().positive().optional(),
    maxRiskScore: z.number().min(0).max(100).optional(),
    chains: z.string().optional(), // comma-separated chain IDs
    exchanges: z.string().optional(), // comma-separated exchange names
    tokens: z.string().optional(), // comma-separated token addresses
    status: z.string().optional(),
    startDate: z.string().datetime().optional(),
    endDate: z.string().datetime().optional()
  })),
  arbitrageController.getOpportunities
);

/**
 * @route   POST /api/arbitrage/execute
 * @desc    Execute an arbitrage opportunity
 * @access  Private (Basic subscription or higher)
 * @params  Body: ExecuteArbitrageDto
 */
router.post(
  '/execute',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('execute_trades'),
  arbitrageLimiter(),
  validate.body(arbitrageSchemas.executeArbitrage),
  arbitrageController.executeArbitrage
);

/**
 * @route   GET /api/arbitrage/history
 * @desc    Get arbitrage execution history
 * @access  Private
 * @params  Query: pagination, filters
 */
router.get(
  '/history',
  requirePermission('read_arbitrage'),
  validate.query(commonSchemas.pagination.extend({
    status: z.string().optional(),
    startDate: z.string().datetime().optional(),
    endDate: z.string().datetime().optional(),
    minProfit: z.number().positive().optional(),
    tokenPair: z.string().optional()
  })),
  arbitrageController.getExecutionHistory
);

/**
 * @route   GET /api/arbitrage/analytics
 * @desc    Get arbitrage analytics and performance metrics
 * @access  Private (Premium subscription or higher)
 * @params  Query: dateRange
 */
router.get(
  '/analytics',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.query(commonSchemas.dateRange),
  arbitrageController.getAnalytics
);

/**
 * @route   GET /api/arbitrage/market-data
 * @desc    Get real-time market data for arbitrage
 * @access  Private
 * @params  Query: pairs, chains
 */
router.get(
  '/market-data',
  requirePermission('read_arbitrage'),
  validate.query(z.object({
    pairs: z.string().optional(), // comma-separated trading pairs
    chains: z.string().optional() // comma-separated chain IDs
  })),
  arbitrageController.getMarketData
);

/**
 * @route   POST /api/arbitrage/opportunities
 * @desc    Submit manual arbitrage opportunity
 * @access  Private (Premium subscription or higher)
 * @params  Body: CreateArbitrageOpportunityDto
 */
router.post(
  '/opportunities',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission(['execute_trades', 'advanced_analytics']),
  validate.body(arbitrageSchemas.createOpportunity),
  arbitrageController.submitOpportunity
);

/**
 * @route   DELETE /api/arbitrage/opportunities/:opportunityId
 * @desc    Cancel arbitrage opportunity
 * @access  Private (Basic subscription or higher)
 * @params  URL: opportunityId (UUID)
 */
router.delete(
  '/opportunities/:opportunityId',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('execute_trades'),
  validate.params(z.object({
    opportunityId: commonSchemas.uuid
  })),
  arbitrageController.cancelOpportunity
);

/**
 * @route   GET /api/arbitrage/cross-chain/status/:messageId
 * @desc    Get cross-chain arbitrage transaction status
 * @access  Private
 * @params  URL: messageId (string)
 */
router.get(
  '/cross-chain/status/:messageId',
  requirePermission('read_arbitrage'),
  validate.params(z.object({
    messageId: z.string().min(1)
  })),
  (req, res, next) => {
    // This would integrate with crossChainService
    res.json({
      success: true,
      data: {
        messageId: req.params.messageId,
        status: 'pending',
        estimatedCompletion: new Date(Date.now() + 300000) // 5 minutes
      }
    });
  }
);

/**
 * @route   GET /api/arbitrage/fees/estimate
 * @desc    Estimate arbitrage execution fees
 * @access  Private
 * @params  Query: sourceChain, targetChain, amount
 */
router.get(
  '/fees/estimate',
  requirePermission('read_arbitrage'),
  validate.query(z.object({
    sourceChain: commonSchemas.chainId,
    targetChain: commonSchemas.chainId,
    amount: commonSchemas.amount
  })),
  (req, res, next) => {
    // This would integrate with crossChainService for fee estimation
    const { sourceChain, targetChain, amount } = req.query;
    
    res.json({
      success: true,
      data: {
        sourceChain,
        targetChain,
        amount,
        estimatedFees: {
          sourceTx: 0.01, // ETH
          bridgeFee: 0.005, // ETH
          targetTx: 0.008, // ETH
          totalFeeETH: 0.023,
          totalFeeUSD: 46.00
        },
        estimatedTime: '5-10 minutes'
      }
    });
  }
);

export default router;
