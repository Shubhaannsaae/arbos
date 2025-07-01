import { Router } from 'express';
import { agentController } from '../controllers/agentController';
import { 
  jwtAuth, 
  populateUser, 
  requirePermission, 
  requireSubscription 
} from '../middleware/auth';
import { validate, agentSchemas, commonSchemas } from '../middleware/validation';
import { 
  tierBasedLimiter, 
  agentOperationLimiter, 
  rateLimitStatus 
} from '../middleware/rateLimiter';
import { SubscriptionTier } from '../models/User';

const router = Router();

// Apply authentication and rate limiting to all routes
router.use(jwtAuth);
router.use(populateUser);
router.use(rateLimitStatus());
router.use(tierBasedLimiter());

/**
 * @route   POST /api/agents
 * @desc    Create a new AI agent
 * @access  Private (Basic subscription or higher)
 * @params  Body: CreateAgentDto
 */
router.post(
  '/',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission(['execute_trades', 'manage_portfolio']),
  agentOperationLimiter(),
  validate.body(agentSchemas.createAgent),
  agentController.createAgent
);

/**
 * @route   GET /api/agents
 * @desc    Get all agents for authenticated user
 * @access  Private
 * @params  Query: pagination, filters
 */
router.get(
  '/',
  requirePermission('read_agents'),
  validate.query(commonSchemas.pagination.extend({
    type: agentSchemas.createAgent.shape.type.optional(),
    status: agentSchemas.createAgent.shape.type.optional(),
    enabled: commonSchemas.uuid.optional()
  })),
  agentController.getUserAgents
);

/**
 * @route   GET /api/agents/:agentId
 * @desc    Get specific agent by ID
 * @access  Private
 * @params  URL: agentId (UUID)
 */
router.get(
  '/:agentId',
  requirePermission('read_agents'),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  agentController.getAgent
);

/**
 * @route   PUT /api/agents/:agentId
 * @desc    Update agent configuration
 * @access  Private (Basic subscription or higher)
 * @params  URL: agentId (UUID), Body: UpdateAgentDto
 */
router.put(
  '/:agentId',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission(['execute_trades', 'manage_portfolio']),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  validate.body(agentSchemas.updateAgent),
  agentController.updateAgent
);

/**
 * @route   POST /api/agents/:agentId/start
 * @desc    Start an agent
 * @access  Private (Basic subscription or higher)
 * @params  URL: agentId (UUID)
 */
router.post(
  '/:agentId/start',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission(['execute_trades', 'manage_portfolio']),
  agentOperationLimiter(),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  agentController.startAgent
);

/**
 * @route   POST /api/agents/:agentId/stop
 * @desc    Stop an agent
 * @access  Private (Basic subscription or higher)
 * @params  URL: agentId (UUID)
 */
router.post(
  '/:agentId/stop',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission(['execute_trades', 'manage_portfolio']),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  agentController.stopAgent
);

/**
 * @route   GET /api/agents/:agentId/performance
 * @desc    Get agent performance metrics
 * @access  Private
 * @params  URL: agentId (UUID), Query: dateRange
 */
router.get(
  '/:agentId/performance',
  requirePermission('read_agents'),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  validate.query(commonSchemas.dateRange),
  agentController.getAgentPerformance
);

/**
 * @route   GET /api/agents/:agentId/logs
 * @desc    Get agent execution logs
 * @access  Private (Premium subscription or higher)
 * @params  URL: agentId (UUID), Query: pagination, filters
 */
router.get(
  '/:agentId/logs',
  requireSubscription(SubscriptionTier.PREMIUM),
  requirePermission('advanced_analytics'),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  validate.query(commonSchemas.pagination.extend({
    level: commonSchemas.uuid.optional(),
    startDate: commonSchemas.uuid.optional(),
    endDate: commonSchemas.uuid.optional()
  })),
  agentController.getAgentLogs
);

/**
 * @route   POST /api/agents/:agentId/execute
 * @desc    Execute agent action manually
 * @access  Private (Basic subscription or higher)
 * @params  URL: agentId (UUID), Body: action, parameters
 */
router.post(
  '/:agentId/execute',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission(['execute_trades', 'manage_portfolio']),
  agentOperationLimiter(),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  validate.body(agentSchemas.executeAction),
  agentController.executeAgentAction
);

/**
 * @route   DELETE /api/agents/:agentId
 * @desc    Delete an agent
 * @access  Private (Basic subscription or higher)
 * @params  URL: agentId (UUID)
 */
router.delete(
  '/:agentId',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission(['execute_trades', 'manage_portfolio']),
  validate.params(commonSchemas.uuid.shape({ agentId: commonSchemas.uuid })),
  agentController.deleteAgent
);

export default router;
