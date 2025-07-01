import { Router } from 'express';
import { userController } from '../controllers/userController';
import { 
  jwtAuth, 
  populateUser, 
  optionalJwtAuth,
  requirePermission, 
  requireSubscription,
  verifyWalletOwnership
} from '../middleware/auth';
import { validate, userSchemas, commonSchemas } from '../middleware/validation';
import { 
  tierBasedLimiter, 
  loginLimiter, 
  rateLimitStatus 
} from '../middleware/rateLimiter';
import { SubscriptionTier } from '../models/User';
import { z } from 'zod';

const router = Router();

/**
 * @route   POST /api/users/register
 * @desc    Register a new user
 * @access  Public
 * @params  Body: CreateUserDto
 */
router.post(
  '/register',
  loginLimiter(),
  validate.body(userSchemas.createUser),
  userController.createUser
);

/**
 * @route   POST /api/users/login
 * @desc    User login with wallet signature
 * @access  Public
 * @params  Body: walletAddress, signature, message
 */
router.post(
  '/login',
  loginLimiter(),
  validate.body(z.object({
    walletAddress: commonSchemas.ethereumAddress,
    signature: z.string().min(1),
    message: z.string().min(1)
  })),
  (req, res, next) => {
    // This would implement wallet signature verification and JWT generation
    const { walletAddress, signature, message } = req.body;
    
    // Mock response - in production, this would verify signature and generate JWT
    res.json({
      success: true,
      data: {
        accessToken: 'mock_jwt_token',
        refreshToken: 'mock_refresh_token',
        user: {
          id: 'mock_user_id',
          walletAddress,
          subscriptionTier: SubscriptionTier.FREE
        }
      },
      message: 'Login successful'
    });
  }
);

/**
 * @route   POST /api/users/refresh
 * @desc    Refresh JWT token
 * @access  Public
 * @params  Body: refreshToken
 */
router.post(
  '/refresh',
  validate.body(z.object({
    refreshToken: z.string().min(1)
  })),
  (req, res, next) => {
    // This would implement refresh token validation and new JWT generation
    res.json({
      success: true,
      data: {
        accessToken: 'new_mock_jwt_token',
        refreshToken: 'new_mock_refresh_token'
      },
      message: 'Token refreshed successfully'
    });
  }
);

// Apply authentication and rate limiting to protected routes
router.use(jwtAuth);
router.use(populateUser);
router.use(rateLimitStatus());
router.use(tierBasedLimiter());

/**
 * @route   GET /api/users/profile
 * @desc    Get user profile
 * @access  Private
 */
router.get(
  '/profile',
  requirePermission('read_profile'),
  userController.getUserProfile
);

/**
 * @route   PUT /api/users/profile
 * @desc    Update user profile
 * @access  Private
 * @params  Body: UpdateUserDto
 */
router.put(
  '/profile',
  requirePermission('manage_profile'),
  validate.body(userSchemas.updateUser),
  userController.updateUserProfile
);

/**
 * @route   GET /api/users/activity
 * @desc    Get user activity history
 * @access  Private
 * @params  Query: pagination, filters
 */
router.get(
  '/activity',
  requirePermission('read_profile'),
  validate.query(commonSchemas.pagination.extend({
    type: z.string().optional(),
    startDate: z.string().datetime().optional(),
    endDate: z.string().datetime().optional()
  })),
  userController.getUserActivity
);

/**
 * @route   GET /api/users/dashboard
 * @desc    Get user dashboard summary
 * @access  Private
 * @params  Query: timeRange
 */
router.get(
  '/dashboard',
  requirePermission('read_profile'),
  validate.query(commonSchemas.dateRange),
  userController.getUserDashboard
);

/**
 * @route   PUT /api/users/subscription
 * @desc    Update user subscription
 * @access  Private
 * @params  Body: tier, paymentMethod
 */
router.put(
  '/subscription',
  requirePermission('manage_profile'),
  validate.body(z.object({
    tier: commonSchemas.subscriptionTier,
    paymentMethod: z.object({
      type: z.enum(['crypto', 'card', 'bank']),
      details: z.record(z.any())
    }).optional()
  })),
  userController.updateSubscription
);

/**
 * @route   GET /api/users/api-usage
 * @desc    Get API usage statistics
 * @access  Private
 * @params  Query: timeRange
 */
router.get(
  '/api-usage',
  requirePermission('read_profile'),
  validate.query(commonSchemas.dateRange),
  userController.getApiUsage
);

/**
 * @route   POST /api/users/api-keys
 * @desc    Generate new API key
 * @access  Private (Basic subscription or higher)
 * @params  Body: name, permissions, expiresIn
 */
router.post(
  '/api-keys',
  requireSubscription(SubscriptionTier.BASIC),
  requirePermission('api_access'),
  validate.body(z.object({
    name: z.string().min(3).max(50),
    permissions: z.array(z.string()).optional(),
    expiresIn: z.number().int().min(3600).max(31536000).optional() // 1 hour to 1 year
  })),
  userController.generateApiKey
);

/**
 * @route   DELETE /api/users/api-keys/:keyId
 * @desc    Revoke API key
 * @access  Private
 * @params  URL: keyId (UUID)
 */
router.delete(
  '/api-keys/:keyId',
  requirePermission('api_access'),
  validate.params(z.object({
    keyId: commonSchemas.uuid
  })),
  userController.revokeApiKey
);

/**
 * @route   POST /api/users/verify-wallet
 * @desc    Verify wallet ownership
 * @access  Private
 * @params  Body: signature, message
 */
router.post(
  '/verify-wallet',
  requirePermission('manage_profile'),
  verifyWalletOwnership,
  (req, res, next) => {
    res.json({
      success: true,
      data: {
        verified: true,
        walletAddress: req.user?.walletAddress,
        verifiedAt: new Date()
      },
      message: 'Wallet ownership verified successfully'
    });
  }
);

/**
 * @route   GET /api/users/export
 * @desc    Export user data (GDPR compliance)
 * @access  Private
 * @params  Query: format, includeTransactions, includeAgentData
 */
router.get(
  '/export',
  requirePermission('read_profile'),
  validate.query(z.object({
    format: z.enum(['json', 'csv', 'xml']).default('json'),
    includeTransactions: z.boolean().default(false),
    includeAgentData: z.boolean().default(false)
  })),
  userController.exportUserData
);

/**
 * @route   DELETE /api/users/account
 * @desc    Delete user account
 * @access  Private
 * @params  Body: confirmation
 */
router.delete(
  '/account',
  requirePermission('manage_profile'),
  verifyWalletOwnership,
  validate.body(z.object({
    confirmation: z.literal('DELETE_MY_ACCOUNT'),
    signature: z.string().min(1),
    message: z.string().min(1)
  })),
  userController.deleteUser
);

/**
 * @route   POST /api/users/logout
 * @desc    User logout (blacklist token)
 * @access  Private
 */
router.post(
  '/logout',
  (req, res, next) => {
    // This would blacklist the current JWT token
    res.json({
      success: true,
      data: null,
      message: 'Logged out successfully'
    });
  }
);

/**
 * @route   GET /api/users/settings
 * @desc    Get user settings and preferences
 * @access  Private
 */
router.get(
  '/settings',
  requirePermission('read_profile'),
  (req, res, next) => {
    res.json({
      success: true,
      data: {
        userId: req.user?.id,
        preferences: req.user?.preferences || {},
        riskTolerance: req.user?.riskTolerance,
        subscriptionTier: req.user?.subscriptionTier,
        permissions: req.user?.permissions || []
      }
    });
  }
);

/**
 * @route   PUT /api/users/settings
 * @desc    Update user settings and preferences
 * @access  Private
 * @params  Body: preferences, riskTolerance
 */
router.put(
  '/settings',
  requirePermission('manage_profile'),
  validate.body(z.object({
    preferences: z.object({
      notifications: z.object({
        email: z.boolean(),
        push: z.boolean(),
        sms: z.boolean(),
        arbitrageAlerts: z.boolean(),
        portfolioRebalancing: z.boolean(),
        securityAlerts: z.boolean()
      }).partial(),
      trading: z.object({
        maxSlippage: z.number().min(0).max(10),
        autoRebalance: z.boolean(),
        rebalanceThreshold: z.number().min(1).max(50),
        preferredDexes: z.array(z.string()),
        blacklistedTokens: z.array(commonSchemas.ethereumAddress)
      }).partial(),
      risk: z.object({
        maxPositionSize: z.number().min(0).max(100),
        stopLossPercentage: z.number().min(0).max(50),
        takeProfitPercentage: z.number().min(0).max(200),
        allowLeverage: z.boolean(),
        maxLeverage: z.number().min(1).max(10)
      }).partial()
    }).partial().optional(),
    riskTolerance: commonSchemas.riskTolerance.optional()
  })),
  (req, res, next) => {
    res.json({
      success: true,
      data: {
        userId: req.user?.id,
        updatedSettings: req.body,
        updatedAt: new Date()
      },
      message: 'Settings updated successfully'
    });
  }
);

export default router;
