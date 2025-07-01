import { Request, Response, NextFunction } from 'express';
import { userService } from '../services/userService';
import { logger } from '../utils/logger';
import { 
  CreateUserDto, 
  UpdateUserDto, 
  RiskTolerance, 
  SubscriptionTier 
} from '../models/User';
import { validateRequest } from '../utils/validators';
import { ApiResponse, PaginationParams } from '../types/api';
import { ethers } from 'ethers';

class UserController {
  /**
   * Create new user account
   */
  public async createUser(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const createUserDto: CreateUserDto = req.body;

      // Validate input
      const validation = validateRequest(createUserDto, {
        walletAddress: { 
          required: true, 
          type: 'string', 
          pattern: /^0x[a-fA-F0-9]{40}$/ 
        },
        email: { 
          required: false, 
          type: 'string', 
          pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/ 
        },
        riskTolerance: { 
          required: false, 
          type: 'string', 
          enum: Object.values(RiskTolerance) 
        }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid user data',
          details: validation.errors
        });
        return;
      }

      // Validate wallet address format
      if (!ethers.utils.isAddress(createUserDto.walletAddress)) {
        res.status(400).json({
          success: false,
          error: 'Invalid Address',
          message: 'Invalid Ethereum wallet address format'
        });
        return;
      }

      // Check if user already exists
      const existingUser = await userService.getUserByWalletAddress(createUserDto.walletAddress);
      if (existingUser) {
        res.status(409).json({
          success: false,
          error: 'User Exists',
          message: 'User with this wallet address already exists'
        });
        return;
      }

      const user = await userService.createUser(createUserDto);

      logger.info(`User created successfully`, {
        userId: user.id,
        walletAddress: user.walletAddress,
        email: user.email ? '***@' + user.email.split('@')[1] : undefined
      });

      // Remove sensitive information from response
      const sanitizedUser = {
        ...user,
        apiKeys: undefined
      };

      const response: ApiResponse<typeof sanitizedUser> = {
        success: true,
        data: sanitizedUser,
        message: 'User created successfully'
      };

      res.status(201).json(response);
    } catch (error) {
      logger.error('Error creating user:', error);
      next(error);
    }
  }

  /**
   * Get user profile
   */
  public async getUserProfile(
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

      const user = await userService.getUserById(userId);
      if (!user) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'User not found'
        });
        return;
      }

      // Get additional profile data
      const profileData = await userService.getUserProfileData(userId);

      // Remove sensitive information from response
      const sanitizedUser = {
        ...user,
        ...profileData,
        apiKeys: undefined
      };

      const response: ApiResponse<typeof sanitizedUser> = {
        success: true,
        data: sanitizedUser
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting user profile:', error);
      next(error);
    }
  }

  /**
   * Update user profile
   */
  public async updateUserProfile(
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

      const updateData: UpdateUserDto = req.body;

      // Validate update data
      if (Object.keys(updateData).length === 0) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'No update data provided'
        });
        return;
      }

      // Validate email format if provided
      if (updateData.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(updateData.email)) {
        res.status(400).json({
          success: false,
          error: 'Invalid Email',
          message: 'Invalid email address format'
        });
        return;
      }

      // Validate risk tolerance if provided
      if (updateData.riskTolerance && !Object.values(RiskTolerance).includes(updateData.riskTolerance)) {
        res.status(400).json({
          success: false,
          error: 'Invalid Risk Tolerance',
          message: 'Invalid risk tolerance value'
        });
        return;
      }

      const updatedUser = await userService.updateUser(userId, updateData);

      logger.info(`User profile updated`, {
        userId,
        updatedFields: Object.keys(updateData)
      });

      // Remove sensitive information from response
      const sanitizedUser = {
        ...updatedUser,
        apiKeys: undefined
      };

      const response: ApiResponse<typeof sanitizedUser> = {
        success: true,
        data: sanitizedUser,
        message: 'User profile updated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error updating user profile:', error);
      next(error);
    }
  }

  /**
   * Get user's activity history
   */
  public async getUserActivity(
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
        sortBy: req.query.sortBy as string || 'timestamp',
        sortOrder: (req.query.sortOrder as 'asc' | 'desc') || 'desc'
      };

      const filters = {
        type: req.query.type as string,
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined
      };

      const activity = await userService.getUserActivity(userId, filters, pagination);

      const response: ApiResponse<typeof activity.activities> = {
        success: true,
        data: activity.activities,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: activity.total,
          totalPages: Math.ceil(activity.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting user activity:', error);
      next(error);
    }
  }

  /**
   * Get user's dashboard summary
   */
  public async getUserDashboard(
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

      const dashboard = await userService.getUserDashboard(userId, timeRange);

      const response: ApiResponse<typeof dashboard> = {
        success: true,
        data: dashboard
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting user dashboard:', error);
      next(error);
    }
  }

  /**
   * Update user subscription
   */
  public async updateSubscription(
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

      const { tier, paymentMethod } = req.body;

      // Validate subscription tier
      if (!Object.values(SubscriptionTier).includes(tier)) {
        res.status(400).json({
          success: false,
          error: 'Invalid Tier',
          message: 'Invalid subscription tier'
        });
        return;
      }

      const subscription = await userService.updateSubscription(userId, tier, paymentMethod);

      logger.info(`User subscription updated`, {
        userId,
        newTier: tier,
        paymentMethod: paymentMethod ? 'provided' : 'not_provided'
      });

      const response: ApiResponse<typeof subscription> = {
        success: true,
        data: subscription,
        message: 'Subscription updated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error updating subscription:', error);
      next(error);
    }
  }

  /**
   * Get user's API usage statistics
   */
  public async getApiUsage(
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
        start: req.query.start ? new Date(req.query.start as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days
        end: req.query.end ? new Date(req.query.end as string) : new Date()
      };

      const usage = await userService.getApiUsage(userId, timeRange);

      const response: ApiResponse<typeof usage> = {
        success: true,
        data: usage
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting API usage:', error);
      next(error);
    }
  }

  /**
   * Generate new API key
   */
  public async generateApiKey(
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

      const { name, permissions, expiresIn } = req.body;

      // Validate input
      const validation = validateRequest(req.body, {
        name: { required: true, type: 'string', minLength: 3, maxLength: 50 },
        permissions: { required: false, type: 'array' },
        expiresIn: { required: false, type: 'number', min: 3600, max: 31536000 } // 1 hour to 1 year
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid API key request',
          details: validation.errors
        });
        return;
      }

      const apiKey = await userService.generateApiKey(userId, {
        name,
        permissions: permissions || ['read'],
        expiresIn: expiresIn || 2592000 // 30 days default
      });

      logger.info(`API key generated`, {
        userId,
        keyName: name,
        permissions: permissions || ['read']
      });

      const response: ApiResponse<typeof apiKey> = {
        success: true,
        data: apiKey,
        message: 'API key generated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error generating API key:', error);
      next(error);
    }
  }

  /**
   * Revoke API key
   */
  public async revokeApiKey(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const keyId = req.params.keyId;

      if (!keyId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'API key ID is required'
        });
        return;
      }

      const revoked = await userService.revokeApiKey(userId, keyId);

      if (!revoked) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'API key not found'
        });
        return;
      }

      logger.info(`API key revoked`, { userId, keyId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'API key revoked successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error revoking API key:', error);
      next(error);
    }
  }

  /**
   * Delete user account
   */
  public async deleteUser(
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

      const { confirmation } = req.body;

      if (confirmation !== 'DELETE_MY_ACCOUNT') {
        res.status(400).json({
          success: false,
          error: 'Invalid Confirmation',
          message: 'Account deletion requires proper confirmation'
        });
        return;
      }

      // Perform account deletion with cleanup
      await userService.deleteUser(userId);

      logger.info(`User account deleted`, { userId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'User account deleted successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error deleting user account:', error);
      next(error);
    }
  }

  /**
   * Export user data (GDPR compliance)
   */
  public async exportUserData(
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

      const format = req.query.format as string || 'json';
      const includeTransactions = req.query.includeTransactions === 'true';
      const includeAgentData = req.query.includeAgentData === 'true';

      const exportData = await userService.exportUserData(userId, {
        format,
        includeTransactions,
        includeAgentData
      });

      logger.info(`User data exported`, {
        userId,
        format,
        includeTransactions,
        includeAgentData
      });

      if (format === 'json') {
        const response: ApiResponse<typeof exportData> = {
          success: true,
          data: exportData,
          message: 'User data exported successfully'
        };
        res.json(response);
      } else {
        // For CSV or other formats, send as file download
        res.setHeader('Content-Type', 'application/octet-stream');
        res.setHeader('Content-Disposition', `attachment; filename=user-data-${userId}.${format}`);
        res.send(exportData);
      }
    } catch (error) {
      logger.error('Error exporting user data:', error);
      next(error);
    }
  }
}

export const userController = new UserController();
export default userController;
