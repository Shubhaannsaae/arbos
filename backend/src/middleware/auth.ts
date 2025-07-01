import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { expressjwt } from 'express-jwt';
import { logger } from '../utils/logger';
import { userService } from '../services/userService';
import { RiskTolerance, SubscriptionTier } from '../models/User';

// Extend Request interface to include user data
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        walletAddress: string;
        email?: string;
        riskTolerance: RiskTolerance;
        subscriptionTier: SubscriptionTier;
        permissions: string[];
        iat?: number;
        exp?: number;
      };
      auth?: any;
    }
  }
}

interface JWTPayload {
  id: string;
  walletAddress: string;
  email?: string;
  riskTolerance: RiskTolerance;
  subscriptionTier: SubscriptionTier;
  permissions: string[];
  iat?: number;
  exp?: number;
}

interface ApiKey {
  id: string;
  userId: string;
  name: string;
  permissions: string[];
  expiresAt?: Date;
  isActive: boolean;
}

class AuthMiddleware {
  private jwtSecret: string;
  private jwtExpiresIn: string;
  private refreshTokenSecret: string;
  private apiKeyCache: Map<string, ApiKey> = new Map();

  constructor() {
    this.jwtSecret = process.env.JWT_SECRET || this.generateSecureSecret();
    this.jwtExpiresIn = process.env.JWT_EXPIRES_IN || '24h';
    this.refreshTokenSecret = process.env.REFRESH_TOKEN_SECRET || this.generateSecureSecret();

    if (!process.env.JWT_SECRET) {
      logger.warn('JWT_SECRET not set in environment variables, using generated secret');
    }
  }

  /**
   * Generate a secure secret key
   */
  private generateSecureSecret(): string {
    return require('crypto').randomBytes(64).toString('hex');
  }

  /**
   * JWT Authentication middleware using express-jwt
   */
  public jwtAuth = expressjwt({
    secret: this.jwtSecret,
    algorithms: ['HS256'],
    credentialsRequired: true,
    getToken: this.extractToken,
    isRevoked: this.isTokenRevoked,
    onExpired: this.handleExpiredToken,
    requestProperty: 'auth'
  });

  /**
   * Optional JWT Authentication - continues even without token
   */
  public optionalJwtAuth = expressjwt({
    secret: this.jwtSecret,
    algorithms: ['HS256'],
    credentialsRequired: false,
    getToken: this.extractToken,
    isRevoked: this.isTokenRevoked,
    requestProperty: 'auth'
  });

  /**
   * Extract token from request headers or query parameters
   */
  private extractToken = (req: Request): string | undefined => {
    // Check Authorization header first
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer ')) {
      return req.headers.authorization.substring(7);
    }

    // Check API key in headers
    if (req.headers['x-api-key']) {
      return req.headers['x-api-key'] as string;
    }

    // Check query parameter (less secure, for testing only)
    if (req.query.token && typeof req.query.token === 'string') {
      return req.query.token;
    }

    return undefined;
  };

  /**
   * Check if token is revoked
   */
  private isTokenRevoked = async (req: Request, token: any): Promise<boolean> => {
    try {
      if (!token || !token.payload) {
        return true;
      }

      const userId = token.payload.id;
      if (!userId) {
        return true;
      }

      // Check if user still exists and is active
      const user = await userService.getUserById(userId);
      if (!user || !user.isActive) {
        logger.warn(`Token revoked for inactive user: ${userId}`);
        return true;
      }

      // Check token blacklist (would be implemented in production)
      const isBlacklisted = await this.isTokenBlacklisted(token.payload.jti);
      if (isBlacklisted) {
        logger.warn(`Token is blacklisted: ${token.payload.jti}`);
        return true;
      }

      return false;
    } catch (error) {
      logger.error('Error checking token revocation:', error);
      return true;
    }
  };

  /**
   * Handle expired tokens
   */
  private handleExpiredToken = (req: Request, res: Response): void => {
    logger.warn(`Expired token attempt from IP: ${req.ip}`);
    res.status(401).json({
      success: false,
      error: 'TokenExpired',
      message: 'JWT token has expired',
      code: 'TOKEN_EXPIRED'
    });
  };

  /**
   * Middleware to populate user data after JWT validation
   */
  public populateUser = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (!req.auth) {
        return next();
      }

      const userId = req.auth.id;
      if (!userId) {
        return next();
      }

      // Get fresh user data
      const user = await userService.getUserById(userId);
      if (!user) {
        return res.status(401).json({
          success: false,
          error: 'UserNotFound',
          message: 'User associated with token not found'
        });
      }

      // Populate request with user data
      req.user = {
        id: user.id,
        walletAddress: user.walletAddress,
        email: user.email,
        riskTolerance: user.riskTolerance,
        subscriptionTier: user.subscriptionTier,
        permissions: this.getUserPermissions(user.subscriptionTier),
        iat: req.auth.iat,
        exp: req.auth.exp
      };

      next();
    } catch (error) {
      logger.error('Error populating user data:', error);
      res.status(500).json({
        success: false,
        error: 'InternalServerError',
        message: 'Error loading user data'
      });
    }
  };

  /**
   * API Key authentication middleware
   */
  public apiKeyAuth = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const apiKey = req.headers['x-api-key'] as string;

      if (!apiKey) {
        return res.status(401).json({
          success: false,
          error: 'MissingApiKey',
          message: 'API key is required'
        });
      }

      // Validate API key
      const keyData = await this.validateApiKey(apiKey);
      if (!keyData) {
        return res.status(401).json({
          success: false,
          error: 'InvalidApiKey',
          message: 'Invalid or expired API key'
        });
      }

      // Get user data
      const user = await userService.getUserById(keyData.userId);
      if (!user || !user.isActive) {
        return res.status(401).json({
          success: false,
          error: 'UserInactive',
          message: 'User account is inactive'
        });
      }

      // Populate request with user data
      req.user = {
        id: user.id,
        walletAddress: user.walletAddress,
        email: user.email,
        riskTolerance: user.riskTolerance,
        subscriptionTier: user.subscriptionTier,
        permissions: keyData.permissions
      };

      next();
    } catch (error) {
      logger.error('Error in API key authentication:', error);
      res.status(500).json({
        success: false,
        error: 'InternalServerError',
        message: 'Authentication error'
      });
    }
  };

  /**
   * Role-based authorization middleware
   */
  public requirePermission = (permission: string | string[]) => {
    return (req: Request, res: Response, next: NextFunction): void => {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'Authentication required'
        });
      }

      const permissions = Array.isArray(permission) ? permission : [permission];
      const userPermissions = req.user.permissions || [];

      const hasPermission = permissions.some(perm => 
        userPermissions.includes(perm) || userPermissions.includes('admin')
      );

      if (!hasPermission) {
        logger.warn(`Permission denied for user ${req.user.id}`, {
          userId: req.user.id,
          requiredPermissions: permissions,
          userPermissions
        });

        return res.status(403).json({
          success: false,
          error: 'Forbidden',
          message: 'Insufficient permissions',
          requiredPermissions: permissions
        });
      }

      next();
    };
  };

  /**
   * Subscription tier middleware
   */
  public requireSubscription = (minTier: SubscriptionTier) => {
    return (req: Request, res: Response, next: NextFunction): void => {
      if (!req.user) {
        return res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'Authentication required'
        });
      }

      const tierHierarchy = {
        [SubscriptionTier.FREE]: 0,
        [SubscriptionTier.BASIC]: 1,
        [SubscriptionTier.PREMIUM]: 2,
        [SubscriptionTier.ENTERPRISE]: 3
      };

      const userTierLevel = tierHierarchy[req.user.subscriptionTier];
      const requiredTierLevel = tierHierarchy[minTier];

      if (userTierLevel < requiredTierLevel) {
        return res.status(403).json({
          success: false,
          error: 'SubscriptionRequired',
          message: `${minTier} subscription or higher required`,
          currentTier: req.user.subscriptionTier,
          requiredTier: minTier
        });
      }

      next();
    };
  };

  /**
   * Wallet ownership verification middleware
   */
  public verifyWalletOwnership = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
    try {
      const { signature, message } = req.body;
      const walletAddress = req.user?.walletAddress;

      if (!signature || !message || !walletAddress) {
        return res.status(400).json({
          success: false,
          error: 'MissingSignature',
          message: 'Wallet signature verification required'
        });
      }

      const isValid = await this.verifyWalletSignature(walletAddress, message, signature);
      if (!isValid) {
        return res.status(401).json({
          success: false,
          error: 'InvalidSignature',
          message: 'Wallet signature verification failed'
        });
      }

      next();
    } catch (error) {
      logger.error('Error verifying wallet ownership:', error);
      res.status(500).json({
        success: false,
        error: 'VerificationError',
        message: 'Wallet verification failed'
      });
    }
  };

  /**
   * Generate JWT token
   */
  public generateToken = (payload: Omit<JWTPayload, 'iat' | 'exp'>): string => {
    try {
      const tokenPayload = {
        ...payload,
        jti: this.generateTokenId(), // JWT ID for blacklisting
        iss: 'arbos-api', // Issuer
        aud: 'arbos-client' // Audience
      };

      return jwt.sign(tokenPayload, this.jwtSecret, {
        expiresIn: this.jwtExpiresIn,
        algorithm: 'HS256'
      });
    } catch (error) {
      logger.error('Error generating JWT token:', error);
      throw new Error('Token generation failed');
    }
  };

  /**
   * Generate refresh token
   */
  public generateRefreshToken = (userId: string): string => {
    try {
      return jwt.sign(
        { 
          userId, 
          type: 'refresh',
          jti: this.generateTokenId()
        },
        this.refreshTokenSecret,
        { expiresIn: '7d', algorithm: 'HS256' }
      );
    } catch (error) {
      logger.error('Error generating refresh token:', error);
      throw new Error('Refresh token generation failed');
    }
  };

  /**
   * Verify and refresh JWT token
   */
  public refreshToken = async (refreshToken: string): Promise<{ accessToken: string; refreshToken: string } | null> => {
    try {
      const decoded = jwt.verify(refreshToken, this.refreshTokenSecret) as any;
      
      if (decoded.type !== 'refresh') {
        return null;
      }

      const user = await userService.getUserById(decoded.userId);
      if (!user || !user.isActive) {
        return null;
      }

      const newAccessToken = this.generateToken({
        id: user.id,
        walletAddress: user.walletAddress,
        email: user.email,
        riskTolerance: user.riskTolerance,
        subscriptionTier: user.subscriptionTier,
        permissions: this.getUserPermissions(user.subscriptionTier)
      });

      const newRefreshToken = this.generateRefreshToken(user.id);

      // Blacklist old refresh token
      await this.blacklistToken(decoded.jti);

      return {
        accessToken: newAccessToken,
        refreshToken: newRefreshToken
      };
    } catch (error) {
      logger.error('Error refreshing token:', error);
      return null;
    }
  };

  /**
   * Blacklist token
   */
  public blacklistToken = async (tokenId: string): Promise<void> => {
    try {
      // Implementation would store blacklisted tokens in Redis or database
      // For now, we'll use in-memory storage
      logger.info(`Token blacklisted: ${tokenId}`);
    } catch (error) {
      logger.error('Error blacklisting token:', error);
    }
  };

  // Helper methods
  private generateTokenId(): string {
    return require('crypto').randomBytes(16).toString('hex');
  }

  private getUserPermissions(subscriptionTier: SubscriptionTier): string[] {
    const basePermissions = ['read_portfolio', 'read_agents', 'read_arbitrage'];
    
    switch (subscriptionTier) {
      case SubscriptionTier.FREE:
        return basePermissions;
      case SubscriptionTier.BASIC:
        return [...basePermissions, 'execute_trades', 'manage_portfolio'];
      case SubscriptionTier.PREMIUM:
        return [...basePermissions, 'execute_trades', 'manage_portfolio', 'advanced_analytics', 'priority_support'];
      case SubscriptionTier.ENTERPRISE:
        return [...basePermissions, 'execute_trades', 'manage_portfolio', 'advanced_analytics', 'priority_support', 'api_access', 'white_label'];
      default:
        return basePermissions;
    }
  }

  private async validateApiKey(apiKey: string): Promise<ApiKey | null> {
    try {
      // Check cache first
      if (this.apiKeyCache.has(apiKey)) {
        const cached = this.apiKeyCache.get(apiKey)!;
        if (cached.expiresAt && cached.expiresAt < new Date()) {
          this.apiKeyCache.delete(apiKey);
          return null;
        }
        return cached;
      }

      // Validate against database (implementation would query actual database)
      const keyData = await this.getApiKeyFromDatabase(apiKey);
      if (keyData && keyData.isActive) {
        // Cache for 5 minutes
        this.apiKeyCache.set(apiKey, keyData);
        setTimeout(() => this.apiKeyCache.delete(apiKey), 5 * 60 * 1000);
        return keyData;
      }

      return null;
    } catch (error) {
      logger.error('Error validating API key:', error);
      return null;
    }
  }

  private async getApiKeyFromDatabase(apiKey: string): Promise<ApiKey | null> {
    // Implementation would query database
    // For now, return null (would be implemented with actual database)
    return null;
  }

  private async isTokenBlacklisted(tokenId: string): Promise<boolean> {
    // Implementation would check Redis or database for blacklisted tokens
    return false;
  }

  private async verifyWalletSignature(walletAddress: string, message: string, signature: string): Promise<boolean> {
    try {
      const { ethers } = require('ethers');
      const recoveredAddress = ethers.utils.verifyMessage(message, signature);
      return recoveredAddress.toLowerCase() === walletAddress.toLowerCase();
    } catch (error) {
      logger.error('Error verifying wallet signature:', error);
      return false;
    }
  }
}

export const authMiddleware = new AuthMiddleware();

// Export individual middleware functions for ease of use
export const jwtAuth = authMiddleware.jwtAuth;
export const optionalJwtAuth = authMiddleware.optionalJwtAuth;
export const populateUser = authMiddleware.populateUser;
export const apiKeyAuth = authMiddleware.apiKeyAuth;
export const requirePermission = authMiddleware.requirePermission;
export const requireSubscription = authMiddleware.requireSubscription;
export const verifyWalletOwnership = authMiddleware.verifyWalletOwnership;
export const generateToken = authMiddleware.generateToken;
export const generateRefreshToken = authMiddleware.generateRefreshToken;
export const refreshToken = authMiddleware.refreshToken;
export const blacklistToken = authMiddleware.blacklistToken;

export default authMiddleware;
