import { Request, Response, NextFunction } from 'express';
import { portfolioService } from '../services/portfolioService';
import { chainlinkService } from '../services/chainlinkService';
import { logger } from '../utils/logger';
import { 
  CreatePortfolioDto, 
  UpdatePortfolioDto,
  RebalanceTrigger,
  RebalanceFrequency 
} from '../models/Portfolio';
import { validateRequest } from '../utils/validators';
import { ApiResponse, PaginationParams } from '../types/api';

class PortfolioController {
  /**
   * Create a new portfolio
   */
  public async createPortfolio(
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

      const createPortfolioDto: CreatePortfolioDto = req.body;

      // Validate input
      const validation = validateRequest(createPortfolioDto, {
        name: { required: true, type: 'string', minLength: 3, maxLength: 50 },
        targetAllocation: { required: true, type: 'array', minLength: 1 },
        rebalanceSettings: { required: true, type: 'object' },
        chainId: { required: true, type: 'number' }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid portfolio configuration',
          details: validation.errors
        });
        return;
      }

      // Validate allocation percentages sum to 100%
      const totalAllocation = createPortfolioDto.targetAllocation.reduce(
        (sum, allocation) => sum + allocation.percentage,
        0
      );

      if (Math.abs(totalAllocation - 100) > 0.01) {
        res.status(400).json({
          success: false,
          error: 'Invalid Allocation',
          message: 'Target allocation percentages must sum to 100%'
        });
        return;
      }

      // Validate token addresses and get current prices
      const enrichedAllocations = await Promise.all(
        createPortfolioDto.targetAllocation.map(async (allocation) => {
          // Get token price from Chainlink if available
          const priceData = await chainlinkService.getTokenPrice(
            allocation.tokenAddress,
            createPortfolioDto.chainId
          );

          return {
            ...allocation,
            priceUSD: priceData?.price || 0,
            decimals: priceData?.decimals || 18,
            isVerified: !!priceData
          };
        })
      );

      const portfolioData = {
        ...createPortfolioDto,
        targetAllocation: enrichedAllocations
      };

      const portfolio = await portfolioService.createPortfolio(userId, portfolioData);

      logger.info(`Portfolio created successfully`, {
        userId,
        portfolioId: portfolio.id,
        name: portfolio.name,
        chainId: portfolio.chainId
      });

      const response: ApiResponse<typeof portfolio> = {
        success: true,
        data: portfolio,
        message: 'Portfolio created successfully'
      };

      res.status(201).json(response);
    } catch (error) {
      logger.error('Error creating portfolio:', error);
      next(error);
    }
  }

  /**
   * Get portfolio by ID
   */
  public async getPortfolio(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      const portfolio = await portfolioService.getPortfolioById(portfolioId, userId);

      if (!portfolio) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'Portfolio not found'
        });
        return;
      }

      // Enrich with real-time prices and performance data
      const enrichedPortfolio = await this.enrichPortfolioData(portfolio);

      const response: ApiResponse<typeof enrichedPortfolio> = {
        success: true,
        data: enrichedPortfolio
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting portfolio:', error);
      next(error);
    }
  }

  /**
   * Get all portfolios for a user
   */
  public async getUserPortfolios(
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
        limit: parseInt(req.query.limit as string) || 10,
        sortBy: req.query.sortBy as string || 'createdAt',
        sortOrder: (req.query.sortOrder as 'asc' | 'desc') || 'desc'
      };

      const filters = {
        chainId: req.query.chainId ? parseInt(req.query.chainId as string) : undefined,
        isActive: req.query.isActive === 'true' ? true : req.query.isActive === 'false' ? false : undefined
      };

      const result = await portfolioService.getUserPortfolios(userId, filters, pagination);

      // Enrich portfolios with real-time data
      const enrichedPortfolios = await Promise.all(
        result.portfolios.map(portfolio => this.enrichPortfolioData(portfolio))
      );

      const response: ApiResponse<typeof enrichedPortfolios> = {
        success: true,
        data: enrichedPortfolios,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: result.total,
          totalPages: Math.ceil(result.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting user portfolios:', error);
      next(error);
    }
  }

  /**
   * Update portfolio
   */
  public async updatePortfolio(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;
      const updateData: UpdatePortfolioDto = req.body;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      // Validate allocation if provided
      if (updateData.targetAllocation) {
        const totalAllocation = updateData.targetAllocation.reduce(
          (sum, allocation) => sum + allocation.percentage,
          0
        );

        if (Math.abs(totalAllocation - 100) > 0.01) {
          res.status(400).json({
            success: false,
            error: 'Invalid Allocation',
            message: 'Target allocation percentages must sum to 100%'
          });
          return;
        }
      }

      const updatedPortfolio = await portfolioService.updatePortfolio(portfolioId, userId, updateData);

      logger.info(`Portfolio updated successfully`, {
        userId,
        portfolioId,
        updates: Object.keys(updateData)
      });

      const response: ApiResponse<typeof updatedPortfolio> = {
        success: true,
        data: updatedPortfolio,
        message: 'Portfolio updated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error updating portfolio:', error);
      next(error);
    }
  }

  /**
   * Rebalance portfolio
   */
  public async rebalancePortfolio(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;
      const { strategy, maxSlippage, gasOptimization } = req.body;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      // Get current portfolio state
      const portfolio = await portfolioService.getPortfolioById(portfolioId, userId);
      if (!portfolio) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'Portfolio not found'
        });
        return;
      }

      // Calculate rebalancing requirements
      const rebalanceAnalysis = await portfolioService.analyzeRebalancingNeeds(portfolioId);
      
      if (!rebalanceAnalysis.needsRebalancing) {
        res.status(200).json({
          success: true,
          data: {
            message: 'Portfolio is already balanced',
            currentAllocation: rebalanceAnalysis.currentAllocation,
            targetAllocation: rebalanceAnalysis.targetAllocation,
            drift: rebalanceAnalysis.maxDrift
          }
        });
        return;
      }

      // Execute rebalancing using Chainlink Automation
      const rebalanceResult = await portfolioService.executeRebalancing({
        portfolioId,
        userId,
        strategy: strategy || 'optimal',
        maxSlippage: maxSlippage || 0.5,
        gasOptimization: gasOptimization !== false,
        analysis: rebalanceAnalysis
      });

      logger.info(`Portfolio rebalancing initiated`, {
        userId,
        portfolioId,
        txHashes: rebalanceResult.transactions.map(tx => tx.hash),
        estimatedGasCost: rebalanceResult.estimatedGasCost
      });

      const response: ApiResponse<typeof rebalanceResult> = {
        success: true,
        data: rebalanceResult,
        message: 'Portfolio rebalancing initiated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error rebalancing portfolio:', error);
      next(error);
    }
  }

  /**
   * Get portfolio performance
   */
  public async getPortfolioPerformance(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      const timeRange = {
        start: req.query.start ? new Date(req.query.start as string) : new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
        end: req.query.end ? new Date(req.query.end as string) : new Date()
      };

      const performance = await portfolioService.getPortfolioPerformance(portfolioId, userId, timeRange);

      const response: ApiResponse<typeof performance> = {
        success: true,
        data: performance
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting portfolio performance:', error);
      next(error);
    }
  }

  /**
   * Get rebalancing history
   */
  public async getRebalancingHistory(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      const pagination: PaginationParams = {
        page: parseInt(req.query.page as string) || 1,
        limit: parseInt(req.query.limit as string) || 20,
        sortBy: 'timestamp',
        sortOrder: 'desc'
      };

      const history = await portfolioService.getRebalancingHistory(portfolioId, userId, pagination);

      const response: ApiResponse<typeof history.records> = {
        success: true,
        data: history.records,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: history.total,
          totalPages: Math.ceil(history.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting rebalancing history:', error);
      next(error);
    }
  }

  /**
   * Simulate portfolio rebalancing
   */
  public async simulateRebalancing(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;
      const { newAllocation, strategy } = req.body;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      if (!newAllocation || !Array.isArray(newAllocation)) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'New allocation is required'
        });
        return;
      }

      const simulation = await portfolioService.simulateRebalancing({
        portfolioId,
        userId,
        newAllocation,
        strategy: strategy || 'optimal'
      });

      const response: ApiResponse<typeof simulation> = {
        success: true,
        data: simulation
      };

      res.json(response);
    } catch (error) {
      logger.error('Error simulating portfolio rebalancing:', error);
      next(error);
    }
  }

  /**
   * Setup automated rebalancing
   */
  public async setupAutomatedRebalancing(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;
      const { 
        trigger, 
        threshold, 
        frequency, 
        slippageTolerance,
        gasOptimization 
      } = req.body;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      // Validate rebalancing parameters
      const validation = validateRequest(req.body, {
        trigger: { required: true, type: 'string', enum: Object.values(RebalanceTrigger) },
        threshold: { required: true, type: 'number', min: 0.1, max: 50 },
        frequency: { required: false, type: 'string', enum: Object.values(RebalanceFrequency) },
        slippageTolerance: { required: false, type: 'number', min: 0.1, max: 10 }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid rebalancing configuration',
          details: validation.errors
        });
        return;
      }

      // Setup Chainlink Automation for automated rebalancing
      const automationSetup = await chainlinkService.setupPortfolioAutomation({
        portfolioId,
        userId,
        trigger,
        threshold,
        frequency,
        slippageTolerance: slippageTolerance || 0.5,
        gasOptimization: gasOptimization !== false
      });

      const updatedPortfolio = await portfolioService.updateRebalanceSettings(portfolioId, userId, {
        enabled: true,
        trigger,
        threshold,
        frequency,
        slippageTolerance: slippageTolerance || 0.5,
        gasOptimization: gasOptimization !== false,
        automationId: automationSetup.upkeepId
      });

      logger.info(`Automated rebalancing setup`, {
        userId,
        portfolioId,
        upkeepId: automationSetup.upkeepId,
        trigger,
        threshold
      });

      const response: ApiResponse<typeof updatedPortfolio> = {
        success: true,
        data: updatedPortfolio,
        message: 'Automated rebalancing setup successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error setting up automated rebalancing:', error);
      next(error);
    }
  }

  /**
   * Delete portfolio
   */
  public async deletePortfolio(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const portfolioId = req.params.portfolioId;

      if (!portfolioId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Portfolio ID is required'
        });
        return;
      }

      await portfolioService.deletePortfolio(portfolioId, userId);

      logger.info(`Portfolio deleted`, { userId, portfolioId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'Portfolio deleted successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error deleting portfolio:', error);
      next(error);
    }
  }

  /**
   * Helper method to enrich portfolio data with real-time information
   */
  private async enrichPortfolioData(portfolio: any): Promise<any> {
    try {
      // Get real-time prices for all holdings
      const enrichedHoldings = await Promise.all(
        portfolio.currentAllocation.map(async (holding: any) => {
          const priceData = await chainlinkService.getTokenPrice(
            holding.tokenAddress,
            portfolio.chainId
          );

          const currentValue = holding.amount * (priceData?.price || holding.priceUSD);
          
          return {
            ...holding,
            currentPriceUSD: priceData?.price || holding.priceUSD,
            currentValueUSD: currentValue,
            priceChange24h: priceData?.change24h || 0,
            isStale: priceData ? Date.now() - priceData.timestamp.getTime() > 300000 : true // 5 minutes
          };
        })
      );

      // Calculate current total value
      const currentTotalValue = enrichedHoldings.reduce(
        (sum, holding) => sum + holding.currentValueUSD,
        0
      );

      // Calculate current allocation percentages
      const currentAllocationPercentages = enrichedHoldings.map(holding => ({
        ...holding,
        currentPercentage: (holding.currentValueUSD / currentTotalValue) * 100
      }));

      return {
        ...portfolio,
        currentAllocation: currentAllocationPercentages,
        totalValueUSD: currentTotalValue,
        lastPriceUpdate: new Date(),
        drift: this.calculatePortfolioDrift(portfolio.targetAllocation, currentAllocationPercentages)
      };
    } catch (error) {
      logger.error('Error enriching portfolio data:', error);
      return portfolio;
    }
  }

  /**
   * Helper method to calculate portfolio drift
   */
  private calculatePortfolioDrift(targetAllocation: any[], currentAllocation: any[]): number {
    let maxDrift = 0;

    for (const target of targetAllocation) {
      const current = currentAllocation.find(c => c.tokenAddress === target.tokenAddress);
      if (current) {
        const drift = Math.abs(target.percentage - current.currentPercentage);
        maxDrift = Math.max(maxDrift, drift);
      }
    }

    return maxDrift;
  }
}

export const portfolioController = new PortfolioController();
export default portfolioController;
