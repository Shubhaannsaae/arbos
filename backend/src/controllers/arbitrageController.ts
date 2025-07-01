import { Request, Response, NextFunction } from 'express';
import { arbitrageService } from '../services/arbitrageService';
import { chainlinkService } from '../services/chainlinkService';
import { crossChainService } from '../services/crossChainService';
import { logger } from '../utils/logger';
import { 
  CreateArbitrageOpportunityDto,
  ArbitrageFilter,
  OpportunityStatus 
} from '../models/ArbitrageOpportunity';
import { validateRequest } from '../utils/validators';
import { ApiResponse, PaginationParams } from '../types/api';

class ArbitrageController {
  /**
   * Get arbitrage opportunities
   */
  public async getOpportunities(
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
        sortBy: req.query.sortBy as string || 'netProfitUSD',
        sortOrder: (req.query.sortOrder as 'asc' | 'desc') || 'desc'
      };

      const filters: ArbitrageFilter = {
        minProfit: req.query.minProfit ? parseFloat(req.query.minProfit as string) : 1,
        maxRiskScore: req.query.maxRiskScore ? parseInt(req.query.maxRiskScore as string) : 80,
        chains: req.query.chains ? (req.query.chains as string).split(',').map(Number) : undefined,
        exchanges: req.query.exchanges ? (req.query.exchanges as string).split(',') : undefined,
        tokens: req.query.tokens ? (req.query.tokens as string).split(',') : undefined,
        status: req.query.status ? [req.query.status as OpportunityStatus] : [OpportunityStatus.DETECTED, OpportunityStatus.APPROVED],
        timeRange: req.query.timeRange ? {
          start: new Date(req.query.startDate as string || Date.now() - 24 * 60 * 60 * 1000),
          end: new Date(req.query.endDate as string || Date.now())
        } : undefined
      };

      const result = await arbitrageService.getOpportunities(userId, filters, pagination);

      // Enrich with real-time Chainlink price data
      const enrichedOpportunities = await Promise.all(
        result.opportunities.map(async (opportunity) => {
          const priceData = await chainlinkService.getLatestPrice(
            opportunity.tokenPair,
            opportunity.sourceChain.chainId
          );
          
          return {
            ...opportunity,
            currentPrice: priceData?.price,
            priceTimestamp: priceData?.timestamp,
            isPriceStale: priceData ? Date.now() - priceData.timestamp.getTime() > 300000 : true // 5 minutes
          };
        })
      );

      const response: ApiResponse<typeof enrichedOpportunities> = {
        success: true,
        data: enrichedOpportunities,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: result.total,
          totalPages: Math.ceil(result.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting arbitrage opportunities:', error);
      next(error);
    }
  }

  /**
   * Execute arbitrage opportunity
   */
  public async executeArbitrage(
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

      const { opportunityId, amount, maxSlippage, gasLimit } = req.body;

      // Validate input
      const validation = validateRequest(req.body, {
        opportunityId: { required: true, type: 'string' },
        amount: { required: true, type: 'number', min: 0 },
        maxSlippage: { required: false, type: 'number', min: 0, max: 10 },
        gasLimit: { required: false, type: 'number', min: 21000 }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid execution parameters',
          details: validation.errors
        });
        return;
      }

      // Get opportunity details
      const opportunity = await arbitrageService.getOpportunityById(opportunityId, userId);
      if (!opportunity) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'Arbitrage opportunity not found'
        });
        return;
      }

      if (opportunity.status !== OpportunityStatus.DETECTED && opportunity.status !== OpportunityStatus.APPROVED) {
        res.status(400).json({
          success: false,
          error: 'Invalid Status',
          message: 'Opportunity is not available for execution'
        });
        return;
      }

      // Validate amount bounds
      if (amount < opportunity.minTradeSize || amount > opportunity.maxTradeSize) {
        res.status(400).json({
          success: false,
          error: 'Invalid Amount',
          message: `Amount must be between ${opportunity.minTradeSize} and ${opportunity.maxTradeSize}`
        });
        return;
      }

      // Check if cross-chain execution is needed
      const isCrossChain = opportunity.sourceChain.chainId !== opportunity.targetChain.chainId;
      
      let executionResult;
      if (isCrossChain) {
        // Use Chainlink CCIP for cross-chain arbitrage
        executionResult = await crossChainService.executeCrossChainArbitrage({
          opportunityId,
          amount,
          maxSlippage: maxSlippage || 0.5,
          gasLimit,
          sourceChain: opportunity.sourceChain.chainId,
          targetChain: opportunity.targetChain.chainId,
          userId
        });
      } else {
        // Same-chain arbitrage execution
        executionResult = await arbitrageService.executeArbitrage({
          opportunityId,
          amount,
          maxSlippage: maxSlippage || 0.5,
          gasLimit,
          userId
        });
      }

      logger.info(`Arbitrage execution initiated`, {
        userId,
        opportunityId,
        amount,
        isCrossChain,
        txHash: executionResult.txHash
      });

      const response: ApiResponse<typeof executionResult> = {
        success: true,
        data: executionResult,
        message: 'Arbitrage execution initiated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error executing arbitrage:', error);
      next(error);
    }
  }

  /**
   * Get arbitrage execution history
   */
  public async getExecutionHistory(
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
        sortBy: req.query.sortBy as string || 'executedAt',
        sortOrder: (req.query.sortOrder as 'asc' | 'desc') || 'desc'
      };

      const filters = {
        status: req.query.status as OpportunityStatus,
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined,
        minProfit: req.query.minProfit ? parseFloat(req.query.minProfit as string) : undefined,
        tokenPair: req.query.tokenPair as string
      };

      const result = await arbitrageService.getExecutionHistory(userId, filters, pagination);

      const response: ApiResponse<typeof result.executions> = {
        success: true,
        data: result.executions,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: result.total,
          totalPages: Math.ceil(result.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting execution history:', error);
      next(error);
    }
  }

  /**
   * Get arbitrage analytics
   */
  public async getAnalytics(
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

      const analytics = await arbitrageService.getAnalytics(userId, timeRange);

      const response: ApiResponse<typeof analytics> = {
        success: true,
        data: analytics
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting arbitrage analytics:', error);
      next(error);
    }
  }

  /**
   * Get real-time market data for arbitrage
   */
  public async getMarketData(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const tokenPairs = req.query.pairs ? (req.query.pairs as string).split(',') : ['ETH/USD', 'AVAX/USD', 'LINK/USD'];
      const chains = req.query.chains ? (req.query.chains as string).split(',').map(Number) : [1, 43114, 43113];

      const marketData = await Promise.all(
        tokenPairs.map(async (pair) => {
          const chainData = await Promise.all(
            chains.map(async (chainId) => {
              const priceData = await chainlinkService.getLatestPrice(pair, chainId);
              const feedInfo = await chainlinkService.getFeedInfo(pair, chainId);
              
              return {
                chainId,
                chainName: chainlinkService.getNetworkName(chainId),
                price: priceData?.price,
                timestamp: priceData?.timestamp,
                feedAddress: feedInfo?.address,
                heartbeat: feedInfo?.heartbeat,
                decimals: feedInfo?.decimals,
                isStale: priceData ? Date.now() - priceData.timestamp.getTime() > (feedInfo?.heartbeat || 3600) * 1000 : true
              };
            })
          );

          return {
            pair,
            chains: chainData,
            maxSpread: this.calculateMaxSpread(chainData),
            arbitrageOpportunities: this.identifyArbitrageOpportunities(chainData)
          };
        })
      );

      const response: ApiResponse<typeof marketData> = {
        success: true,
        data: marketData
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting market data:', error);
      next(error);
    }
  }

  /**
   * Submit manual arbitrage opportunity
   */
  public async submitOpportunity(
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

      const opportunityDto: CreateArbitrageOpportunityDto = req.body;

      // Validate input
      const validation = validateRequest(opportunityDto, {
        tokenPair: { required: true, type: 'string' },
        sourceExchange: { required: true, type: 'object' },
        targetExchange: { required: true, type: 'object' },
        sourceChain: { required: true, type: 'object' },
        targetChain: { required: true, type: 'object' },
        priceDifference: { required: true, type: 'number', min: 0 },
        potentialProfit: { required: true, type: 'number', min: 0 },
        estimatedGasCost: { required: true, type: 'number', min: 0 }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid opportunity data',
          details: validation.errors
        });
        return;
      }

      // Verify price data with Chainlink feeds
      const sourcePrice = await chainlinkService.getLatestPrice(
        opportunityDto.tokenPair,
        opportunityDto.sourceChain.chainId
      );
      const targetPrice = await chainlinkService.getLatestPrice(
        opportunityDto.tokenPair,
        opportunityDto.targetChain.chainId
      );

      if (!sourcePrice || !targetPrice) {
        res.status(400).json({
          success: false,
          error: 'Price Data Unavailable',
          message: 'Unable to verify prices with Chainlink feeds'
        });
        return;
      }

      const actualPriceDifference = Math.abs(sourcePrice.price - targetPrice.price) / Math.min(sourcePrice.price, targetPrice.price) * 100;
      
      if (Math.abs(actualPriceDifference - opportunityDto.priceDifference) > 1.0) {
        res.status(400).json({
          success: false,
          error: 'Price Mismatch',
          message: 'Submitted price difference does not match Chainlink data'
        });
        return;
      }

      const opportunity = await arbitrageService.createOpportunity(userId, opportunityDto);

      logger.info(`Manual arbitrage opportunity submitted`, {
        userId,
        opportunityId: opportunity.id,
        tokenPair: opportunity.tokenPair,
        priceDifference: opportunity.priceDifferencePercentage
      });

      const response: ApiResponse<typeof opportunity> = {
        success: true,
        data: opportunity,
        message: 'Arbitrage opportunity submitted successfully'
      };

      res.status(201).json(response);
    } catch (error) {
      logger.error('Error submitting arbitrage opportunity:', error);
      next(error);
    }
  }

  /**
   * Cancel arbitrage opportunity
   */
  public async cancelOpportunity(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const opportunityId = req.params.opportunityId;

      if (!opportunityId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Opportunity ID is required'
        });
        return;
      }

      const cancelled = await arbitrageService.cancelOpportunity(opportunityId, userId);

      if (!cancelled) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'Opportunity not found or cannot be cancelled'
        });
        return;
      }

      logger.info(`Arbitrage opportunity cancelled`, { userId, opportunityId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'Arbitrage opportunity cancelled successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error cancelling arbitrage opportunity:', error);
      next(error);
    }
  }

  /**
   * Helper method to calculate maximum spread across chains
   */
  private calculateMaxSpread(chainData: any[]): number {
    const prices = chainData
      .filter(data => data.price && !data.isStale)
      .map(data => data.price);
    
    if (prices.length < 2) return 0;
    
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    
    return ((maxPrice - minPrice) / minPrice) * 100;
  }

  /**
   * Helper method to identify arbitrage opportunities
   */
  private identifyArbitrageOpportunities(chainData: any[]): any[] {
    const opportunities = [];
    const validData = chainData.filter(data => data.price && !data.isStale);
    
    for (let i = 0; i < validData.length; i++) {
      for (let j = i + 1; j < validData.length; j++) {
        const source = validData[i];
        const target = validData[j];
        
        const priceDiff = Math.abs(source.price - target.price) / Math.min(source.price, target.price) * 100;
        
        if (priceDiff > 0.5) { // Minimum 0.5% difference
          opportunities.push({
            sourceChain: source.chainId,
            targetChain: target.chainId,
            sourcePrice: source.price,
            targetPrice: target.price,
            priceDifferencePercentage: priceDiff,
            direction: source.price > target.price ? 'buy_target_sell_source' : 'buy_source_sell_target'
          });
        }
      }
    }
    
    return opportunities.sort((a, b) => b.priceDifferencePercentage - a.priceDifferencePercentage);
  }
}

export const arbitrageController = new ArbitrageController();
export default arbitrageController;
