import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import { CostCalculator, RouteCost } from './CostCalculator';
import { RouteOptimizer, OptimizationResult } from './RouteOptimizer';

export interface RouteOption {
  id: string;
  protocol: 'ccip' | 'layerzero' | 'polygon' | 'arbitrum' | 'avalanche';
  sourceChain: number;
  destinationChain: number;
  estimatedTime: number; // minutes
  reliability: number; // 0-100
  gasLimit: number;
  supportedTokens: string[];
}

export interface CrossChainRequest {
  sourceChain: number;
  destinationChain: number;
  token: string;
  amount: bigint;
  recipient: string;
  priorityMode: 'cost' | 'speed' | 'reliability';
  maxSlippage: number; // basis points
  deadline?: number; // timestamp
}

export interface RouteResult {
  selectedRoute: RouteOption;
  estimatedCost: RouteCost;
  estimatedTime: number;
  confidenceScore: number;
  backupRoutes: RouteOption[];
  warnings: string[];
}

export class OptimalRouter {
  private logger: Logger;
  private costCalculator: CostCalculator;
  private routeOptimizer: RouteOptimizer;
  private availableRoutes: Map<string, RouteOption> = new Map();

  // Official route configurations based on Chainlink CCIP and major bridges
  private readonly SUPPORTED_ROUTES: RouteOption[] = [
    // CCIP Routes (Primary - most reliable)
    {
      id: 'ccip-eth-avax',
      protocol: 'ccip',
      sourceChain: 1,
      destinationChain: 43114,
      estimatedTime: 10,
      reliability: 99,
      gasLimit: 500000,
      supportedTokens: ['0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8', '0x514910771AF9Ca656af840dff83E8264EcF986CA']
    },
    {
      id: 'ccip-eth-polygon',
      protocol: 'ccip',
      sourceChain: 1,
      destinationChain: 137,
      estimatedTime: 8,
      reliability: 99,
      gasLimit: 400000,
      supportedTokens: ['0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8', '0x514910771AF9Ca656af840dff83E8264EcF986CA']
    },
    {
      id: 'ccip-eth-arbitrum',
      protocol: 'ccip',
      sourceChain: 1,
      destinationChain: 42161,
      estimatedTime: 12,
      reliability: 99,
      gasLimit: 600000,
      supportedTokens: ['0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8', '0x514910771AF9Ca656af840dff83E8264EcF986CA']
    },
    // LayerZero Routes (Secondary - fast but higher cost)
    {
      id: 'lz-eth-avax',
      protocol: 'layerzero',
      sourceChain: 1,
      destinationChain: 43114,
      estimatedTime: 5,
      reliability: 95,
      gasLimit: 350000,
      supportedTokens: ['0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8', '0x514910771AF9Ca656af840dff83E8264EcF986CA']
    },
    // Native Bridge Routes (Chain-specific optimizations)
    {
      id: 'polygon-pos-bridge',
      protocol: 'polygon',
      sourceChain: 1,
      destinationChain: 137,
      estimatedTime: 25,
      reliability: 98,
      gasLimit: 200000,
      supportedTokens: ['0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8', '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599']
    },
    {
      id: 'arbitrum-bridge',
      protocol: 'arbitrum',
      sourceChain: 1,
      destinationChain: 42161,
      estimatedTime: 15,
      reliability: 97,
      gasLimit: 800000,
      supportedTokens: ['0xA0b86a33E6417aE4c2b09c2B8A3aBA6bb7D4A0F8', '0x514910771AF9Ca656af840dff83E8264EcF986CA']
    }
  ];

  constructor(costCalculator: CostCalculator, routeOptimizer: RouteOptimizer) {
    this.costCalculator = costCalculator;
    this.routeOptimizer = routeOptimizer;

    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/optimal-router.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });

    this.initializeRoutes();
  }

  /**
   * Find optimal route for cross-chain request
   */
  async findOptimalRoute(request: CrossChainRequest): Promise<RouteResult> {
    try {
      this.validateRequest(request);

      // Get available routes for the chain pair
      const availableRoutes = this.getAvailableRoutes(
        request.sourceChain,
        request.destinationChain,
        request.token
      );

      if (availableRoutes.length === 0) {
        throw new Error(`No routes available from chain ${request.sourceChain} to ${request.destinationChain}`);
      }

      // Calculate costs for all available routes
      const routesWithCosts = await Promise.all(
        availableRoutes.map(async (route) => ({
          route,
          cost: await this.costCalculator.calculateRouteCost(route, request.amount, request.token)
        }))
      );

      // Optimize route selection based on priority
      const optimizationResult = await this.routeOptimizer.optimizeRoute(
        routesWithCosts,
        request.priorityMode,
        {
          maxSlippage: request.maxSlippage,
          deadline: request.deadline,
          amount: request.amount
        }
      );

      // Select primary route and backups
      const selectedRoute = optimizationResult.primaryRoute.route;
      const backupRoutes = optimizationResult.backupRoutes.map(r => r.route);

      // Generate warnings
      const warnings = this.generateWarnings(request, selectedRoute, optimizationResult);

      const result: RouteResult = {
        selectedRoute,
        estimatedCost: optimizationResult.primaryRoute.cost,
        estimatedTime: selectedRoute.estimatedTime,
        confidenceScore: this.calculateConfidenceScore(selectedRoute, optimizationResult),
        backupRoutes,
        warnings
      };

      this.logger.info(`Optimal route selected`, {
        routeId: selectedRoute.id,
        protocol: selectedRoute.protocol,
        estimatedCost: optimizationResult.primaryRoute.cost.totalCost.toString(),
        estimatedTime: selectedRoute.estimatedTime,
        confidenceScore: result.confidenceScore
      });

      return result;

    } catch (error) {
      this.logger.error('Failed to find optimal route', {
        request,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Get route by protocol and chain pair
   */
  getRouteByProtocol(
    protocol: string,
    sourceChain: number,
    destinationChain: number
  ): RouteOption | undefined {
    const routeKey = `${protocol}-${sourceChain}-${destinationChain}`;
    return this.availableRoutes.get(routeKey);
  }

  /**
   * Check if route is currently available
   */
  async isRouteAvailable(routeId: string): Promise<boolean> {
    const route = Array.from(this.availableRoutes.values())
      .find(r => r.id === routeId);

    if (!route) {
      return false;
    }

    // Check route health (simplified - in production, ping actual services)
    try {
      const healthCheck = await this.performRouteHealthCheck(route);
      return healthCheck.isHealthy;
    } catch (error) {
      this.logger.warn(`Route health check failed for ${routeId}`, { error });
      return false;
    }
  }

  /**
   * Get route statistics
   */
  async getRouteStatistics(routeId: string): Promise<{
    successRate: number;
    averageTime: number;
    averageCost: bigint;
    lastUsed: number;
  } | null> {
    // In production, this would query metrics database
    const route = Array.from(this.availableRoutes.values())
      .find(r => r.id === routeId);

    if (!route) {
      return null;
    }

    // Simplified statistics based on route reliability
    return {
      successRate: route.reliability,
      averageTime: route.estimatedTime,
      averageCost: BigInt(route.gasLimit) * BigInt(20000000000), // Estimated cost
      lastUsed: Date.now() - Math.random() * 3600000 // Random recent usage
    };
  }

  /**
   * Update route configuration
   */
  updateRoute(routeId: string, updates: Partial<RouteOption>): boolean {
    const existingRoute = Array.from(this.availableRoutes.values())
      .find(r => r.id === routeId);

    if (!existingRoute) {
      return false;
    }

    const updatedRoute: RouteOption = { ...existingRoute, ...updates };
    const routeKey = this.getRouteKey(updatedRoute);
    this.availableRoutes.set(routeKey, updatedRoute);

    this.logger.info(`Route updated: ${routeId}`, { updates });
    return true;
  }

  /**
   * Add custom route
   */
  addCustomRoute(route: RouteOption): void {
    const routeKey = this.getRouteKey(route);
    this.availableRoutes.set(routeKey, route);
    
    this.logger.info(`Custom route added: ${route.id}`, { route });
  }

  /**
   * Remove route
   */
  removeRoute(routeId: string): boolean {
    for (const [key, route] of this.availableRoutes.entries()) {
      if (route.id === routeId) {
        this.availableRoutes.delete(key);
        this.logger.info(`Route removed: ${routeId}`);
        return true;
      }
    }
    return false;
  }

  /**
   * Get all supported chain pairs
   */
  getSupportedChainPairs(): { sourceChain: number; destinationChain: number }[] {
    const pairs = new Set<string>();
    
    for (const route of this.availableRoutes.values()) {
      pairs.add(`${route.sourceChain}-${route.destinationChain}`);
    }

    return Array.from(pairs).map(pair => {
      const [sourceChain, destinationChain] = pair.split('-').map(Number);
      return { sourceChain, destinationChain };
    });
  }

  /**
   * Initialize available routes
   */
  private initializeRoutes(): void {
    for (const route of this.SUPPORTED_ROUTES) {
      const routeKey = this.getRouteKey(route);
      this.availableRoutes.set(routeKey, route);
    }

    this.logger.info(`Initialized ${this.availableRoutes.size} routes`);
  }

  /**
   * Get available routes for chain pair and token
   */
  private getAvailableRoutes(
    sourceChain: number,
    destinationChain: number,
    token: string
  ): RouteOption[] {
    return Array.from(this.availableRoutes.values()).filter(route => 
      route.sourceChain === sourceChain &&
      route.destinationChain === destinationChain &&
      route.supportedTokens.includes(token)
    );
  }

  /**
   * Validate cross-chain request
   */
  private validateRequest(request: CrossChainRequest): void {
    if (request.sourceChain === request.destinationChain) {
      throw new Error('Source and destination chains cannot be the same');
    }

    if (request.amount <= 0n) {
      throw new Error('Amount must be greater than 0');
    }

    if (request.maxSlippage < 0 || request.maxSlippage > 10000) {
      throw new Error('Invalid slippage value');
    }

    if (request.deadline && request.deadline <= Date.now() / 1000) {
      throw new Error('Deadline must be in the future');
    }

    if (!['cost', 'speed', 'reliability'].includes(request.priorityMode)) {
      throw new Error('Invalid priority mode');
    }
  }

  /**
   * Generate warnings for route selection
   */
  private generateWarnings(
    request: CrossChainRequest,
    selectedRoute: RouteOption,
    optimization: OptimizationResult
  ): string[] {
    const warnings: string[] = [];

    // High cost warning
    if (optimization.primaryRoute.cost.totalCost > request.amount / 10n) {
      warnings.push('Transaction cost is more than 10% of transfer amount');
    }

    // Reliability warning
    if (selectedRoute.reliability < 95) {
      warnings.push(`Route reliability is ${selectedRoute.reliability}% - consider backup options`);
    }

    // Time warning
    if (request.deadline) {
      const deadlineMinutes = (request.deadline - Date.now() / 1000) / 60;
      if (selectedRoute.estimatedTime > deadlineMinutes) {
        warnings.push('Selected route may not meet deadline requirement');
      }
    }

    // Protocol-specific warnings
    if (selectedRoute.protocol === 'layerzero' && request.amount > 1000000n) {
      warnings.push('Large amounts may experience delays on LayerZero');
    }

    if (selectedRoute.protocol === 'polygon' && selectedRoute.estimatedTime > 20) {
      warnings.push('Polygon bridge may require manual checkpoint submission');
    }

    return warnings;
  }

  /**
   * Calculate confidence score for route selection
   */
  private calculateConfidenceScore(
    route: RouteOption,
    optimization: OptimizationResult
  ): number {
    let score = route.reliability;

    // Boost score if this is significantly better than alternatives
    if (optimization.backupRoutes.length > 0) {
      const bestBackup = optimization.backupRoutes[0];
      const costImprovement = Number(bestBackup.cost.totalCost - optimization.primaryRoute.cost.totalCost) / Number(bestBackup.cost.totalCost);
      
      if (costImprovement > 0.1) { // 10% better
        score = Math.min(100, score + 5);
      }
    }

    // Reduce score for experimental protocols
    if (route.protocol === 'layerzero') {
      score = Math.max(0, score - 2);
    }

    return Math.round(score);
  }

  /**
   * Perform health check for route
   */
  private async performRouteHealthCheck(route: RouteOption): Promise<{ isHealthy: boolean; latency: number }> {
    const startTime = Date.now();
    
    try {
      // Simplified health check - in production, ping actual bridge/protocol endpoints
      await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
      
      const latency = Date.now() - startTime;
      const isHealthy = latency < 500 && Math.random() > 0.05; // 95% uptime simulation
      
      return { isHealthy, latency };
    } catch (error) {
      return { isHealthy: false, latency: Date.now() - startTime };
    }
  }

  /**
   * Generate route key for mapping
   */
  private getRouteKey(route: RouteOption): string {
    return `${route.protocol}-${route.sourceChain}-${route.destinationChain}`;
  }

  /**
   * Get route performance metrics
   */
  async getRoutePerformanceMetrics(): Promise<{
    totalRoutes: number;
    healthyRoutes: number;
    averageReliability: number;
    protocolDistribution: { [protocol: string]: number };
  }> {
    const routes = Array.from(this.availableRoutes.values());
    
    const healthChecks = await Promise.all(
      routes.map(async route => ({
        route,
        health: await this.performRouteHealthCheck(route)
      }))
    );

    const healthyRoutes = healthChecks.filter(check => check.health.isHealthy).length;
    const averageReliability = routes.reduce((sum, route) => sum + route.reliability, 0) / routes.length;
    
    const protocolDistribution: { [protocol: string]: number } = {};
    routes.forEach(route => {
      protocolDistribution[route.protocol] = (protocolDistribution[route.protocol] || 0) + 1;
    });

    return {
      totalRoutes: routes.length,
      healthyRoutes,
      averageReliability: Math.round(averageReliability),
      protocolDistribution
    };
  }
}
