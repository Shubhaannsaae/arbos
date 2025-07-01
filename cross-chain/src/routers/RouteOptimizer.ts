import { Logger } from 'winston';
import { createLogger, format, transports } from 'winston';
import { RouteOption } from './OptimalRouter';
import { RouteCost } from './CostCalculator';

export interface RouteWithCost {
  route: RouteOption;
  cost: RouteCost;
}

export interface OptimizationCriteria {
  maxSlippage: number;
  deadline?: number;
  amount: bigint;
  priorityWeights?: PriorityWeights;
}

export interface PriorityWeights {
  cost: number;
  speed: number;
  reliability: number;
}

export interface OptimizationResult {
  primaryRoute: RouteWithCost;
  backupRoutes: RouteWithCost[];
  optimizationScore: number;
  decisionFactors: DecisionFactors;
}

export interface DecisionFactors {
  costScore: number;
  speedScore: number;
  reliabilityScore: number;
  overallScore: number;
  reasoning: string[];
}

export class RouteOptimizer {
  private logger: Logger;

  // Default priority weights for different modes
  private readonly DEFAULT_WEIGHTS: { [mode: string]: PriorityWeights } = {
    cost: { cost: 0.7, speed: 0.2, reliability: 0.1 },
    speed: { cost: 0.2, speed: 0.7, reliability: 0.1 },
    reliability: { cost: 0.1, speed: 0.2, reliability: 0.7 }
  };

  constructor() {
    this.logger = createLogger({
      level: 'info',
      format: format.combine(
        format.timestamp(),
        format.errors({ stack: true }),
        format.json()
      ),
      transports: [
        new transports.File({ filename: 'logs/route-optimizer.log' }),
        new transports.Console({ format: format.simple() })
      ]
    });
  }

  /**
   * Optimize route selection based on criteria
   */
  async optimizeRoute(
    routesWithCosts: RouteWithCost[],
    priorityMode: 'cost' | 'speed' | 'reliability',
    criteria: OptimizationCriteria
  ): Promise<OptimizationResult> {
    try {
      if (routesWithCosts.length === 0) {
        throw new Error('No routes provided for optimization');
      }

      // Get priority weights
      const weights = criteria.priorityWeights || this.DEFAULT_WEIGHTS[priorityMode];

      // Filter routes that meet basic criteria
      const validRoutes = this.filterValidRoutes(routesWithCosts, criteria);
      
      if (validRoutes.length === 0) {
        throw new Error('No routes meet the specified criteria');
      }

      // Calculate optimization scores for each route
      const scoredRoutes = await this.calculateOptimizationScores(validRoutes, weights, criteria);

      // Sort by optimization score (descending)
      scoredRoutes.sort((a, b) => b.score - a.score);

      // Select primary route and backups
      const primaryRoute = scoredRoutes[0];
      const backupRoutes = scoredRoutes.slice(1, 4); // Top 3 backups

      // Generate decision factors
      const decisionFactors = this.generateDecisionFactors(
        primaryRoute,
        scoredRoutes,
        weights,
        priorityMode
      );

      const result: OptimizationResult = {
        primaryRoute: primaryRoute.routeWithCost,
        backupRoutes: backupRoutes.map(r => r.routeWithCost),
        optimizationScore: primaryRoute.score,
        decisionFactors
      };

      this.logger.info(`Route optimization completed`, {
        primaryRouteId: primaryRoute.routeWithCost.route.id,
        optimizationScore: primaryRoute.score,
        backupCount: backupRoutes.length,
        priorityMode
      });

      return result;

    } catch (error) {
      this.logger.error('Route optimization failed', {
        routeCount: routesWithCosts.length,
        priorityMode,
        criteria,
        error: error instanceof Error ? error.message : String(error)
      });
      throw error;
    }
  }

  /**
   * Analyze route performance across different scenarios
   */
  async analyzeRoutePerformance(
    routes: RouteWithCost[],
    scenarios: { priorityMode: string; criteria: OptimizationCriteria }[]
  ): Promise<{
    route: RouteOption;
    performanceMetrics: {
      scenario: string;
      rank: number;
      score: number;
      selected: boolean;
    }[];
    overallRank: number;
  }[]> {
    const routePerformance = new Map<string, any[]>();

    // Initialize performance tracking for each route
    for (const routeWithCost of routes) {
      routePerformance.set(routeWithCost.route.id, []);
    }

    // Analyze each scenario
    for (const scenario of scenarios) {
      const optimization = await this.optimizeRoute(
        routes,
        scenario.priorityMode as any,
        scenario.criteria
      );

      // Rank all routes for this scenario
      const weights = this.DEFAULT_WEIGHTS[scenario.priorityMode];
      const scoredRoutes = await this.calculateOptimizationScores(routes, weights, scenario.criteria);
      scoredRoutes.sort((a, b) => b.score - a.score);

      // Record performance for each route
      scoredRoutes.forEach((scoredRoute, index) => {
        const performance = routePerformance.get(scoredRoute.routeWithCost.route.id)!;
        performance.push({
          scenario: scenario.priorityMode,
          rank: index + 1,
          score: scoredRoute.score,
          selected: index === 0
        });
      });
    }

    // Calculate overall rankings
    const results = Array.from(routePerformance.entries()).map(([routeId, metrics]) => {
      const route = routes.find(r => r.route.id === routeId)!.route;
      const averageRank = metrics.reduce((sum: number, m: any) => sum + m.rank, 0) / metrics.length;
      
      return {
        route,
        performanceMetrics: metrics,
        overallRank: averageRank
      };
    });

    // Sort by overall rank
    results.sort((a, b) => a.overallRank - b.overallRank);

    return results;
  }

  /**
   * Simulate route optimization under different market conditions
   */
  async simulateMarketConditions(
    baseRoutes: RouteWithCost[],
    marketScenarios: {
      name: string;
      gasPriceMultiplier: number;
      congestionFactor: number;
      reliabilityImpact: number;
    }[]
  ): Promise<{
    scenario: string;
    optimalRoutes: { routeId: string; score: number }[];
    marketImpact: {
      costIncrease: number;
      timeIncrease: number;
      reliabilityDecrease: number;
    };
  }[]> {
    const results = [];

    for (const scenario of marketScenarios) {
      // Adjust routes based on market conditions
      const adjustedRoutes = this.adjustRoutesForMarketConditions(baseRoutes, scenario);

      // Optimize with standard balanced weights
      const weights = { cost: 0.4, speed: 0.3, reliability: 0.3 };
      const criteria: OptimizationCriteria = {
        maxSlippage: 500, // 5%
        amount: ethers.parseEther('1000') // 1000 ETH equivalent
      };

      const scoredRoutes = await this.calculateOptimizationScores(adjustedRoutes, weights, criteria);
      scoredRoutes.sort((a, b) => b.score - a.score);

      // Calculate market impact
      const baseOptimal = baseRoutes[0];
      const adjustedOptimal = adjustedRoutes[0];
      
      const marketImpact = {
        costIncrease: Number(adjustedOptimal.cost.totalCost - baseOptimal.cost.totalCost) / Number(baseOptimal.cost.totalCost),
        timeIncrease: (adjustedOptimal.route.estimatedTime - baseOptimal.route.estimatedTime) / baseOptimal.route.estimatedTime,
        reliabilityDecrease: (baseOptimal.route.reliability - adjustedOptimal.route.reliability) / baseOptimal.route.reliability
      };

      results.push({
        scenario: scenario.name,
        optimalRoutes: scoredRoutes.slice(0, 3).map(r => ({
          routeId: r.routeWithCost.route.id,
          score: r.score
        })),
        marketImpact
      });
    }

    return results;
  }

  /**
   * Filter routes that meet basic criteria
   */
  private filterValidRoutes(
    routesWithCosts: RouteWithCost[],
    criteria: OptimizationCriteria
  ): RouteWithCost[] {
    return routesWithCosts.filter(routeWithCost => {
      const route = routeWithCost.route;
      
      // Check deadline constraint
      if (criteria.deadline) {
        const deadlineMinutes = (criteria.deadline - Date.now() / 1000) / 60;
        if (route.estimatedTime > deadlineMinutes) {
          return false;
        }
      }

      // Check if route supports the token amount (basic liquidity check)
      const maxSupportedAmount = this.getMaxSupportedAmount(route);
      if (criteria.amount > maxSupportedAmount) {
        return false;
      }

      return true;
    });
  }

  /**
   * Calculate optimization scores for routes
   */
  private async calculateOptimizationScores(
    routes: RouteWithCost[],
    weights: PriorityWeights,
    criteria: OptimizationCriteria
  ): Promise<{ routeWithCost: RouteWithCost; score: number }[]> {
    // Normalize factors for scoring
    const maxCost = Math.max(...routes.map(r => Number(r.cost.totalCost)));
    const minCost = Math.min(...routes.map(r => Number(r.cost.totalCost)));
    const maxTime = Math.max(...routes.map(r => r.route.estimatedTime));
    const minTime = Math.min(...routes.map(r => r.route.estimatedTime));
    const maxReliability = Math.max(...routes.map(r => r.route.reliability));
    const minReliability = Math.min(...routes.map(r => r.route.reliability));

    return routes.map(routeWithCost => {
      const route = routeWithCost.route;
      const cost = routeWithCost.cost;

      // Calculate normalized scores (0-100)
      const costScore = maxCost > minCost 
        ? 100 - ((Number(cost.totalCost) - minCost) / (maxCost - minCost)) * 100
        : 100;

      const speedScore = maxTime > minTime
        ? 100 - ((route.estimatedTime - minTime) / (maxTime - minTime)) * 100
        : 100;

      const reliabilityScore = maxReliability > minReliability
        ? ((route.reliability - minReliability) / (maxReliability - minReliability)) * 100
        : 100;

      // Apply protocol-specific adjustments
      const protocolBonus = this.getProtocolBonus(route.protocol);
      
      // Calculate weighted score
      const score = (
        (costScore * weights.cost) +
        (speedScore * weights.speed) +
        (reliabilityScore * weights.reliability)
      ) + protocolBonus;

      return {
        routeWithCost,
        score: Math.max(0, Math.min(100, score))
      };
    });
  }

  /**
   * Generate decision factors explanation
   */
  private generateDecisionFactors(
    primaryRoute: { routeWithCost: RouteWithCost; score: number },
    allRoutes: { routeWithCost: RouteWithCost; score: number }[],
    weights: PriorityWeights,
    priorityMode: string
  ): DecisionFactors {
    const route = primaryRoute.routeWithCost.route;
    const cost = primaryRoute.routeWithCost.cost;
    
    // Calculate individual factor scores
    const costScore = this.calculateIndividualScore(allRoutes, 'cost');
    const speedScore = this.calculateIndividualScore(allRoutes, 'speed');
    const reliabilityScore = this.calculateIndividualScore(allRoutes, 'reliability');

    // Generate reasoning
    const reasoning: string[] = [];
    
    if (weights.cost > 0.5) {
      reasoning.push(`Cost optimization prioritized (${(weights.cost * 100).toFixed(0)}% weight)`);
      reasoning.push(`Selected route costs ${ethers.formatEther(cost.totalCost)} ETH ($${cost.costInUSD.toFixed(2)})`);
    }
    
    if (weights.speed > 0.5) {
      reasoning.push(`Speed optimization prioritized (${(weights.speed * 100).toFixed(0)}% weight)`);
      reasoning.push(`Selected route completes in ~${route.estimatedTime} minutes`);
    }
    
    if (weights.reliability > 0.5) {
      reasoning.push(`Reliability optimization prioritized (${(weights.reliability * 100).toFixed(0)}% weight)`);
      reasoning.push(`Selected route has ${route.reliability}% reliability rating`);
    }

    reasoning.push(`Protocol: ${route.protocol.toUpperCase()}`);
    
    if (allRoutes.length > 1) {
      const scoreDiff = primaryRoute.score - allRoutes[1].score;
      reasoning.push(`Selected route scored ${scoreDiff.toFixed(1)} points higher than next best option`);
    }

    return {
      costScore,
      speedScore,
      reliabilityScore,
      overallScore: primaryRoute.score,
      reasoning
    };
  }

  /**
   * Calculate individual factor score
   */
  private calculateIndividualScore(
    routes: { routeWithCost: RouteWithCost; score: number }[],
    factor: 'cost' | 'speed' | 'reliability'
  ): number {
    // Simplified individual scoring
    const primaryRoute = routes[0].routeWithCost;
    
    switch (factor) {
      case 'cost':
        const costRank = routes
          .sort((a, b) => Number(a.routeWithCost.cost.totalCost - b.routeWithCost.cost.totalCost))
          .findIndex(r => r.routeWithCost.route.id === primaryRoute.route.id) + 1;
        return Math.max(0, 100 - (costRank - 1) * 20);
        
      case 'speed':
        const speedRank = routes
          .sort((a, b) => a.routeWithCost.route.estimatedTime - b.routeWithCost.route.estimatedTime)
          .findIndex(r => r.routeWithCost.route.id === primaryRoute.route.id) + 1;
        return Math.max(0, 100 - (speedRank - 1) * 20);
        
      case 'reliability':
        const reliabilityRank = routes
          .sort((a, b) => b.routeWithCost.route.reliability - a.routeWithCost.route.reliability)
          .findIndex(r => r.routeWithCost.route.id === primaryRoute.route.id) + 1;
        return Math.max(0, 100 - (reliabilityRank - 1) * 20);
        
      default:
        return 50;
    }
  }

  /**
   * Get protocol-specific bonus points
   */
  private getProtocolBonus(protocol: string): number {
    const bonuses: { [protocol: string]: number } = {
      ccip: 5,        // CCIP gets bonus for being official Chainlink standard
      layerzero: 2,   // LayerZero gets small bonus for speed
      polygon: 3,     // Polygon gets bonus for low cost
      arbitrum: 2,    // Arbitrum gets small bonus
      avalanche: 1    // Avalanche gets minimal bonus
    };

    return bonuses[protocol] || 0;
  }

  /**
   * Get maximum supported amount for route
   */
  private getMaxSupportedAmount(route: RouteOption): bigint {
    // Protocol-specific limits
    const limits: { [protocol: string]: bigint } = {
      ccip: ethers.parseEther('1000000'),      // 1M ETH
      layerzero: ethers.parseEther('500000'),  // 500K ETH
      polygon: ethers.parseEther('100000'),    // 100K ETH
      arbitrum: ethers.parseEther('1000000'),  // 1M ETH
      avalanche: ethers.parseEther('200000')   // 200K ETH
    };

    return limits[route.protocol] || ethers.parseEther('10000');
  }

  /**
   * Adjust routes for market conditions
   */
  private adjustRoutesForMarketConditions(
    routes: RouteWithCost[],
    scenario: {
      gasPriceMultiplier: number;
      congestionFactor: number;
      reliabilityImpact: number;
    }
  ): RouteWithCost[] {
    return routes.map(routeWithCost => {
      const adjustedCost: RouteCost = {
        ...routeWithCost.cost,
        gasFee: routeWithCost.cost.gasFee * BigInt(Math.floor(scenario.gasPriceMultiplier * 100)) / 100n,
        totalCost: routeWithCost.cost.totalCost * BigInt(Math.floor(scenario.gasPriceMultiplier * 100)) / 100n
      };

      const adjustedRoute: RouteOption = {
        ...routeWithCost.route,
        estimatedTime: Math.floor(routeWithCost.route.estimatedTime * scenario.congestionFactor),
        reliability: Math.max(50, Math.floor(routeWithCost.route.reliability * (1 - scenario.reliabilityImpact)))
      };

      return {
        route: adjustedRoute,
        cost: adjustedCost
      };
    });
  }
}
