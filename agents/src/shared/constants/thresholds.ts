export const RISK_THRESHOLDS = {
  // Risk scores (0-100)
  LOW_RISK: 25,
  MEDIUM_RISK: 50,
  HIGH_RISK: 75,
  CRITICAL_RISK: 90,

  // Confidence levels (0-1)
  MIN_CONFIDENCE: 0.7,
  HIGH_CONFIDENCE: 0.85,
  VERY_HIGH_CONFIDENCE: 0.95,

  // Volatility thresholds
  LOW_VOLATILITY: 0.1,    // 10%
  MEDIUM_VOLATILITY: 0.3, // 30%
  HIGH_VOLATILITY: 0.5,   // 50%

  // Liquidity thresholds (USD)
  MIN_LIQUIDITY: 100000,      // $100k
  GOOD_LIQUIDITY: 1000000,    // $1M
  EXCELLENT_LIQUIDITY: 10000000, // $10M

  // Price impact thresholds for different trade sizes
  PRICE_IMPACT_1K: 0.1,   // 0.1% for $1k trade
  PRICE_IMPACT_10K: 0.5,  // 0.5% for $10k trade
  PRICE_IMPACT_100K: 2.0, // 2% for $100k trade

  // Slippage tolerance
  MIN_SLIPPAGE: 0.1,      // 0.1%
  MAX_SLIPPAGE: 5.0,      // 5%
  DEFAULT_SLIPPAGE: 1.0,  // 1%

  // Profit thresholds
  MIN_PROFIT_PERCENTAGE: 0.1,    // 0.1%
  GOOD_PROFIT_PERCENTAGE: 1.0,   // 1%
  EXCELLENT_PROFIT_PERCENTAGE: 5.0, // 5%

  // Gas price thresholds (Gwei)
  LOW_GAS: 20,
  MEDIUM_GAS: 50,
  HIGH_GAS: 100,
  EXTREME_GAS: 200
};

export const ARBITRAGE_THRESHOLDS = {
  // Minimum profit thresholds
  MIN_PROFIT_USD: 10,           // $10 minimum profit
  MIN_PROFIT_PERCENTAGE: 0.5,   // 0.5% minimum profit margin
  
  // Maximum position sizes
  MAX_POSITION_USD: 100000,     // $100k maximum position
  MAX_POSITION_PERCENTAGE: 10,  // 10% of available liquidity
  
  // Time constraints
  MAX_EXECUTION_TIME: 300,      // 5 minutes maximum execution time
  OPPORTUNITY_EXPIRY: 60,       // Opportunities expire after 1 minute
  
  // Cross-chain specific
  MIN_BRIDGE_AMOUNT: 1000,      // $1k minimum for cross-chain
  MAX_BRIDGE_TIME: 3600,        // 1 hour maximum bridge time
  BRIDGE_SLIPPAGE_BUFFER: 0.5,  // 0.5% extra slippage for bridges
  
  // Risk management
  MAX_CONCURRENT_ARBITRAGES: 3,
  MAX_DAILY_ARBITRAGES: 50,
  MAX_FAILED_ATTEMPTS: 5,
  
  // Chainlink data requirements
  MAX_PRICE_AGE: 300,           // 5 minutes maximum price age
  MIN_PRICE_CONFIDENCE: 0.95,   // 95% minimum confidence
  REQUIRED_CONFIRMATIONS: 3
};

export const PORTFOLIO_THRESHOLDS = {
  // Rebalancing triggers
  REBALANCE_THRESHOLD: 5.0,     // 5% deviation triggers rebalance
  MIN_REBALANCE_AMOUNT: 100,    // $100 minimum rebalance
  MAX_REBALANCE_FREQUENCY: 4,   // Maximum 4 rebalances per day
  
  // Position limits
  MAX_SINGLE_POSITION: 50,      // 50% maximum single position
  MIN_POSITION_SIZE: 1,         // 1% minimum position
  MAX_POSITIONS: 20,            // Maximum 20 positions
  
  // Risk management
  MAX_PORTFOLIO_VOLATILITY: 0.4, // 40% maximum volatility
  MIN_SHARPE_RATIO: 0.5,        // Minimum 0.5 Sharpe ratio
  MAX_DRAWDOWN: 20,             // 20% maximum drawdown
  
  // Performance tracking
  BENCHMARK_TRACKING_ERROR: 5,  // 5% maximum tracking error
  MIN_CORRELATION: 0.7,         // 70% minimum correlation to benchmark
  
  // Automation settings
  AUTOMATION_GAS_LIMIT: 2000000,
  AUTOMATION_CHECK_FREQUENCY: 300, // 5 minutes
  
  // Chainlink data requirements
  PRICE_UPDATE_FREQUENCY: 60,   // 1 minute price updates
  MAX_DATA_STALENESS: 900      // 15 minutes maximum staleness
};

export const YIELD_THRESHOLDS = {
  // Yield requirements
  MIN_APY: 1.0,                 // 1% minimum APY
  GOOD_APY: 5.0,               // 5% good APY
  EXCELLENT_APY: 15.0,         // 15% excellent APY
  
  // Risk assessment
  MAX_PROTOCOL_RISK: 70,        // 70% maximum protocol risk score
  MIN_AUDIT_SCORE: 80,          // 80% minimum audit score
  MAX_IMPERMANENT_LOSS: 10,     // 10% maximum impermanent loss risk
  
  // Position management
  MIN_DEPOSIT: 100,             // $100 minimum deposit
  MAX_POOL_CONCENTRATION: 25,   // 25% maximum in single pool
  DIVERSIFICATION_MINIMUM: 3,   // Minimum 3 different protocols
  
  // Performance tracking
  YIELD_TRACKING_PERIOD: 30,    // 30 days yield tracking
  MIN_REALIZED_YIELD: 2.0,      // 2% minimum realized yield
  
  // Emergency thresholds
  EMERGENCY_EXIT_LOSS: 5,       // 5% loss triggers emergency exit
  TVL_DROP_THRESHOLD: 50,       // 50% TVL drop triggers alert
  
  // Chainlink automation
  HARVEST_FREQUENCY: 86400,     // Daily harvest (24 hours)
  COMPOUND_THRESHOLD: 0.1       // 0.1% threshold for auto-compound
};

export const SECURITY_THRESHOLDS = {
  // Anomaly detection
  UNUSUAL_VOLUME_MULTIPLIER: 5,    // 5x normal volume triggers alert
  PRICE_DEVIATION_THRESHOLD: 10,   // 10% price deviation
  GAS_SPIKE_MULTIPLIER: 3,         // 3x normal gas price
  
  // Transaction monitoring
  MAX_TRANSACTION_VALUE: 1000000,  // $1M transaction limit
  SUSPICIOUS_PATTERN_COUNT: 3,     // 3+ suspicious patterns
  RAPID_TRANSACTION_THRESHOLD: 10, // 10+ transactions in 5 minutes
  
  // Wallet security
  MAX_EXPOSURE_PER_PROTOCOL: 30,   // 30% maximum exposure
  WALLET_DIVERSIFICATION_MIN: 5,   // Minimum 5 different positions
  
  // Smart contract risks
  UNVERIFIED_CONTRACT_RISK: 90,    // 90% risk for unverified contracts
  NEW_CONTRACT_RISK: 70,           // 70% risk for contracts < 30 days
  NO_AUDIT_RISK: 80,              // 80% risk for unaudited contracts
  
  // Emergency response
  CRITICAL_ALERT_RESPONSE_TIME: 60,  // 1 minute response for critical alerts
  EMERGENCY_PAUSE_THRESHOLD: 85,     // 85% risk score triggers pause
  
  // Chainlink data validation
  ORACLE_DEVIATION_THRESHOLD: 5,   // 5% deviation between oracles
  DATA_FRESHNESS_REQUIREMENT: 600  // 10 minutes maximum data age
};

export const ORCHESTRATION_THRESHOLDS = {
  // Resource allocation
  MAX_GAS_PER_AGENT: 5000000,     // 5M gas per agent per execution
  MAX_API_CALLS_PER_MINUTE: 100,  // 100 API calls per minute
  MAX_CONCURRENT_EXECUTIONS: 5,   // 5 concurrent executions
  
  // Performance monitoring
  MIN_AGENT_UPTIME: 95,            // 95% minimum uptime
  MAX_RESPONSE_TIME: 30,           // 30 seconds maximum response
  MAX_ERROR_RATE: 5,               // 5% maximum error rate
  
  // Conflict resolution
  PRIORITY_OVERRIDE_THRESHOLD: 90, // 90% confidence to override
  COORDINATION_TIMEOUT: 300,       // 5 minutes coordination timeout
  
  // System health
  MIN_SYSTEM_RESOURCES: 20,        // 20% minimum available resources
  MAX_QUEUE_SIZE: 100,             // 100 maximum queued tasks
  
  // Auto-scaling triggers
  CPU_SCALE_UP_THRESHOLD: 80,      // 80% CPU triggers scale up
  MEMORY_SCALE_UP_THRESHOLD: 85,   // 85% memory triggers scale up
  QUEUE_SCALE_UP_THRESHOLD: 75,    // 75% queue triggers scale up
  
  // Emergency protocols
  SYSTEM_SHUTDOWN_THRESHOLD: 95,   // 95% system stress triggers shutdown
  DATA_INTEGRITY_THRESHOLD: 99     // 99% data integrity required
};

export const CHAINLINK_THRESHOLDS = {
  // Data Feeds
  MAX_PRICE_DEVIATION: 5,          // 5% maximum price deviation
  HEARTBEAT_TOLERANCE: 1.5,        // 1.5x heartbeat tolerance
  MIN_AGGREGATOR_RESPONSES: 3,     // Minimum 3 oracle responses
  
  // CCIP
  MAX_CCIP_GAS_LIMIT: 2000000,     // 2M gas limit for CCIP
  CCIP_CONFIRMATION_BLOCKS: 5,     // 5 blocks for CCIP confirmation
  MAX_CCIP_DATA_SIZE: 10000,       // 10KB maximum data size
  
  // Automation
  MIN_UPKEEP_BALANCE: 1,           // 1 LINK minimum balance
  MAX_UPKEEP_GAS: 5000000,         // 5M gas maximum per upkeep
  UPKEEP_CHECK_FREQUENCY: 30,      // 30 seconds check frequency
  
  // Functions
  MAX_FUNCTION_DURATION: 10,       // 10 seconds maximum execution
  MIN_FUNCTION_BALANCE: 0.1,       // 0.1 LINK minimum balance
  MAX_FUNCTION_RESPONSE_SIZE: 256, // 256 bytes maximum response
  
  // VRF
  MIN_VRF_CONFIRMATIONS: 3,        // 3 minimum confirmations
  MAX_VRF_WORDS: 10,               // 10 maximum random words
  VRF_CALLBACK_GAS_LIMIT: 200000,  // 200k gas for VRF callback
  
  // Cost management
  MAX_DAILY_CHAINLINK_COST: 100,   // $100 maximum daily cost
  COST_OPTIMIZATION_THRESHOLD: 80, // 80% of budget triggers optimization
  
  // Service reliability
  MIN_SERVICE_UPTIME: 99.9,        // 99.9% minimum uptime
  MAX_SERVICE_LATENCY: 5000        // 5 seconds maximum latency
};

// Utility functions for threshold checking
export function checkRiskLevel(score: number): 'low' | 'medium' | 'high' | 'critical' {
  if (score <= RISK_THRESHOLDS.LOW_RISK) return 'low';
  if (score <= RISK_THRESHOLDS.MEDIUM_RISK) return 'medium';
  if (score <= RISK_THRESHOLDS.HIGH_RISK) return 'high';
  return 'critical';
}

export function checkConfidenceLevel(confidence: number): 'low' | 'medium' | 'high' | 'very_high' {
  if (confidence < RISK_THRESHOLDS.MIN_CONFIDENCE) return 'low';
  if (confidence < RISK_THRESHOLDS.HIGH_CONFIDENCE) return 'medium';
  if (confidence < RISK_THRESHOLDS.VERY_HIGH_CONFIDENCE) return 'high';
  return 'very_high';
}

export function checkGasLevel(gasPrice: number): 'low' | 'medium' | 'high' | 'extreme' {
  if (gasPrice <= RISK_THRESHOLDS.LOW_GAS) return 'low';
  if (gasPrice <= RISK_THRESHOLDS.MEDIUM_GAS) return 'medium';
  if (gasPrice <= RISK_THRESHOLDS.HIGH_GAS) return 'high';
  return 'extreme';
}

export function isArbitrageViable(
  profitPercentage: number,
  profitUsd: number,
  riskScore: number,
  confidence: number
): boolean {
  return (
    profitPercentage >= ARBITRAGE_THRESHOLDS.MIN_PROFIT_PERCENTAGE &&
    profitUsd >= ARBITRAGE_THRESHOLDS.MIN_PROFIT_USD &&
    riskScore <= RISK_THRESHOLDS.HIGH_RISK &&
    confidence >= RISK_THRESHOLDS.MIN_CONFIDENCE
  );
}

export function shouldRebalancePortfolio(
  deviationPercentage: number,
  timeSinceLastRebalance: number,
  portfolioValue: number
): boolean {
  return (
    deviationPercentage >= PORTFOLIO_THRESHOLDS.REBALANCE_THRESHOLD &&
    portfolioValue >= PORTFOLIO_THRESHOLDS.MIN_REBALANCE_AMOUNT &&
    timeSinceLastRebalance >= (24 * 60 * 60 / PORTFOLIO_THRESHOLDS.MAX_REBALANCE_FREQUENCY)
  );
}

export function isYieldOpportunityAttractive(
  apy: number,
  riskScore: number,
  auditScore: number | null
): boolean {
  return (
    apy >= YIELD_THRESHOLDS.MIN_APY &&
    riskScore <= YIELD_THRESHOLDS.MAX_PROTOCOL_RISK &&
    (auditScore === null || auditScore >= YIELD_THRESHOLDS.MIN_AUDIT_SCORE)
  );
}

export function isSecurityThreatCritical(
  riskScore: number,
  confidence: number,
  affectedValue: number
): boolean {
  return (
    riskScore >= SECURITY_THRESHOLDS.EMERGENCY_PAUSE_THRESHOLD ||
    (confidence >= RISK_THRESHOLDS.HIGH_CONFIDENCE && 
     affectedValue >= SECURITY_THRESHOLDS.MAX_TRANSACTION_VALUE)
  );
}
