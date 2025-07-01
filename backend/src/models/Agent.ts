export interface Agent {
  id: string;
  userId: string;
  name: string;
  type: AgentType;
  status: AgentStatus;
  configuration: AgentConfiguration;
  permissions: AgentPermissions;
  performance: AgentPerformance;
  resources: AgentResources;
  createdAt: Date;
  updatedAt: Date;
  lastActiveAt: Date;
  isEnabled: boolean;
  version: string;
}

export interface AgentConfiguration {
  model: AIModel;
  parameters: AgentParameters;
  constraints: AgentConstraints;
  triggers: AgentTrigger[];
  tools: AgentTool[];
  notifications: AgentNotifications;
}

export interface AgentParameters {
  riskTolerance: number;
  maxPositionSize: number;
  slippageTolerance: number;
  gasOptimization: boolean;
  frequencyMs: number;
  timeoutMs: number;
  retryAttempts: number;
  confidenceThreshold: number;
  profitThreshold: number;
  stopLossThreshold: number;
}

export interface AgentConstraints {
  maxDailyTransactions: number;
  maxDailyVolume: number;
  allowedTokens?: string[];
  blockedTokens?: string[];
  allowedProtocols?: string[];
  blockedProtocols?: string[];
  allowedChains: number[];
  tradingHours?: TradingHours;
}

export interface TradingHours {
  enabled: boolean;
  timezone: string;
  startTime: string;
  endTime: string;
  weekends: boolean;
}

export interface AgentTrigger {
  type: TriggerType;
  condition: TriggerCondition;
  parameters: Record<string, any>;
  enabled: boolean;
}

export interface TriggerCondition {
  field: string;
  operator: ComparisonOperator;
  value: any;
  tolerance?: number;
}

export interface AgentTool {
  name: string;
  type: ToolType;
  enabled: boolean;
  configuration: Record<string, any>;
  permissions: string[];
}

export interface AgentNotifications {
  onError: boolean;
  onSuccess: boolean;
  onThreshold: boolean;
  channels: NotificationChannel[];
  webhook?: string;
}

export interface AgentPermissions {
  canTrade: boolean;
  canRebalance: boolean;
  canWithdraw: boolean;
  canApprove: boolean;
  canCrossChain: boolean;
  maxTransactionValue: number;
  allowedContracts?: string[];
  emergencyStop: boolean;
}

export interface AgentPerformance {
  totalTransactions: number;
  successfulTransactions: number;
  failedTransactions: number;
  totalVolume: number;
  totalProfit: number;
  totalLoss: number;
  winRate: number;
  averageProfit: number;
  averageLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  sharpeRatio: number;
  uptime: number;
  lastUpdate: Date;
}

export interface AgentResources {
  cpuUsage: number;
  memoryUsage: number;
  networkRequests: number;
  apiCallsRemaining: number;
  gasUsed: string;
  estimatedMonthlyCost: number;
}

export enum AgentType {
  ARBITRAGE = 'arbitrage',
  PORTFOLIO = 'portfolio',
  YIELD = 'yield',
  SECURITY = 'security',
  ORCHESTRATOR = 'orchestrator',
  CUSTOM = 'custom'
}

export enum AgentStatus {
  INITIALIZING = 'initializing',
  ACTIVE = 'active',
  IDLE = 'idle',
  PAUSED = 'paused',
  STOPPED = 'stopped',
  ERROR = 'error',
  MAINTENANCE = 'maintenance'
}

export enum AIModel {
  GPT_4 = 'gpt-4',
  GPT_4_TURBO = 'gpt-4-turbo',
  CLAUDE_3 = 'claude-3',
  GEMINI_PRO = 'gemini-pro',
  CUSTOM = 'custom'
}

export enum TriggerType {
  PRICE_CHANGE = 'price_change',
  VOLUME_SPIKE = 'volume_spike',
  ARBITRAGE_OPPORTUNITY = 'arbitrage_opportunity',
  PORTFOLIO_DRIFT = 'portfolio_drift',
  MARKET_VOLATILITY = 'market_volatility',
  TIME_BASED = 'time_based',
  CUSTOM = 'custom'
}

export enum ComparisonOperator {
  GREATER_THAN = 'gt',
  LESS_THAN = 'lt',
  GREATER_THAN_OR_EQUAL = 'gte',
  LESS_THAN_OR_EQUAL = 'lte',
  EQUAL = 'eq',
  NOT_EQUAL = 'ne',
  CONTAINS = 'contains',
  IN = 'in',
  NOT_IN = 'not_in'
}

export enum ToolType {
  CHAINLINK_FEED = 'chainlink_feed',
  CHAINLINK_AUTOMATION = 'chainlink_automation',
  CHAINLINK_FUNCTIONS = 'chainlink_functions',
  CHAINLINK_VRF = 'chainlink_vrf',
  CHAINLINK_CCIP = 'chainlink_ccip',
  DEX_AGGREGATOR = 'dex_aggregator',
  YIELD_PROTOCOL = 'yield_protocol',
  ML_MODEL = 'ml_model',
  CUSTOM_API = 'custom_api'
}

export enum NotificationChannel {
  EMAIL = 'email',
  SMS = 'sms',
  PUSH = 'push',
  WEBHOOK = 'webhook',
  SLACK = 'slack',
  DISCORD = 'discord'
}

export interface CreateAgentDto {
  name: string;
  type: AgentType;
  configuration: AgentConfiguration;
  permissions: AgentPermissions;
}

export interface UpdateAgentDto {
  name?: string;
  configuration?: Partial<AgentConfiguration>;
  permissions?: Partial<AgentPermissions>;
  isEnabled?: boolean;
}
