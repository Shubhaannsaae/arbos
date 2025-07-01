import { BigNumber } from 'ethers';

export interface AgentContext {
  agentId: string;
  agentType: 'arbitrage' | 'portfolio' | 'yield' | 'security' | 'orchestrator';
  userId: string;
  sessionId: string;
  networkIds: number[];
  timestamp: number;
  gasPrice: BigNumber;
  nonce: number;
}

export interface AgentMemory {
  shortTerm: Record<string, any>;
  longTerm: Record<string, any>;
  episodic: AgentEpisode[];
  semantic: Record<string, any>;
}

export interface AgentEpisode {
  id: string;
  timestamp: number;
  action: string;
  context: Record<string, any>;
  result: Record<string, any>;
  success: boolean;
  lessons: string[];
}

export interface AgentDecision {
  action: string;
  confidence: number;
  reasoning: string;
  parameters: Record<string, any>;
  riskScore: number;
  expectedOutcome: any;
  alternatives: Array<{
    action: string;
    probability: number;
    outcome: any;
  }>;
}

export interface AgentExecution {
  id: string;
  agentId: string;
  decision: AgentDecision;
  startTime: number;
  endTime?: number;
  status: 'pending' | 'executing' | 'completed' | 'failed' | 'cancelled';
  transactions: string[];
  gasUsed: BigNumber;
  actualOutcome: any;
  errors: string[];
}

export interface AgentPerformance {
  agentId: string;
  period: {
    start: number;
    end: number;
  };
  metrics: {
    totalExecutions: number;
    successfulExecutions: number;
    failedExecutions: number;
    successRate: number;
    averageExecutionTime: number;
    totalGasUsed: BigNumber;
    averageGasUsed: BigNumber;
    totalProfit: BigNumber;
    totalLoss: BigNumber;
    netProfit: BigNumber;
    profitFactor: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
  };
  chainlinkUsage: {
    dataFeedCalls: number;
    automationExecutions: number;
    functionsRequests: number;
    vrfRequests: number;
    ccipMessages: number;
    totalCost: BigNumber;
  };
}

export interface AgentConfiguration {
  enabled: boolean;
  executionInterval: number;
  maxConcurrentExecutions: number;
  riskTolerance: 'low' | 'medium' | 'high';
  constraints: {
    maxDailyTransactions: number;
    maxDailyVolume: BigNumber;
    maxPositionSize: BigNumber;
    minProfitThreshold: number;
    maxSlippage: number;
    maxGasPrice: BigNumber;
    allowedTokens: string[];
    blockedTokens: string[];
    allowedProtocols: string[];
    blockedProtocols: string[];
  };
  notifications: {
    successNotifications: boolean;
    errorNotifications: boolean;
    warningNotifications: boolean;
    webhookUrl?: string;
    emailRecipients: string[];
    slackChannel?: string;
  };
  chainlinkServices: {
    dataFeeds: {
      enabled: boolean;
      subscribedFeeds: string[];
      updateFrequency: number;
    };
    automation: {
      enabled: boolean;
      upkeepIds: string[];
      gasLimit: BigNumber;
    };
    functions: {
      enabled: boolean;
      subscriptionId: string;
      gasLimit: BigNumber;
      secrets: Record<string, string>;
    };
    vrf: {
      enabled: boolean;
      subscriptionId: string;
      keyHash: string;
      requestConfirmations: number;
      callbackGasLimit: BigNumber;
    };
    ccip: {
      enabled: boolean;
      supportedChains: number[];
      maxGasLimit: BigNumber;
      maxDataLength: number;
    };
  };
}

export interface AgentState {
  agentId: string;
  status: 'idle' | 'analyzing' | 'executing' | 'paused' | 'error' | 'maintenance';
  currentTask?: string;
  lastExecution?: number;
  nextExecution?: number;
  healthScore: number;
  errorCount: number;
  warningCount: number;
  memory: AgentMemory;
  configuration: AgentConfiguration;
  performance: AgentPerformance;
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    apiCallsRemaining: number;
    gasAllowanceRemaining: BigNumber;
  };
}

export interface AgentMessage {
  id: string;
  fromAgent: string;
  toAgent: string;
  type: 'request' | 'response' | 'notification' | 'alert';
  priority: 'low' | 'medium' | 'high' | 'critical';
  content: any;
  timestamp: number;
  expiresAt?: number;
  acknowledged: boolean;
}

export interface AgentCoordination {
  coordinatorId: string;
  participants: string[];
  objective: string;
  strategy: string;
  resources: {
    totalGasAllowance: BigNumber;
    timeLimit: number;
    apiQuotas: Record<string, number>;
  };
  status: 'planning' | 'executing' | 'completed' | 'failed';
  results: Record<string, any>;
  conflicts: Array<{
    agents: string[];
    issue: string;
    resolution?: string;
  }>;
}

export interface AgentEvent {
  id: string;
  agentId: string;
  eventType: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  data: Record<string, any>;
  timestamp: number;
  resolved: boolean;
  resolutionTime?: number;
  tags: string[];
}

export interface AgentCapability {
  name: string;
  version: string;
  description: string;
  inputSchema: any;
  outputSchema: any;
  dependencies: string[];
  constraints: Record<string, any>;
  performance: {
    averageExecutionTime: number;
    successRate: number;
    resourceUsage: Record<string, number>;
  };
}

export interface AgentPlugin {
  id: string;
  name: string;
  version: string;
  capabilities: AgentCapability[];
  configuration: Record<string, any>;
  enabled: boolean;
  lastUpdated: number;
}
