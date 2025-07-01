export interface ModelConfig {
  provider: 'aws-bedrock';
  modelId: string;
  region: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  stopSequences: string[];
  timeout: number;
  retryAttempts: number;
  rateLimits: {
    requestsPerMinute: number;
    tokensPerMinute: number;
  };
}

export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  'claude-3-haiku': {
    provider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
    region: 'us-east-1',
    maxTokens: 4096,
    temperature: 0.1,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:', '\n\nHuman:', '\n\nAssistant:'],
    timeout: 30000,
    retryAttempts: 3,
    rateLimits: {
      requestsPerMinute: 1000,
      tokensPerMinute: 100000
    }
  },

  'claude-3-sonnet': {
    provider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-sonnet-20240229-v1:0',
    region: 'us-east-1',
    maxTokens: 4096,
    temperature: 0.1,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:', '\n\nHuman:', '\n\nAssistant:'],
    timeout: 45000,
    retryAttempts: 3,
    rateLimits: {
      requestsPerMinute: 500,
      tokensPerMinute: 50000
    }
  },

  'claude-3-opus': {
    provider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-opus-20240229-v1:0',
    region: 'us-east-1',
    maxTokens: 4096,
    temperature: 0.1,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:', '\n\nHuman:', '\n\nAssistant:'],
    timeout: 60000,
    retryAttempts: 3,
    rateLimits: {
      requestsPerMinute: 100,
      tokensPerMinute: 20000
    }
  }
};

export interface PromptTemplate {
  name: string;
  template: string;
  variables: string[];
  examples?: Array<{
    input: Record<string, any>;
    output: string;
  }>;
}

export const PROMPT_TEMPLATES: Record<string, PromptTemplate> = {
  arbitrage_analysis: {
    name: 'Arbitrage Analysis',
    template: `Analyze the following arbitrage opportunity:

Token Pair: {{tokenPair}}
Source Exchange: {{sourceExchange}} ({{sourceChain}})
Target Exchange: {{targetExchange}} ({{targetChain}})
Price Difference: {{priceDifference}}%
Available Liquidity: {{liquidity}}
Estimated Gas Costs: {{gasCosts}}
Current Market Conditions: {{marketConditions}}

Please provide a comprehensive analysis including:
1. Profit calculation after all fees and costs
2. Risk assessment (liquidity, slippage, MEV)
3. Execution strategy and timing
4. Confidence level and reasoning

Respond in JSON format with the following structure:
{
  "analysis": "detailed analysis",
  "recommendation": "execute|reject|monitor",
  "expectedProfit": number,
  "riskScore": number,
  "executionPlan": "step-by-step plan",
  "confidence": number
}`,
    variables: ['tokenPair', 'sourceExchange', 'sourceChain', 'targetExchange', 'targetChain', 'priceDifference', 'liquidity', 'gasCosts', 'marketConditions']
  },

  portfolio_optimization: {
    name: 'Portfolio Optimization',
    template: `Optimize the following portfolio allocation:

Current Portfolio:
{{currentAllocation}}

Market Data:
{{marketData}}

Risk Metrics:
{{riskMetrics}}

User Preferences:
- Risk Tolerance: {{riskTolerance}}
- Investment Horizon: {{investmentHorizon}}
- Target Return: {{targetReturn}}

Constraints:
{{constraints}}

Provide optimization recommendations including:
1. Suggested allocation changes
2. Rebalancing strategy
3. Risk/return projections
4. Timing considerations

Respond in JSON format:
{
  "currentAnalysis": "current portfolio assessment",
  "optimizedAllocation": [{"asset": "string", "percentage": number}],
  "rebalancingActions": [{"action": "buy|sell", "asset": "string", "amount": number}],
  "projectedMetrics": {"expectedReturn": number, "volatility": number, "sharpeRatio": number},
  "executionPlan": "detailed execution strategy",
  "confidence": number
}`,
    variables: ['currentAllocation', 'marketData', 'riskMetrics', 'riskTolerance', 'investmentHorizon', 'targetReturn', 'constraints']
  },

  yield_opportunity: {
    name: 'Yield Opportunity Analysis',
    template: `Evaluate the following yield farming opportunity:

Protocol: {{protocolName}}
Pool: {{poolName}}
Current APY: {{currentAPY}}%
TVL: {{totalValueLocked}}
Token Emissions: {{tokenEmissions}}
Impermanent Loss Risk: {{impermanentLossRisk}}
Smart Contract Audit: {{auditStatus}}
Pool Composition: {{poolComposition}}

Historical Performance:
{{historicalData}}

Consider the following factors:
1. Yield sustainability and token emission schedule
2. Impermanent loss potential
3. Protocol risks and security
4. Liquidity and exit conditions
5. Tax implications

Provide analysis in JSON format:
{
  "yieldAnalysis": "comprehensive yield assessment",
  "riskAssessment": "detailed risk evaluation",
  "recommendation": "enter|avoid|monitor",
  "projectedYield": {"conservative": number, "expected": number, "optimistic": number},
  "riskFactors": ["list of key risks"],
  "migrationStrategy": "entry and exit strategy",
  "confidence": number
}`,
    variables: ['protocolName', 'poolName', 'currentAPY', 'totalValueLocked', 'tokenEmissions', 'impermanentLossRisk', 'auditStatus', 'poolComposition', 'historicalData']
  },

  security_alert: {
    name: 'Security Alert Analysis',
    template: `Analyze the following security alert:

Alert Type: {{alertType}}
Detected Activity: {{detectedActivity}}
Affected Accounts: {{affectedAccounts}}
Transaction Patterns: {{transactionPatterns}}
Risk Indicators: {{riskIndicators}}
Timestamp: {{timestamp}}

Context:
{{contextualData}}

Evaluate the security threat and provide:
1. Threat classification and severity
2. Potential impact assessment
3. Evidence analysis
4. Recommended immediate actions
5. Long-term mitigation strategies

Respond in JSON format:
{
  "threatLevel": "critical|high|medium|low",
  "threatType": "specific threat classification",
  "evidenceAnalysis": "detailed evidence review",
  "impactAssessment": "potential damage evaluation",
  "immediateActions": ["list of urgent actions"],
  "mitigationStrategy": "long-term security measures",
  "falsePositiveProbability": number,
  "confidence": number
}`,
    variables: ['alertType', 'detectedActivity', 'affectedAccounts', 'transactionPatterns', 'riskIndicators', 'timestamp', 'contextualData']
  },

  orchestration_decision: {
    name: 'Orchestration Decision',
    template: `Make an orchestration decision for the following system state:

Active Agents: {{activeAgents}}
Pending Tasks: {{pendingTasks}}
Resource Usage: {{resourceUsage}}
System Performance: {{systemPerformance}}
Conflicting Recommendations: {{conflicts}}

Market Conditions: {{marketConditions}}
Gas Prices: {{gasPrices}}
Network Congestion: {{networkCongestion}}

Prioritize and coordinate activities considering:
1. Resource allocation and optimization
2. Task prioritization and scheduling
3. Conflict resolution between agents
4. System performance and stability
5. Risk management and safety

Provide coordination plan in JSON format:
{
  "systemAssessment": "overall system health and status",
  "taskPrioritization": [{"task": "string", "priority": number, "assignedAgent": "string"}],
  "resourceAllocation": {"gas": number, "compute": number, "api": number},
  "conflictResolution": "strategy for handling conflicts",
  "riskMitigation": "system-wide risk management",
  "executionPlan": "coordinated execution strategy",
  "confidence": number
}`,
    variables: ['activeAgents', 'pendingTasks', 'resourceUsage', 'systemPerformance', 'conflicts', 'marketConditions', 'gasPrices', 'networkCongestion']
  }
};

export function getModelConfig(modelName: string): ModelConfig {
  const config = MODEL_CONFIGS[modelName];
  if (!config) {
    throw new Error(`Model configuration not found: ${modelName}`);
  }
  return config;
}

export function getPromptTemplate(templateName: string): PromptTemplate {
  const template = PROMPT_TEMPLATES[templateName];
  if (!template) {
    throw new Error(`Prompt template not found: ${templateName}`);
  }
  return template;
}

export function renderPromptTemplate(templateName: string, variables: Record<string, any>): string {
  const template = getPromptTemplate(templateName);
  let rendered = template.template;

  for (const [key, value] of Object.entries(variables)) {
    const placeholder = `{{${key}}}`;
    rendered = rendered.replace(new RegExp(placeholder, 'g'), String(value));
  }

  // Check for any unreplaced variables
  const unreplacedVars = rendered.match(/\{\{[^}]+\}\}/g);
  if (unreplacedVars) {
    throw new Error(`Unreplaced variables in template: ${unreplacedVars.join(', ')}`);
  }

  return rendered;
}
