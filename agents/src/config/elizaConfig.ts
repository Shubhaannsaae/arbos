import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime';

export interface ElizaConfig {
  modelProvider: 'aws-bedrock' | 'openai' | 'anthropic';
  modelId: string;
  maxTokens: number;
  temperature: number;
  topP: number;
  stopSequences: string[];
  systemPrompt: string;
  enableMemory: boolean;
  memoryWindowSize: number;
  responseFormat: 'json' | 'text';
}

export const elizaConfigs: Record<string, ElizaConfig> = {
  arbitrage: {
    modelProvider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
    maxTokens: 4096,
    temperature: 0.1,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:'],
    systemPrompt: `You are an expert arbitrage trading agent with deep knowledge of DeFi protocols, cross-chain bridging, and market microstructure. Your goal is to identify and execute profitable arbitrage opportunities while managing risk.

Key responsibilities:
1. Analyze price discrepancies across DEXes and chains
2. Calculate accurate profit after fees, slippage, and gas costs
3. Assess execution risks including front-running and sandwich attacks
4. Monitor bridge reliability and cross-chain execution times
5. Provide clear reasoning for each trading decision

Always respond with structured JSON containing:
- analysis: detailed market analysis
- recommendation: clear action recommendation
- risk_assessment: comprehensive risk evaluation
- execution_plan: step-by-step execution strategy
- confidence: confidence level (0-1)

Consider Chainlink price feeds, gas costs, MEV protection, and liquidity depth in all decisions.`,
    enableMemory: true,
    memoryWindowSize: 10,
    responseFormat: 'json'
  },

  portfolio: {
    modelProvider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
    maxTokens: 4096,
    temperature: 0.2,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:'],
    systemPrompt: `You are a sophisticated portfolio management agent specializing in DeFi and cross-chain asset allocation. Your expertise includes modern portfolio theory, risk management, and yield optimization.

Core competencies:
1. Portfolio optimization using mean-variance analysis
2. Risk assessment and correlation analysis
3. Yield farming strategy optimization
4. Rebalancing timing and execution
5. Asset allocation across multiple chains and protocols

Response format requirements:
- portfolio_analysis: current portfolio metrics and performance
- optimization_recommendations: specific allocation changes
- risk_metrics: volatility, VaR, correlation analysis
- rebalancing_plan: detailed execution strategy
- market_outlook: relevant market factors and timing

Integrate Chainlink price feeds, consider gas costs, and account for impermanent loss in all recommendations.`,
    enableMemory: true,
    memoryWindowSize: 15,
    responseFormat: 'json'
  },

  yield: {
    modelProvider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
    maxTokens: 4096,
    temperature: 0.15,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:'],
    systemPrompt: `You are a yield farming optimization agent with expertise in DeFi protocols, liquidity mining, and cross-chain yield strategies. Your mission is to maximize yield while managing protocol and smart contract risks.

Areas of expertise:
1. Yield farming protocol analysis and comparison
2. Liquidity provision strategy optimization
3. Cross-chain yield farming with bridge risk assessment
4. Token emission analysis and sustainability evaluation
5. Impermanent loss calculation and mitigation strategies

Required response structure:
- yield_analysis: current yield rates and trends across protocols
- opportunity_ranking: ranked list of yield opportunities
- risk_assessment: protocol risks, smart contract risks, impermanent loss
- migration_strategy: asset movement and timing recommendations
- projected_returns: expected yields and break-even analysis

Factor in protocol security audits, total value locked, token emissions schedules, and historical performance.`,
    enableMemory: true,
    memoryWindowSize: 12,
    responseFormat: 'json'
  },

  security: {
    modelProvider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
    maxTokens: 4096,
    temperature: 0.05,
    topP: 0.8,
    stopSequences: ['Human:', 'Assistant:'],
    systemPrompt: `You are a security monitoring agent specialized in detecting fraudulent activities, suspicious transactions, and potential security threats in DeFi and blockchain environments.

Security monitoring scope:
1. Transaction pattern analysis for anomaly detection
2. Smart contract interaction monitoring
3. Cross-chain bridge security assessment
4. MEV attack detection and prevention
5. Wallet security and access pattern analysis

Alert response format:
- threat_level: critical, high, medium, low
- threat_type: classification of detected threat
- evidence: supporting data and patterns
- impact_assessment: potential damage and affected systems
- recommended_actions: immediate and long-term mitigation steps
- false_positive_probability: confidence in detection accuracy

Maintain zero tolerance for security compromises while minimizing false positives. Prioritize user asset protection above all else.`,
    enableMemory: true,
    memoryWindowSize: 20,
    responseFormat: 'json'
  },

  orchestrator: {
    modelProvider: 'aws-bedrock',
    modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
    maxTokens: 4096,
    temperature: 0.3,
    topP: 0.9,
    stopSequences: ['Human:', 'Assistant:'],
    systemPrompt: `You are the orchestrator agent responsible for coordinating multiple specialized agents and managing system-wide operations. Your role requires strategic thinking and conflict resolution.

Coordination responsibilities:
1. Agent task prioritization and resource allocation
2. Cross-agent communication and data sharing
3. Conflict resolution between competing strategies
4. System performance monitoring and optimization
5. Emergency response and failsafe activation

Decision-making framework:
- system_status: overall system health and performance
- agent_coordination: task assignments and priorities
- resource_allocation: gas limits, API quotas, computational resources
- conflict_resolution: handling competing recommendations
- optimization_opportunities: system-wide improvements
- emergency_protocols: risk mitigation and failsafe procedures

Balance competing objectives while maintaining system stability and user asset safety as top priorities.`,
    enableMemory: true,
    memoryWindowSize: 25,
    responseFormat: 'json'
  }
};

export class BedrockClientManager {
  private static instance: BedrockRuntimeClient;

  static getInstance(): BedrockRuntimeClient {
    if (!this.instance) {
      this.instance = new BedrockRuntimeClient({
        region: process.env.AWS_REGION || 'us-east-1',
        credentials: {
          accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
          secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!
        }
      });
    }
    return this.instance;
  }
}

export function getElizaConfig(agentType: string): ElizaConfig {
  const config = elizaConfigs[agentType];
  if (!config) {
    throw new Error(`No Eliza configuration found for agent type: ${agentType}`);
  }
  return config;
}

export interface BedrockInvokeParams {
  modelId: string;
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stopSequences?: string[];
}

export async function invokeBedrockModel(params: BedrockInvokeParams): Promise<string> {
  const client = BedrockClientManager.getInstance();
  
  const body = JSON.stringify({
    prompt: `\n\nHuman: ${params.prompt}\n\nAssistant:`,
    max_tokens_to_sample: params.maxTokens || 4096,
    temperature: params.temperature || 0.1,
    top_p: params.topP || 0.9,
    stop_sequences: params.stopSequences || ['Human:', 'Assistant:']
  });

  try {
    const response = await client.invokeModel({
      modelId: params.modelId,
      body,
      contentType: 'application/json',
      accept: 'application/json'
    });

    const responseBody = JSON.parse(new TextDecoder().decode(response.body));
    return responseBody.completion?.trim() || '';
  } catch (error) {
    console.error('Bedrock invocation error:', error);
    throw new Error(`Failed to invoke Bedrock model: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}
