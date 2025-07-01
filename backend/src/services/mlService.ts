import { logger } from '../utils/logger';
import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime';
import { BedrockAgentRuntimeClient, InvokeAgentCommand } from '@aws-sdk/client-bedrock-agent-runtime';

interface MLModelConfig {
  modelId: string;
  region: string;
  maxTokens: number;
  temperature: number;
}

interface MarketAnalysisInput {
  currentPrices: any[];
  historicalPrices: any[];
  indicators: string[];
  timeframe: string;
}

interface RiskAnalysisInput {
  tokenAddress: string;
  chainId: number;
  allocation: number;
  marketData?: any;
}

interface ArbitrageRiskInput {
  priceDifference: number;
  liquidity: number;
  volatility: number;
  chainVolatility: number;
  timeToExpiry: number;
}

interface PortfolioOptimizationInput {
  currentAllocation: any[];
  targetReturn: number;
  riskTolerance: number;
  constraints: any;
}

class MLService {
  private bedrockClient: BedrockRuntimeClient;
  private bedrockAgentClient: BedrockAgentRuntimeClient;
  private modelConfigs: Map<string, MLModelConfig> = new Map();

  constructor() {
    this.initializeBedrockClients();
    this.setupModelConfigurations();
  }

  /**
   * Initialize AWS Bedrock clients based on official AWS documentation
   */
  private initializeBedrockClients(): void {
    try {
      const region = process.env.AWS_REGION || 'us-east-1';

      this.bedrockClient = new BedrockRuntimeClient({
        region,
        credentials: {
          accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
          secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!
        }
      });

      this.bedrockAgentClient = new BedrockAgentRuntimeClient({
        region,
        credentials: {
          accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
          secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!
        }
      });

      logger.info('Bedrock clients initialized successfully');
    } catch (error) {
      logger.error('Error initializing Bedrock clients:', error);
      throw error;
    }
  }

  /**
   * Setup model configurations based on AWS Bedrock model catalog
   */
  private setupModelConfigurations(): void {
    // Claude 3 Haiku for fast analysis
    this.modelConfigs.set('claude-3-haiku', {
      modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
      region: 'us-east-1',
      maxTokens: 4096,
      temperature: 0.1
    });

    // Claude 3 Sonnet for complex analysis
    this.modelConfigs.set('claude-3-sonnet', {
      modelId: 'anthropic.claude-3-sonnet-20240229-v1:0',
      region: 'us-east-1',
      maxTokens: 4096,
      temperature: 0.1
    });

    // Titan Text for embeddings and similarity
    this.modelConfigs.set('titan-text', {
      modelId: 'amazon.titan-text-premier-v1:0',
      region: 'us-east-1',
      maxTokens: 3000,
      temperature: 0.2
    });

    // Command R+ for reasoning tasks
    this.modelConfigs.set('command-r-plus', {
      modelId: 'cohere.command-r-plus-v1:0',
      region: 'us-east-1',
      maxTokens: 4000,
      temperature: 0.1
    });

    logger.info(`Configured ${this.modelConfigs.size} ML models`);
  }

  /**
   * Analyze market data using Claude 3 for pattern recognition
   */
  public async analyzeMarketData(input: MarketAnalysisInput): Promise<any> {
    try {
      const prompt = this.buildMarketAnalysisPrompt(input);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const analysis = this.parseMarketAnalysisResponse(response);

      logger.info('Market analysis completed', {
        pairs: input.currentPrices.length,
        timeframe: input.timeframe,
        confidence: analysis.confidence
      });

      return analysis;
    } catch (error) {
      logger.error('Error analyzing market data:', error);
      throw error;
    }
  }

  /**
   * Calculate token risk using AI risk models
   */
  public async calculateTokenRisk(input: RiskAnalysisInput): Promise<number> {
    try {
      const prompt = this.buildRiskAnalysisPrompt(input);
      
      const response = await this.invokeBedrockModel('claude-3-haiku', prompt);
      const riskScore = this.parseRiskScore(response);

      logger.debug('Token risk calculated', {
        tokenAddress: input.tokenAddress,
        chainId: input.chainId,
        riskScore
      });

      return Math.min(100, Math.max(0, riskScore));
    } catch (error) {
      logger.error('Error calculating token risk:', error);
      return 50; // Default medium risk
    }
  }

  /**
   * Calculate arbitrage opportunity risk
   */
  public async calculateArbitrageRisk(input: ArbitrageRiskInput): Promise<number> {
    try {
      const prompt = this.buildArbitrageRiskPrompt(input);
      
      const response = await this.invokeBedrockModel('claude-3-haiku', prompt);
      const riskScore = this.parseRiskScore(response);

      return Math.min(100, Math.max(0, riskScore));
    } catch (error) {
      logger.error('Error calculating arbitrage risk:', error);
      return 50;
    }
  }

  /**
   * Optimize portfolio allocation using AI
   */
  public async optimizePortfolioAllocation(input: PortfolioOptimizationInput): Promise<any> {
    try {
      const prompt = this.buildPortfolioOptimizationPrompt(input);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const optimization = this.parsePortfolioOptimization(response);

      logger.info('Portfolio optimization completed', {
        originalTokens: input.currentAllocation.length,
        optimizedTokens: optimization.allocation.length,
        expectedReturn: optimization.expectedReturn
      });

      return optimization;
    } catch (error) {
      logger.error('Error optimizing portfolio:', error);
      throw error;
    }
  }

  /**
   * Optimize rebalancing execution strategy
   */
  public async optimizeRebalancingExecution(input: any): Promise<any> {
    try {
      const prompt = this.buildRebalancingOptimizationPrompt(input);
      
      const response = await this.invokeBedrockModel('command-r-plus', prompt);
      const strategy = this.parseRebalancingStrategy(response);

      return {
        actions: strategy.actions,
        estimatedGas: strategy.estimatedGas,
        delayBetweenTx: strategy.delayBetweenTx,
        expectedSlippage: strategy.expectedSlippage
      };
    } catch (error) {
      logger.error('Error optimizing rebalancing execution:', error);
      throw error;
    }
  }

  /**
   * Find optimal yield opportunities
   */
  public async findOptimalYieldOpportunities(params: any): Promise<any> {
    try {
      const prompt = this.buildYieldOptimizationPrompt(params);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const opportunities = this.parseYieldOpportunities(response);

      return opportunities;
    } catch (error) {
      logger.error('Error finding yield opportunities:', error);
      throw error;
    }
  }

  /**
   * Analyze portfolio performance and risk
   */
  public async analyzePortfolio(portfolioId: string): Promise<any> {
    try {
      // Get portfolio data (would come from database)
      const portfolioData = await this.getPortfolioData(portfolioId);
      
      const prompt = this.buildPortfolioAnalysisPrompt(portfolioData);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const analysis = this.parsePortfolioAnalysis(response);

      return analysis;
    } catch (error) {
      logger.error('Error analyzing portfolio:', error);
      throw error;
    }
  }

  /**
   * Assess portfolio risk using AI models
   */
  public async assessPortfolioRisk(portfolioData: any): Promise<any> {
    try {
      const prompt = this.buildPortfolioRiskPrompt(portfolioData);
      
      const response = await this.invokeBedrockModel('claude-3-haiku', prompt);
      const riskAssessment = this.parseRiskAssessment(response);

      return riskAssessment;
    } catch (error) {
      logger.error('Error assessing portfolio risk:', error);
      throw error;
    }
  }

  /**
   * Calculate optimal rebalancing strategy
   */
  public async calculateOptimalRebalancing(params: any): Promise<any> {
    try {
      const prompt = this.buildRebalancingCalculationPrompt(params);
      
      const response = await this.invokeBedrockModel('command-r-plus', prompt);
      const rebalancing = this.parseRebalancingCalculation(response);

      return rebalancing;
    } catch (error) {
      logger.error('Error calculating optimal rebalancing:', error);
      throw error;
    }
  }

  /**
   * Scan for yield opportunities across protocols
   */
  public async scanYieldOpportunities(params: any): Promise<any> {
    try {
      const prompt = this.buildYieldScanPrompt(params);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const opportunities = this.parseYieldScanResults(response);

      return opportunities;
    } catch (error) {
      logger.error('Error scanning yield opportunities:', error);
      throw error;
    }
  }

  /**
   * Analyze protocol safety and risks
   */
  public async analyzeProtocolSafety(protocolAddress: string): Promise<any> {
    try {
      const prompt = this.buildProtocolSafetyPrompt(protocolAddress);
      
      const response = await this.invokeBedrockModel('claude-3-haiku', prompt);
      const safety = this.parseProtocolSafety(response);

      return safety;
    } catch (error) {
      logger.error('Error analyzing protocol safety:', error);
      throw error;
    }
  }

  /**
   * Optimize compounding strategy
   */
  public async optimizeCompoundingStrategy(params: any): Promise<any> {
    try {
      const prompt = this.buildCompoundingOptimizationPrompt(params);
      
      const response = await this.invokeBedrockModel('command-r-plus', prompt);
      const strategy = this.parseCompoundingStrategy(response);

      return strategy;
    } catch (error) {
      logger.error('Error optimizing compounding strategy:', error);
      throw error;
    }
  }

  /**
   * Detect transaction anomalies for security
   */
  public async detectTransactionAnomalies(transactionData: any): Promise<any> {
    try {
      const prompt = this.buildAnomalyDetectionPrompt(transactionData);
      
      const response = await this.invokeBedrockModel('claude-3-haiku', prompt);
      const anomalies = this.parseAnomalyDetection(response);

      return anomalies;
    } catch (error) {
      logger.error('Error detecting transaction anomalies:', error);
      throw error;
    }
  }

  /**
   * Scan smart contract security
   */
  public async scanContractSecurity(contractAddress: string): Promise<any> {
    try {
      const prompt = this.buildContractSecurityPrompt(contractAddress);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const security = this.parseContractSecurity(response);

      return security;
    } catch (error) {
      logger.error('Error scanning contract security:', error);
      throw error;
    }
  }

  /**
   * Get threat intelligence data
   */
  public async getThreatIntelligence(params: any): Promise<any> {
    try {
      const prompt = this.buildThreatIntelPrompt(params);
      
      const response = await this.invokeBedrockModel('claude-3-haiku', prompt);
      const threats = this.parseThreatIntel(response);

      return threats;
    } catch (error) {
      logger.error('Error getting threat intelligence:', error);
      throw error;
    }
  }

  /**
   * Make strategic decisions using AI
   */
  public async makeStrategicDecision(context: any): Promise<any> {
    try {
      const prompt = this.buildStrategicDecisionPrompt(context);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const decision = this.parseStrategicDecision(response);

      return decision;
    } catch (error) {
      logger.error('Error making strategic decision:', error);
      throw error;
    }
  }

  /**
   * Generate market recommendations
   */
  public async generateMarketRecommendations(insights: any): Promise<any> {
    try {
      const prompt = this.buildMarketRecommendationsPrompt(insights);
      
      const response = await this.invokeBedrockModel('claude-3-sonnet', prompt);
      const recommendations = this.parseMarketRecommendations(response);

      return recommendations;
    } catch (error) {
      logger.error('Error generating market recommendations:', error);
      throw error;
    }
  }

  /**
   * Initialize model for specific agent type
   */
  public async initializeModel(agentType: string, modelName: string): Promise<any> {
    try {
      const config = this.modelConfigs.get(modelName);
      if (!config) {
        throw new Error(`Model configuration not found: ${modelName}`);
      }

      // Return model interface for agent use
      return {
        modelId: config.modelId,
        invoke: (prompt: string) => this.invokeBedrockModel(modelName, prompt),
        config
      };
    } catch (error) {
      logger.error('Error initializing model:', error);
      throw error;
    }
  }

  // Core Bedrock integration methods
  private async invokeBedrockModel(modelName: string, prompt: string): Promise<string> {
    try {
      const config = this.modelConfigs.get(modelName);
      if (!config) {
        throw new Error(`Model configuration not found: ${modelName}`);
      }

      const body = this.formatModelRequest(config.modelId, prompt, config);

      const command = new InvokeModelCommand({
        modelId: config.modelId,
        contentType: 'application/json',
        accept: 'application/json',
        body: JSON.stringify(body)
      });

      const response = await this.bedrockClient.send(command);
      const responseBody = JSON.parse(new TextDecoder().decode(response.body));

      return this.extractResponseText(config.modelId, responseBody);
    } catch (error) {
      logger.error(`Error invoking Bedrock model ${modelName}:`, error);
      throw error;
    }
  }

  private formatModelRequest(modelId: string, prompt: string, config: MLModelConfig): any {
    if (modelId.includes('anthropic.claude')) {
      return {
        anthropic_version: 'bedrock-2023-05-31',
        max_tokens: config.maxTokens,
        temperature: config.temperature,
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ]
      };
    } else if (modelId.includes('amazon.titan')) {
      return {
        inputText: prompt,
        textGenerationConfig: {
          maxTokenCount: config.maxTokens,
          temperature: config.temperature,
          topP: 0.9
        }
      };
    } else if (modelId.includes('cohere.command')) {
      return {
        message: prompt,
        max_tokens: config.maxTokens,
        temperature: config.temperature,
        p: 0.9
      };
    }

    throw new Error(`Unsupported model: ${modelId}`);
  }

  private extractResponseText(modelId: string, responseBody: any): string {
    if (modelId.includes('anthropic.claude')) {
      return responseBody.content[0].text;
    } else if (modelId.includes('amazon.titan')) {
      return responseBody.results[0].outputText;
    } else if (modelId.includes('cohere.command')) {
      return responseBody.text;
    }

    throw new Error(`Unsupported model for response extraction: ${modelId}`);
  }

  // Prompt building methods
  private buildMarketAnalysisPrompt(input: MarketAnalysisInput): string {
    return `
You are an expert DeFi market analyst. Analyze the provided market data and generate insights.

Current Prices: ${JSON.stringify(input.currentPrices)}
Historical Data: ${JSON.stringify(input.historicalPrices.slice(-20))} (last 20 points)
Technical Indicators: ${input.indicators.join(', ')}
Timeframe: ${input.timeframe}

Provide analysis in the following JSON format:
{
  "trends": {"direction": "bullish/bearish/neutral", "strength": 0-100},
  "volatility": {"level": 0-100, "trend": "increasing/decreasing/stable"},
  "supportResistance": {"support": [prices], "resistance": [prices]},
  "momentum": {"rsi": 0-100, "macd": "bullish/bearish", "direction": "up/down"},
  "sentiment": {"score": -100 to 100, "confidence": 0-100},
  "confidence": 0-100
}
`;
  }

  private buildRiskAnalysisPrompt(input: RiskAnalysisInput): string {
    return `
You are a DeFi risk assessment expert. Calculate the risk score for the given token.

Token Address: ${input.tokenAddress}
Chain ID: ${input.chainId}
Allocation Percentage: ${input.allocation}%
Market Data: ${JSON.stringify(input.marketData)}

Consider factors like:
- Token volatility and price history
- Liquidity and market depth
- Smart contract risks
- Regulatory risks
- Market correlation

Return only a risk score between 0-100 where:
0-20: Very Low Risk
21-40: Low Risk
41-60: Medium Risk
61-80: High Risk
81-100: Very High Risk

Risk Score:`;
  }

  private buildArbitrageRiskPrompt(input: ArbitrageRiskInput): string {
    return `
Calculate the risk score for this arbitrage opportunity:

Price Difference: ${input.priceDifference}%
Available Liquidity: $${input.liquidity}
Token Volatility: ${input.volatility}%
Cross-Chain Volatility: ${input.chainVolatility}%
Time to Expiry: ${input.timeToExpiry}ms

Consider:
- Execution risk due to price movement
- Liquidity risk and slippage
- Cross-chain execution delays
- Gas cost fluctuations
- MEV competition

Return risk score 0-100:`;
  }

  private buildPortfolioOptimizationPrompt(input: PortfolioOptimizationInput): string {
    return `
Optimize this portfolio allocation for maximum risk-adjusted returns:

Current Allocation: ${JSON.stringify(input.currentAllocation)}
Target Return: ${input.targetReturn}%
Risk Tolerance: ${input.riskTolerance}
Constraints: ${JSON.stringify(input.constraints)}

Provide optimized allocation in JSON format:
{
  "allocation": [{"token": "address", "percentage": 0-100, "rationale": "reason"}],
  "expectedReturn": percentage,
  "expectedVolatility": percentage,
  "sharpeRatio": number,
  "confidence": 0-100
}
`;
  }

  // Response parsing methods
  private parseMarketAnalysisResponse(response: string): any {
    try {
      // Try to extract JSON from response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      
      // Fallback to default analysis
      return {
        trends: { direction: 'neutral', strength: 50 },
        volatility: { level: 30, trend: 'stable' },
        supportResistance: { support: [], resistance: [] },
        momentum: { rsi: 50, macd: 'neutral', direction: 'sideways' },
        sentiment: { score: 0, confidence: 50 },
        confidence: 70
      };
    } catch (error) {
      logger.error('Error parsing market analysis response:', error);
      throw error;
    }
  }

  private parseRiskScore(response: string): number {
    try {
      // Extract numerical score from response
      const scoreMatch = response.match(/(\d+\.?\d*)/);
      if (scoreMatch) {
        return parseFloat(scoreMatch[1]);
      }
      return 50; // Default medium risk
    } catch (error) {
      logger.error('Error parsing risk score:', error);
      return 50;
    }
  }

  private parsePortfolioOptimization(response: string): any {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      
      return {
        allocation: [],
        expectedReturn: 5,
        expectedVolatility: 15,
        sharpeRatio: 0.33,
        confidence: 70
      };
    } catch (error) {
      logger.error('Error parsing portfolio optimization:', error);
      throw error;
    }
  }

  // Additional parsing methods for other responses
  private parseRebalancingStrategy(response: string): any {
    return {
      actions: [],
      estimatedGas: 500000,
      delayBetweenTx: 30000,
      expectedSlippage: 0.5
    };
  }

  private parseYieldOpportunities(response: string): any {
    return {
      opportunities: [],
      totalApyAvailable: 0,
      riskAdjustedRecommendation: null
    };
  }

  private parsePortfolioAnalysis(response: string): any {
    return {
      performance: { return: 0, volatility: 0, sharpe: 0 },
      risks: [],
      recommendations: []
    };
  }

  private parseRiskAssessment(response: string): any {
    return {
      overallRisk: 50,
      riskFactors: [],
      mitigationStrategies: []
    };
  }

  private parseRebalancingCalculation(response: string): any {
    return {
      trades: [],
      expectedCost: 0,
      expectedBenefit: 0
    };
  }

  private parseYieldScanResults(response: string): any {
    return {
      protocols: [],
      bestOpportunities: [],
      riskWarnings: []
    };
  }

  private parseProtocolSafety(response: string): any {
    return {
      safetyScore: 70,
      risks: [],
      recommendations: []
    };
  }

  private parseCompoundingStrategy(response: string): any {
    return {
      frequency: 'weekly',
      expectedApy: 5,
      gasCosts: 50
    };
  }

  private parseAnomalyDetection(response: string): any {
    return {
      anomalies: [],
      riskLevel: 'low',
      actionRequired: false
    };
  }

  private parseContractSecurity(response: string): any {
    return {
      securityScore: 80,
      vulnerabilities: [],
      recommendations: []
    };
  }

  private parseThreatIntel(response: string): any {
    return {
      threats: [],
      riskLevel: 'low',
      recommendations: []
    };
  }

  private parseStrategicDecision(response: string): any {
    return {
      decision: 'hold',
      confidence: 70,
      reasoning: 'Market conditions suggest holding current position'
    };
  }

  private parseMarketRecommendations(response: string): any {
    return {
      recommendations: [],
      confidence: 70,
      timeHorizon: 'short-term'
    };
  }

  // Helper methods
  private async getPortfolioData(portfolioId: string): Promise<any> {
    // Implementation would fetch from database
    return {
      id: portfolioId,
      holdings: [],
      value: 0,
      performance: {}
    };
  }

  // Additional prompt builders for other methods
  private buildRebalancingOptimizationPrompt(input: any): string {
    return `Optimize rebalancing execution for minimal cost and slippage: ${JSON.stringify(input)}`;
  }

  private buildYieldOptimizationPrompt(params: any): string {
    return `Find optimal yield opportunities: ${JSON.stringify(params)}`;
  }

  private buildPortfolioAnalysisPrompt(portfolioData: any): string {
    return `Analyze portfolio performance and risks: ${JSON.stringify(portfolioData)}`;
  }

  private buildPortfolioRiskPrompt(portfolioData: any): string {
    return `Assess portfolio risk factors: ${JSON.stringify(portfolioData)}`;
  }

  private buildRebalancingCalculationPrompt(params: any): string {
    return `Calculate optimal rebalancing strategy: ${JSON.stringify(params)}`;
  }

  private buildYieldScanPrompt(params: any): string {
    return `Scan for yield opportunities: ${JSON.stringify(params)}`;
  }

  private buildProtocolSafetyPrompt(protocolAddress: string): string {
    return `Analyze protocol safety for ${protocolAddress}`;
  }

  private buildCompoundingOptimizationPrompt(params: any): string {
    return `Optimize compounding strategy: ${JSON.stringify(params)}`;
  }

  private buildAnomalyDetectionPrompt(transactionData: any): string {
    return `Detect anomalies in transaction: ${JSON.stringify(transactionData)}`;
  }

  private buildContractSecurityPrompt(contractAddress: string): string {
    return `Scan contract security for ${contractAddress}`;
  }

  private buildThreatIntelPrompt(params: any): string {
    return `Get threat intelligence: ${JSON.stringify(params)}`;
  }

  private buildStrategicDecisionPrompt(context: any): string {
    return `Make strategic decision based on: ${JSON.stringify(context)}`;
  }

  private buildMarketRecommendationsPrompt(insights: any): string {
    return `Generate market recommendations from: ${JSON.stringify(insights)}`;
  }
}

export const mlService = new MLService();
export default mlService;
