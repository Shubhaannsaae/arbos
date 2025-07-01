import { logger } from '../utils/logger';
import { AgentType } from '../models/Agent';

interface AgentTypeConfig {
  name: string;
  description: string;
  capabilities: string[];
  tools: string[];
  permissions: string[];
  resources: {
    minCpu: number;
    minMemory: number;
    maxConcurrentTasks: number;
  };
  configuration: {
    defaultParameters: any;
    requiredParameters: string[];
    validationRules: any;
  };
  integrations: {
    chainlink: string[];
    aws: string[];
    external: string[];
  };
}

interface ModelConfiguration {
  provider: 'aws-bedrock' | 'openai' | 'anthropic' | 'custom';
  modelId: string;
  parameters: {
    temperature: number;
    maxTokens: number;
    topP: number;
    frequencyPenalty: number;
    presencePenalty: number;
  };
  costPerToken: {
    input: number;
    output: number;
  };
  rateLimits: {
    requestsPerMinute: number;
    tokensPerMinute: number;
  };
}

interface AgentConfiguration {
  types: { [key in AgentType]: AgentTypeConfig };
  models: { [modelName: string]: ModelConfiguration };
  global: {
    maxAgentsPerUser: { [tier: string]: number };
    executionTimeout: number;
    memoryLimit: number;
    storageLimit: number;
    logRetentionDays: number;
  };
  security: {
    sandboxing: boolean;
    networkAccess: string[];
    fileSystemAccess: boolean;
    maxExecutionTime: number;
    resourceLimits: {
      cpu: number;
      memory: number;
      network: number;
    };
  };
}

class AgentConfigService {
  private config: AgentConfiguration;

  constructor() {
    this.config = this.loadAgentConfiguration();
    this.validateConfiguration();
  }

  /**
   * Load agent configuration based on environment and requirements
   */
  private loadAgentConfiguration(): AgentConfiguration {
    return {
      types: {
        [AgentType.ARBITRAGE]: {
          name: 'Arbitrage Agent',
          description: 'Detects and executes arbitrage opportunities across DEXes and chains',
          capabilities: [
            'cross_chain_arbitrage',
            'dex_price_monitoring',
            'gas_optimization',
            'slippage_calculation',
            'profit_estimation',
            'risk_assessment'
          ],
          tools: [
            'chainlink_price_feeds',
            'chainlink_ccip',
            'dex_aggregators',
            'gas_estimators',
            'profit_calculators'
          ],
          permissions: [
            'read_prices',
            'execute_trades',
            'bridge_tokens',
            'access_dex_apis',
            'monitor_mempool'
          ],
          resources: {
            minCpu: 2,
            minMemory: 4096, // MB
            maxConcurrentTasks: 10
          },
          configuration: {
            defaultParameters: {
              minProfitThreshold: 0.5, // %
              maxSlippage: 1.0, // %
              gasOptimization: true,
              riskTolerance: 'medium',
              executionTimeout: 300000, // 5 minutes
              priceUpdateInterval: 30000, // 30 seconds
              maxPositionSize: 10000 // USD
            },
            requiredParameters: [
              'minProfitThreshold',
              'maxSlippage',
              'allowedChains',
              'allowedTokens'
            ],
            validationRules: {
              minProfitThreshold: { min: 0.1, max: 10 },
              maxSlippage: { min: 0.1, max: 5 },
              maxPositionSize: { min: 100, max: 100000 }
            }
          },
          integrations: {
            chainlink: ['data_feeds', 'ccip', 'automation'],
            aws: ['bedrock', 'lambda'],
            external: ['dex_apis', 'block_explorers']
          }
        },
        [AgentType.PORTFOLIO]: {
          name: 'Portfolio Management Agent',
          description: 'Manages and optimizes portfolio allocations with automatic rebalancing',
          capabilities: [
            'portfolio_analysis',
            'risk_assessment',
            'performance_tracking',
            'rebalancing',
            'yield_optimization',
            'diversification_analysis'
          ],
          tools: [
            'chainlink_price_feeds',
            'chainlink_automation',
            'portfolio_analyzers',
            'risk_calculators',
            'yield_scanners'
          ],
          permissions: [
            'read_portfolio',
            'execute_rebalancing',
            'access_price_data',
            'modify_allocations',
            'access_yield_protocols'
          ],
          resources: {
            minCpu: 1,
            minMemory: 2048,
            maxConcurrentTasks: 5
          },
          configuration: {
            defaultParameters: {
              rebalanceThreshold: 5.0, // %
              rebalanceFrequency: 'weekly',
              riskTolerance: 'moderate',
              maxDiversification: 20, // max tokens
              minAllocation: 1.0, // %
              maxAllocation: 30.0, // %
              performanceReviewPeriod: 30 // days
            },
            requiredParameters: [
              'targetAllocation',
              'rebalanceThreshold',
              'riskTolerance'
            ],
            validationRules: {
              rebalanceThreshold: { min: 1, max: 50 },
              maxDiversification: { min: 2, max: 50 },
              minAllocation: { min: 0.1, max: 10 },
              maxAllocation: { min: 10, max: 80 }
            }
          },
          integrations: {
            chainlink: ['data_feeds', 'automation', 'functions'],
            aws: ['bedrock'],
            external: ['defi_protocols', 'analytics_apis']
          }
        },
        [AgentType.YIELD]: {
          name: 'Yield Optimization Agent',
          description: 'Finds and manages optimal yield farming opportunities',
          capabilities: [
            'yield_scanning',
            'protocol_analysis',
            'auto_compounding',
            'risk_monitoring',
            'apy_comparison',
            'liquidity_optimization'
          ],
          tools: [
            'chainlink_price_feeds',
            'chainlink_automation',
            'yield_scanners',
            'protocol_analyzers',
            'compound_optimizers'
          ],
          permissions: [
            'access_defi_protocols',
            'execute_yield_strategies',
            'manage_liquidity',
            'compound_rewards',
            'migrate_positions'
          ],
          resources: {
            minCpu: 1,
            minMemory: 2048,
            maxConcurrentTasks: 8
          },
          configuration: {
            defaultParameters: {
              minAPY: 5.0, // %
              maxRiskScore: 70,
              compoundFrequency: 'daily',
              diversificationLimit: 10,
              emergencyExitThreshold: 20.0, // % loss
              gasOptimization: true
            },
            requiredParameters: [
              'minAPY',
              'maxRiskScore',
              'allowedProtocols'
            ],
            validationRules: {
              minAPY: { min: 1, max: 100 },
              maxRiskScore: { min: 10, max: 100 },
              diversificationLimit: { min: 1, max: 20 }
            }
          },
          integrations: {
            chainlink: ['data_feeds', 'automation'],
            aws: ['bedrock'],
            external: ['defi_protocols', 'yield_apis']
          }
        },
        [AgentType.SECURITY]: {
          name: 'Security Monitoring Agent',
          description: 'Monitors transactions and portfolios for security threats',
          capabilities: [
            'anomaly_detection',
            'risk_scoring',
            'threat_monitoring',
            'compliance_checking',
            'incident_response',
            'forensic_analysis'
          ],
          tools: [
            'chainlink_price_feeds',
            'anomaly_detectors',
            'threat_intel_apis',
            'compliance_checkers',
            'forensic_tools'
          ],
          permissions: [
            'monitor_transactions',
            'access_security_apis',
            'generate_alerts',
            'emergency_actions',
            'read_all_data'
          ],
          resources: {
            minCpu: 2,
            minMemory: 4096,
            maxConcurrentTasks: 15
          },
          configuration: {
            defaultParameters: {
              alertThreshold: 'medium',
              monitoringScope: 'full',
              responseMode: 'automatic',
              retentionPeriod: 90, // days
              sensitivityLevel: 'high',
              falsePositiveReduction: true
            },
            requiredParameters: [
              'monitoringScope',
              'alertThreshold',
              'responseMode'
            ],
            validationRules: {
              retentionPeriod: { min: 30, max: 365 },
              alertThreshold: { enum: ['low', 'medium', 'high', 'critical'] }
            }
          },
          integrations: {
            chainlink: ['data_feeds', 'functions'],
            aws: ['bedrock', 'guardduty'],
            external: ['threat_intel', 'compliance_apis']
          }
        },
        [AgentType.ORCHESTRATOR]: {
          name: 'Orchestrator Agent',
          description: 'Coordinates and manages multiple specialized agents',
          capabilities: [
            'agent_coordination',
            'task_scheduling',
            'resource_management',
            'workflow_optimization',
            'conflict_resolution',
            'performance_monitoring'
          ],
          tools: [
            'chainlink_automation',
            'task_schedulers',
            'resource_managers',
            'workflow_engines',
            'monitoring_systems'
          ],
          permissions: [
            'manage_agents',
            'coordinate_tasks',
            'allocate_resources',
            'modify_workflows',
            'emergency_override'
          ],
          resources: {
            minCpu: 4,
            minMemory: 8192,
            maxConcurrentTasks: 50
          },
          configuration: {
            defaultParameters: {
              maxManagedAgents: 20,
              coordinationInterval: 60000, // 1 minute
              resourceAllocationStrategy: 'dynamic',
              conflictResolutionMode: 'priority_based',
              performanceReviewInterval: 3600000, // 1 hour
              emergencyProtocols: true
            },
            requiredParameters: [
              'maxManagedAgents',
              'coordinationStrategy',
              'emergencyProtocols'
            ],
            validationRules: {
              maxManagedAgents: { min: 1, max: 100 },
              coordinationInterval: { min: 10000, max: 300000 }
            }
          },
          integrations: {
            chainlink: ['automation', 'functions'],
            aws: ['bedrock', 'step_functions'],
            external: ['monitoring_apis']
          }
        }
      },
      models: {
        'claude-3-haiku': {
          provider: 'aws-bedrock',
          modelId: 'anthropic.claude-3-haiku-20240307-v1:0',
          parameters: {
            temperature: 0.1,
            maxTokens: 4096,
            topP: 0.9,
            frequencyPenalty: 0,
            presencePenalty: 0
          },
          costPerToken: {
            input: 0.00025,
            output: 0.00125
          },
          rateLimits: {
            requestsPerMinute: 1000,
            tokensPerMinute: 100000
          }
        },
        'claude-3-sonnet': {
          provider: 'aws-bedrock',
          modelId: 'anthropic.claude-3-sonnet-20240229-v1:0',
          parameters: {
            temperature: 0.1,
            maxTokens: 4096,
            topP: 0.9,
            frequencyPenalty: 0,
            presencePenalty: 0
          },
          costPerToken: {
            input: 0.003,
            output: 0.015
          },
          rateLimits: {
            requestsPerMinute: 500,
            tokensPerMinute: 50000
          }
        },
        'titan-text': {
          provider: 'aws-bedrock',
          modelId: 'amazon.titan-text-premier-v1:0',
          parameters: {
            temperature: 0.2,
            maxTokens: 3000,
            topP: 0.9,
            frequencyPenalty: 0,
            presencePenalty: 0
          },
          costPerToken: {
            input: 0.0005,
            output: 0.0015
          },
          rateLimits: {
            requestsPerMinute: 800,
            tokensPerMinute: 80000
          }
        },
        'command-r-plus': {
          provider: 'aws-bedrock',
          modelId: 'cohere.command-r-plus-v1:0',
          parameters: {
            temperature: 0.1,
            maxTokens: 4000,
            topP: 0.9,
            frequencyPenalty: 0,
            presencePenalty: 0
          },
          costPerToken: {
            input: 0.003,
            output: 0.015
          },
          rateLimits: {
            requestsPerMinute: 300,
            tokensPerMinute: 30000
          }
        }
      },
      global: {
        maxAgentsPerUser: {
          free: 1,
          basic: 5,
          premium: 20,
          enterprise: 100
        },
        executionTimeout: parseInt(process.env.AGENT_EXECUTION_TIMEOUT || '300000'), // 5 minutes
        memoryLimit: parseInt(process.env.AGENT_MEMORY_LIMIT || '8192'), // MB
        storageLimit: parseInt(process.env.AGENT_STORAGE_LIMIT || '10240'), // MB
        logRetentionDays: parseInt(process.env.AGENT_LOG_RETENTION || '30')
      },
      security: {
        sandboxing: process.env.AGENT_SANDBOXING !== 'false',
        networkAccess: (process.env.AGENT_NETWORK_ACCESS || 'restricted,blockchain,apis').split(','),
        fileSystemAccess: process.env.AGENT_FILESYSTEM_ACCESS === 'true',
        maxExecutionTime: parseInt(process.env.AGENT_MAX_EXECUTION_TIME || '600000'), // 10 minutes
        resourceLimits: {
          cpu: parseInt(process.env.AGENT_CPU_LIMIT || '80'), // %
          memory: parseInt(process.env.AGENT_MEMORY_PERCENT_LIMIT || '90'), // %
          network: parseInt(process.env.AGENT_NETWORK_LIMIT || '1000') // KB/s
        }
      }
    };
  }

  /**
   * Validate agent configuration
   */
  private validateConfiguration(): void {
    // Validate agent types
    for (const [type, config] of Object.entries(this.config.types)) {
      if (!config.name || !config.description) {
        throw new Error(`Invalid configuration for agent type ${type}: missing name or description`);
      }

      if (!config.capabilities || config.capabilities.length === 0) {
        throw new Error(`Invalid configuration for agent type ${type}: no capabilities defined`);
      }

      if (!config.tools || config.tools.length === 0) {
        throw new Error(`Invalid configuration for agent type ${type}: no tools defined`);
      }

      if (!config.permissions || config.permissions.length === 0) {
        throw new Error(`Invalid configuration for agent type ${type}: no permissions defined`);
      }
    }

    // Validate models
    for (const [modelName, modelConfig] of Object.entries(this.config.models)) {
      if (!modelConfig.modelId || !modelConfig.provider) {
        throw new Error(`Invalid configuration for model ${modelName}: missing modelId or provider`);
      }

      if (modelConfig.parameters.temperature < 0 || modelConfig.parameters.temperature > 2) {
        throw new Error(`Invalid temperature for model ${modelName}: must be between 0 and 2`);
      }

      if (modelConfig.parameters.maxTokens <= 0) {
        throw new Error(`Invalid maxTokens for model ${modelName}: must be positive`);
      }
    }

    // Validate global settings
    if (this.config.global.executionTimeout <= 0) {
      throw new Error('Invalid executionTimeout: must be positive');
    }

    if (this.config.global.memoryLimit <= 0) {
      throw new Error('Invalid memoryLimit: must be positive');
    }

    // Validate security settings
    if (this.config.security.maxExecutionTime <= this.config.global.executionTimeout) {
      throw new Error('maxExecutionTime must be greater than executionTimeout');
    }

    logger.info('Agent configuration validated successfully', {
      agentTypes: Object.keys(this.config.types).length,
      models: Object.keys(this.config.models).length,
      sandboxing: this.config.security.sandboxing
    });
  }

  /**
   * Get configuration for specific agent type
   */
  public getAgentConfig(type: AgentType): AgentTypeConfig | null {
    return this.config.types[type] || null;
  }

  /**
   * Get model configuration
   */
  public getModelConfig(modelName: string): ModelConfiguration | null {
    return this.config.models[modelName] || null;
  }

  /**
   * Get all available agent types
   */
  public getAvailableAgentTypes(): AgentType[] {
    return Object.keys(this.config.types) as AgentType[];
  }

  /**
   * Get all available models
   */
  public getAvailableModels(): string[] {
    return Object.keys(this.config.models);
  }

  /**
   * Get global configuration
   */
  public getGlobalConfig(): AgentConfiguration['global'] {
    return this.config.global;
  }

  /**
   * Get security configuration
   */
  public getSecurityConfig(): AgentConfiguration['security'] {
    return this.config.security;
  }

  /**
   * Check if agent type supports specific capability
   */
  public hasCapability(type: AgentType, capability: string): boolean {
    const config = this.getAgentConfig(type);
    return config ? config.capabilities.includes(capability) : false;
  }

  /**
   * Check if agent type has specific tool
   */
  public hasTool(type: AgentType, tool: string): boolean {
    const config = this.getAgentConfig(type);
    return config ? config.tools.includes(tool) : false;
  }

  /**
   * Check if agent type has specific permission
   */
  public hasPermission(type: AgentType, permission: string): boolean {
    const config = this.getAgentConfig(type);
    return config ? config.permissions.includes(permission) : false;
  }

  /**
   * Get max agents allowed for subscription tier
   */
  public getMaxAgentsForTier(tier: string): number {
    return this.config.global.maxAgentsPerUser[tier] || 1;
  }

  /**
   * Validate agent parameters against configuration
   */
  public validateAgentParameters(type: AgentType, parameters: any): { isValid: boolean; errors: string[] } {
    const config = this.getAgentConfig(type);
    if (!config) {
      return { isValid: false, errors: ['Invalid agent type'] };
    }

    const errors: string[] = [];

    // Check required parameters
    for (const required of config.configuration.requiredParameters) {
      if (!(required in parameters)) {
        errors.push(`Missing required parameter: ${required}`);
      }
    }

    // Validate parameter values
    for (const [param, value] of Object.entries(parameters)) {
      const rule = config.configuration.validationRules[param];
      if (rule) {
        if (rule.min !== undefined && value < rule.min) {
          errors.push(`${param} must be at least ${rule.min}`);
        }
        if (rule.max !== undefined && value > rule.max) {
          errors.push(`${param} must be at most ${rule.max}`);
        }
        if (rule.enum && !rule.enum.includes(value)) {
          errors.push(`${param} must be one of: ${rule.enum.join(', ')}`);
        }
      }
    }

    return { isValid: errors.length === 0, errors };
  }

  /**
   * Get recommended model for agent type
   */
  public getRecommendedModel(type: AgentType): string {
    const typeModelMap = {
      [AgentType.ARBITRAGE]: 'claude-3-haiku', // Fast response needed
      [AgentType.PORTFOLIO]: 'claude-3-sonnet', // Complex analysis
      [AgentType.YIELD]: 'claude-3-haiku', // Fast scanning
      [AgentType.SECURITY]: 'claude-3-sonnet', // Thorough analysis
      [AgentType.ORCHESTRATOR]: 'command-r-plus' // Complex reasoning
    };

    return typeModelMap[type] || 'claude-3-haiku';
  }

  /**
   * Get estimated cost for agent operation
   */
  public estimateOperationCost(modelName: string, inputTokens: number, outputTokens: number): number {
    const modelConfig = this.getModelConfig(modelName);
    if (!modelConfig) {
      return 0;
    }

    const inputCost = inputTokens * modelConfig.costPerToken.input;
    const outputCost = outputTokens * modelConfig.costPerToken.output;
    
    return inputCost + outputCost;
  }

  /**
   * Check if operation is within rate limits
   */
  public checkRateLimit(modelName: string, requestsInMinute: number, tokensInMinute: number): boolean {
    const modelConfig = this.getModelConfig(modelName);
    if (!modelConfig) {
      return false;
    }

    return requestsInMinute <= modelConfig.rateLimits.requestsPerMinute &&
           tokensInMinute <= modelConfig.rateLimits.tokensPerMinute;
  }

  /**
   * Get configuration summary for logging
   */
  public getConfigSummary(): any {
    return {
      agentTypes: Object.keys(this.config.types),
      models: Object.keys(this.config.models),
      security: {
        sandboxing: this.config.security.sandboxing,
        networkAccess: this.config.security.networkAccess,
        fileSystemAccess: this.config.security.fileSystemAccess
      },
      limits: {
        executionTimeout: this.config.global.executionTimeout,
        memoryLimit: this.config.global.memoryLimit,
        storageLimit: this.config.global.storageLimit
      }
    };
  }

  /**
   * Get default parameters for agent type
   */
  public getDefaultParameters(type: AgentType): any {
    const config = this.getAgentConfig(type);
    return config ? { ...config.configuration.defaultParameters } : {};
  }

  /**
   * Merge user parameters with defaults
   */
  public mergeWithDefaults(type: AgentType, userParameters: any): any {
    const defaults = this.getDefaultParameters(type);
    return { ...defaults, ...userParameters };
  }

  /**
   * Get tools required for agent type
   */
  public getRequiredTools(type: AgentType): string[] {
    const config = this.getAgentConfig(type);
    return config ? [...config.tools] : [];
  }

  /**
   * Get integrations for agent type
   */
  public getIntegrations(type: AgentType): AgentTypeConfig['integrations'] | null {
    const config = this.getAgentConfig(type);
    return config ? config.integrations : null;
  }

  /**
   * Check if model is suitable for agent type
   */
  public isModelSuitableForAgent(type: AgentType, modelName: string): boolean {
    const agentConfig = this.getAgentConfig(type);
    const modelConfig = this.getModelConfig(modelName);
    
    if (!agentConfig || !modelConfig) {
      return false;
    }

    // Check if model can handle the agent's requirements
    const requiresHighPerformance = agentConfig.capabilities.includes('real_time_analysis') ||
                                   agentConfig.capabilities.includes('high_frequency_trading');
    
    if (requiresHighPerformance && modelConfig.rateLimits.requestsPerMinute < 500) {
      return false;
    }

    return true;
  }
}

// Create singleton instance
export const agentConfigService = new AgentConfigService();

// Export configuration class
export { AgentConfigService };

// Export types
export type { AgentConfiguration, AgentTypeConfig, ModelConfiguration };

// Default export
export default agentConfigService;
