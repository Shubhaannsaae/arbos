import winston from 'winston';

export interface LogContext {
  agentId?: string;
  userId?: string;
  sessionId?: string;
  transactionHash?: string;
  chainId?: number;
  action?: string;
  duration?: number;
  gasUsed?: string;
  error?: Error;
  [key: string]: any;
}

class AgentLogger {
  private logger: winston.Logger;

  constructor() {
    this.logger = winston.createLogger({
      level: process.env.LOG_LEVEL || 'info',
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json(),
        winston.format.printf(({ timestamp, level, message, ...meta }) => {
          return JSON.stringify({
            timestamp,
            level,
            message,
            ...meta
          });
        })
      ),
      defaultMeta: {
        service: 'arbos-agents',
        version: process.env.npm_package_version || '1.0.0',
        environment: process.env.NODE_ENV || 'development'
      },
      transports: [
        new winston.transports.Console({
          format: winston.format.combine(
            winston.format.colorize(),
            winston.format.simple()
          )
        }),
        new winston.transports.File({
          filename: 'logs/error.log',
          level: 'error',
          maxsize: 5242880, // 5MB
          maxFiles: 5
        }),
        new winston.transports.File({
          filename: 'logs/combined.log',
          maxsize: 5242880, // 5MB
          maxFiles: 5
        })
      ]
    });

    // Handle uncaught exceptions
    this.logger.exceptions.handle(
      new winston.transports.File({
        filename: 'logs/exceptions.log',
        maxsize: 5242880,
        maxFiles: 5
      })
    );

    // Handle unhandled promise rejections
    this.logger.rejections.handle(
      new winston.transports.File({
        filename: 'logs/rejections.log',
        maxsize: 5242880,
        maxFiles: 5
      })
    );
  }

  private formatContext(context: LogContext): object {
    const formatted: any = { ...context };
    
    // Convert BigNumber objects to strings
    for (const [key, value] of Object.entries(formatted)) {
      if (value && typeof value === 'object' && value._isBigNumber) {
        formatted[key] = value.toString();
      }
    }

    // Format error objects
    if (formatted.error && formatted.error instanceof Error) {
      formatted.error = {
        name: formatted.error.name,
        message: formatted.error.message,
        stack: formatted.error.stack
      };
    }

    return formatted;
  }

  info(message: string, context: LogContext = {}): void {
    this.logger.info(message, this.formatContext(context));
  }

  warn(message: string, context: LogContext = {}): void {
    this.logger.warn(message, this.formatContext(context));
  }

  error(message: string, context: LogContext = {}): void {
    this.logger.error(message, this.formatContext(context));
  }

  debug(message: string, context: LogContext = {}): void {
    this.logger.debug(message, this.formatContext(context));
  }

  verbose(message: string, context: LogContext = {}): void {
    this.logger.verbose(message, this.formatContext(context));
  }

  // Agent-specific logging methods
  agentStarted(agentId: string, agentType: string, context: LogContext = {}): void {
    this.info(`Agent started: ${agentType}`, {
      agentId,
      agentType,
      event: 'agent_started',
      ...context
    });
  }

  agentStopped(agentId: string, agentType: string, context: LogContext = {}): void {
    this.info(`Agent stopped: ${agentType}`, {
      agentId,
      agentType,
      event: 'agent_stopped',
      ...context
    });
  }

  agentError(agentId: string, agentType: string, error: Error, context: LogContext = {}): void {
    this.error(`Agent error: ${agentType}`, {
      agentId,
      agentType,
      event: 'agent_error',
      error,
      ...context
    });
  }

  decisionMade(agentId: string, decision: string, confidence: number, context: LogContext = {}): void {
    this.info(`Decision made: ${decision}`, {
      agentId,
      decision,
      confidence,
      event: 'decision_made',
      ...context
    });
  }

  executionStarted(agentId: string, executionId: string, action: string, context: LogContext = {}): void {
    this.info(`Execution started: ${action}`, {
      agentId,
      executionId,
      action,
      event: 'execution_started',
      ...context
    });
  }

  executionCompleted(agentId: string, executionId: string, success: boolean, context: LogContext = {}): void {
    const level = success ? 'info' : 'warn';
    this.logger.log(level, `Execution ${success ? 'completed' : 'failed'}`, {
      agentId,
      executionId,
      success,
      event: 'execution_completed',
      ...this.formatContext(context)
    });
  }

  transactionSent(agentId: string, txHash: string, chainId: number, context: LogContext = {}): void {
    this.info(`Transaction sent: ${txHash}`, {
      agentId,
      transactionHash: txHash,
      chainId,
      event: 'transaction_sent',
      ...context
    });
  }

  transactionConfirmed(agentId: string, txHash: string, gasUsed: string, context: LogContext = {}): void {
    this.info(`Transaction confirmed: ${txHash}`, {
      agentId,
      transactionHash: txHash,
      gasUsed,
      event: 'transaction_confirmed',
      ...context
    });
  }

  chainlinkCall(agentId: string, service: string, operation: string, context: LogContext = {}): void {
    this.debug(`Chainlink ${service} call: ${operation}`, {
      agentId,
      chainlinkService: service,
      operation,
      event: 'chainlink_call',
      ...context
    });
  }

  performanceMetric(agentId: string, metric: string, value: number, context: LogContext = {}): void {
    this.debug(`Performance metric: ${metric} = ${value}`, {
      agentId,
      metric,
      value,
      event: 'performance_metric',
      ...context
    });
  }

  securityAlert(agentId: string, alertType: string, severity: string, context: LogContext = {}): void {
    const level = severity === 'critical' ? 'error' : severity === 'high' ? 'warn' : 'info';
    this.logger.log(level, `Security alert: ${alertType}`, {
      agentId,
      alertType,
      severity,
      event: 'security_alert',
      ...this.formatContext(context)
    });
  }

  coordinationEvent(coordinatorId: string, event: string, participants: string[], context: LogContext = {}): void {
    this.info(`Coordination event: ${event}`, {
      coordinatorId,
      event: 'coordination',
      coordinationEvent: event,
      participants,
      ...context
    });
  }

  // Structured logging for analytics
  analytics(event: string, data: Record<string, any>, context: LogContext = {}): void {
    this.info(`Analytics: ${event}`, {
      event: 'analytics',
      analyticsEvent: event,
      analyticsData: data,
      ...context
    });
  }

  // Performance tracking
  startTimer(label: string): () => void {
    const start = Date.now();
    return () => {
      const duration = Date.now() - start;
      this.debug(`Timer ${label}: ${duration}ms`, {
        event: 'timer',
        label,
        duration
      });
      return duration;
    };
  }

  // Create child logger with persistent context
  child(context: LogContext): AgentLogger {
    const childLogger = new AgentLogger();
    const originalFormat = childLogger.logger.format;
    
    childLogger.logger.format = winston.format.combine(
      originalFormat,
      winston.format.printf((info) => {
        return JSON.stringify({
          ...info,
          ...this.formatContext(context)
        });
      })
    );

    return childLogger;
  }

  // Flush logs (useful for graceful shutdown)
  async flush(): Promise<void> {
    return new Promise((resolve) => {
      this.logger.on('finish', resolve);
      this.logger.end();
    });
  }
}

export const logger = new AgentLogger();
export default logger;
