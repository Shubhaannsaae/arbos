import winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';
import { ElasticsearchTransport } from 'winston-elasticsearch';
import { Client } from '@elastic/elasticsearch';
import path from 'path';

// Define log levels based on RFC5424
const logLevels = {
  error: 0,
  warn: 1,
  info: 2,
  http: 3,
  verbose: 4,
  debug: 5,
  silly: 6
};

const logColors = {
  error: 'red',
  warn: 'yellow',
  info: 'green',
  http: 'magenta',
  verbose: 'grey',
  debug: 'blue',
  silly: 'rainbow'
};

interface LoggerConfig {
  level: string;
  environment: string;
  serviceName: string;
  version: string;
  enableConsole: boolean;
  enableFile: boolean;
  enableElasticsearch: boolean;
  logDirectory: string;
  maxFiles: string;
  maxSize: string;
  elasticsearchUrl?: string;
  elasticsearchIndex?: string;
  enableChainlinkIntegration: boolean;
}

interface ChainlinkLogContext {
  chainId?: number;
  networkName?: string;
  contractAddress?: string;
  transactionHash?: string;
  blockNumber?: number;
  gasUsed?: string;
  gasPrice?: string;
  jobId?: string;
  requestId?: string;
  nodeAddress?: string;
  oracleAddress?: string;
}

class Logger {
  private logger: winston.Logger;
  private config: LoggerConfig;
  private elasticsearchClient?: Client;

  constructor() {
    this.config = this.loadConfiguration();
    this.initializeElasticsearch();
    this.logger = this.createLogger();
    this.setupUncaughtExceptionHandlers();
    this.setupProcessHandlers();
  }

  /**
   * Load logger configuration from environment variables
   */
  private loadConfiguration(): LoggerConfig {
    return {
      level: process.env.LOG_LEVEL || 'info',
      environment: process.env.NODE_ENV || 'development',
      serviceName: process.env.SERVICE_NAME || 'arbos-backend',
      version: process.env.SERVICE_VERSION || '1.0.0',
      enableConsole: process.env.LOG_ENABLE_CONSOLE !== 'false',
      enableFile: process.env.LOG_ENABLE_FILE !== 'false',
      enableElasticsearch: process.env.LOG_ENABLE_ELASTICSEARCH === 'true',
      logDirectory: process.env.LOG_DIRECTORY || './logs',
      maxFiles: process.env.LOG_MAX_FILES || '14d',
      maxSize: process.env.LOG_MAX_SIZE || '20m',
      elasticsearchUrl: process.env.ELASTICSEARCH_URL,
      elasticsearchIndex: process.env.ELASTICSEARCH_INDEX || 'arbos-logs',
      enableChainlinkIntegration: process.env.ENABLE_CHAINLINK_LOGGING !== 'false'
    };
  }

  /**
   * Initialize Elasticsearch client if enabled
   */
  private initializeElasticsearch(): void {
    if (this.config.enableElasticsearch && this.config.elasticsearchUrl) {
      try {
        this.elasticsearchClient = new Client({
          node: this.config.elasticsearchUrl,
          auth: {
            username: process.env.ELASTICSEARCH_USERNAME || '',
            password: process.env.ELASTICSEARCH_PASSWORD || ''
          },
          tls: {
            rejectUnauthorized: process.env.NODE_ENV === 'production'
          }
        });
      } catch (error) {
        console.error('Failed to initialize Elasticsearch client:', error);
      }
    }
  }

  /**
   * Create Winston logger instance with multiple transports
   */
  private createLogger(): winston.Logger {
    winston.addColors(logColors);

    const transports: winston.transport[] = [];

    // Console transport for development
    if (this.config.enableConsole) {
      transports.push(
        new winston.transports.Console({
          level: this.config.level,
          format: winston.format.combine(
            winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss:ms' }),
            winston.format.colorize({ all: true }),
            winston.format.printf(this.consoleFormat)
          )
        })
      );
    }

    // File transport with rotation
    if (this.config.enableFile) {
      // Error log file
      transports.push(
        new DailyRotateFile({
          filename: path.join(this.config.logDirectory, 'error-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          level: 'error',
          maxFiles: this.config.maxFiles,
          maxSize: this.config.maxSize,
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.errors({ stack: true }),
            winston.format.json()
          )
        })
      );

      // Combined log file
      transports.push(
        new DailyRotateFile({
          filename: path.join(this.config.logDirectory, 'combined-%DATE%.log'),
          datePattern: 'YYYY-MM-DD',
          maxFiles: this.config.maxFiles,
          maxSize: this.config.maxSize,
          format: winston.format.combine(
            winston.format.timestamp(),
            winston.format.errors({ stack: true }),
            winston.format.json()
          )
        })
      );

      // Chainlink specific log file
      if (this.config.enableChainlinkIntegration) {
        transports.push(
          new DailyRotateFile({
            filename: path.join(this.config.logDirectory, 'chainlink-%DATE%.log'),
            datePattern: 'YYYY-MM-DD',
            maxFiles: this.config.maxFiles,
            maxSize: this.config.maxSize,
            format: winston.format.combine(
              winston.format.timestamp(),
              winston.format.printf((info) => {
                if (info.chainlink || info.chainId || info.contractAddress) {
                  return JSON.stringify(info);
                }
                return null;
              }),
              winston.format((info) => info !== null ? info : false)()
            )
          })
        );
      }
    }

    // Elasticsearch transport for production
    if (this.config.enableElasticsearch && this.elasticsearchClient) {
      transports.push(
        new ElasticsearchTransport({
          client: this.elasticsearchClient,
          level: this.config.level,
          index: this.config.elasticsearchIndex,
          typeName: '_doc',
          transformer: this.elasticsearchTransformer.bind(this)
        })
      );
    }

    return winston.createLogger({
      level: this.config.level,
      levels: logLevels,
      defaultMeta: {
        service: this.config.serviceName,
        version: this.config.version,
        environment: this.config.environment,
        timestamp: () => new Date().toISOString()
      },
      transports,
      exitOnError: false
    });
  }

  /**
   * Console log format for development
   */
  private consoleFormat = (info: winston.Logform.TransformableInfo): string => {
    const { timestamp, level, message, service, ...meta } = info;
    const metaString = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : '';
    return `${timestamp} [${service}] ${level}: ${message} ${metaString}`;
  };

  /**
   * Elasticsearch document transformer
   */
  private elasticsearchTransformer(logData: any): any {
    return {
      '@timestamp': new Date().toISOString(),
      severity: logData.level,
      message: logData.message,
      service: this.config.serviceName,
      environment: this.config.environment,
      version: this.config.version,
      ...logData.meta
    };
  }

  /**
   * Setup uncaught exception handlers
   */
  private setupUncaughtExceptionHandlers(): void {
    this.logger.exceptions.handle(
      new winston.transports.File({
        filename: path.join(this.config.logDirectory, 'exceptions.log'),
        format: winston.format.combine(
          winston.format.timestamp(),
          winston.format.errors({ stack: true }),
          winston.format.json()
        )
      })
    );

    this.logger.rejections.handle(
      new winston.transports.File({
        filename: path.join(this.config.logDirectory, 'rejections.log'),
        format: winston.format.combine(
          winston.format.timestamp(),
          winston.format.errors({ stack: true }),
          winston.format.json()
        )
      })
    );
  }

  /**
   * Setup process event handlers
   */
  private setupProcessHandlers(): void {
    process.on('SIGINT', () => {
      this.info('Received SIGINT signal, shutting down gracefully');
      this.gracefulShutdown();
    });

    process.on('SIGTERM', () => {
      this.info('Received SIGTERM signal, shutting down gracefully');
      this.gracefulShutdown();
    });

    process.on('uncaughtException', (error: Error) => {
      this.error('Uncaught exception occurred', { error: error.message, stack: error.stack });
    });

    process.on('unhandledRejection', (reason: any, promise: Promise<any>) => {
      this.error('Unhandled promise rejection', { reason, promise: promise.toString() });
    });
  }

  /**
   * Graceful shutdown
   */
  private gracefulShutdown(): void {
    this.logger.end(() => {
      process.exit(0);
    });
  }

  // Standard logging methods
  public error(message: string, meta?: any): void {
    this.logger.error(message, this.formatMeta(meta));
  }

  public warn(message: string, meta?: any): void {
    this.logger.warn(message, this.formatMeta(meta));
  }

  public info(message: string, meta?: any): void {
    this.logger.info(message, this.formatMeta(meta));
  }

  public http(message: string, meta?: any): void {
    this.logger.http(message, this.formatMeta(meta));
  }

  public verbose(message: string, meta?: any): void {
    this.logger.verbose(message, this.formatMeta(meta));
  }

  public debug(message: string, meta?: any): void {
    this.logger.debug(message, this.formatMeta(meta));
  }

  public silly(message: string, meta?: any): void {
    this.logger.silly(message, this.formatMeta(meta));
  }

  // Chainlink-specific logging methods
  public chainlinkInfo(message: string, context: ChainlinkLogContext = {}): void {
    this.info(message, {
      chainlink: true,
      ...context,
      category: 'chainlink_operation'
    });
  }

  public chainlinkError(message: string, error: Error, context: ChainlinkLogContext = {}): void {
    this.error(message, {
      chainlink: true,
      error: {
        message: error.message,
        stack: error.stack,
        name: error.name
      },
      ...context,
      category: 'chainlink_error'
    });
  }

  public chainlinkDebug(message: string, context: ChainlinkLogContext = {}): void {
    this.debug(message, {
      chainlink: true,
      ...context,
      category: 'chainlink_debug'
    });
  }

  // Transaction logging
  public logTransaction(
    type: 'sent' | 'received' | 'confirmed' | 'failed',
    txHash: string,
    context: ChainlinkLogContext = {}
  ): void {
    this.info(`Transaction ${type}`, {
      transactionHash: txHash,
      transactionType: type,
      chainlink: true,
      ...context,
      category: 'blockchain_transaction'
    });
  }

  // Oracle request logging
  public logOracleRequest(
    requestId: string,
    jobId: string,
    context: ChainlinkLogContext = {}
  ): void {
    this.info('Oracle request initiated', {
      requestId,
      jobId,
      chainlink: true,
      ...context,
      category: 'oracle_request'
    });
  }

  // CCIP logging
  public logCCIPMessage(
    messageId: string,
    sourceChain: number,
    targetChain: number,
    status: 'sent' | 'received' | 'executed' | 'failed',
    context: ChainlinkLogContext = {}
  ): void {
    this.info(`CCIP message ${status}`, {
      messageId,
      sourceChain,
      targetChain,
      ccipStatus: status,
      chainlink: true,
      ...context,
      category: 'ccip_message'
    });
  }

  // VRF logging
  public logVRFRequest(
    requestId: string,
    status: 'requested' | 'fulfilled' | 'failed',
    context: ChainlinkLogContext = {}
  ): void {
    this.info(`VRF request ${status}`, {
      requestId,
      vrfStatus: status,
      chainlink: true,
      ...context,
      category: 'vrf_request'
    });
  }

  // Automation logging
  public logAutomationUpkeep(
    upkeepId: string,
    status: 'performed' | 'skipped' | 'failed',
    context: ChainlinkLogContext = {}
  ): void {
    this.info(`Automation upkeep ${status}`, {
      upkeepId,
      automationStatus: status,
      chainlink: true,
      ...context,
      category: 'automation_upkeep'
    });
  }

  // Functions logging
  public logFunctionsRequest(
    requestId: string,
    donId: string,
    status: 'sent' | 'fulfilled' | 'failed',
    context: ChainlinkLogContext = {}
  ): void {
    this.info(`Functions request ${status}`, {
      requestId,
      donId,
      functionsStatus: status,
      chainlink: true,
      ...context,
      category: 'functions_request'
    });
  }

  // Performance logging
  public logPerformance(
    operation: string,
    duration: number,
    success: boolean,
    context: any = {}
  ): void {
    this.info(`Performance: ${operation}`, {
      operation,
      duration,
      success,
      ...context,
      category: 'performance'
    });
  }

  // Security logging
  public logSecurityEvent(
    event: string,
    severity: 'low' | 'medium' | 'high' | 'critical',
    context: any = {}
  ): void {
    const logMethod = severity === 'critical' || severity === 'high' ? 'error' : 
                     severity === 'medium' ? 'warn' : 'info';
    
    this[logMethod](`Security event: ${event}`, {
      securityEvent: event,
      severity,
      ...context,
      category: 'security'
    });
  }

  // Audit logging
  public logAuditEvent(
    action: string,
    userId: string,
    resource: string,
    result: 'success' | 'failure',
    context: any = {}
  ): void {
    this.info(`Audit: ${action}`, {
      auditAction: action,
      userId,
      resource,
      result,
      ...context,
      category: 'audit',
      timestamp: new Date().toISOString()
    });
  }

  // Critical logging for immediate attention
  public critical(message: string, meta?: any): void {
    this.error(`[CRITICAL] ${message}`, {
      ...this.formatMeta(meta),
      critical: true,
      alertRequired: true
    });
  }

  /**
   * Format metadata for logging
   */
  private formatMeta(meta?: any): any {
    if (!meta) return {};

    // Sanitize sensitive information
    const sanitized = this.sanitizeSensitiveData(meta);
    
    return {
      ...sanitized,
      pid: process.pid,
      memory: process.memoryUsage(),
      uptime: process.uptime()
    };
  }

  /**
   * Sanitize sensitive data from logs
   */
  private sanitizeSensitiveData(obj: any): any {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }

    const sensitiveKeys = [
      'password',
      'privateKey',
      'mnemonic',
      'secret',
      'token',
      'apiKey',
      'authorization',
      'cookie',
      'session'
    ];

    const sanitized = { ...obj };

    for (const key in sanitized) {
      if (sensitiveKeys.some(sensitiveKey => 
        key.toLowerCase().includes(sensitiveKey.toLowerCase())
      )) {
        sanitized[key] = '[REDACTED]';
      } else if (typeof sanitized[key] === 'object') {
        sanitized[key] = this.sanitizeSensitiveData(sanitized[key]);
      }
    }

    return sanitized;
  }

  /**
   * Get logger statistics
   */
  public getStats(): any {
    return {
      level: this.config.level,
      environment: this.config.environment,
      service: this.config.serviceName,
      transports: this.logger.transports.length,
      elasticsearchEnabled: this.config.enableElasticsearch,
      chainlinkIntegration: this.config.enableChainlinkIntegration
    };
  }

  /**
   * Update log level dynamically
   */
  public setLevel(level: string): void {
    if (level in logLevels) {
      this.logger.level = level;
      this.config.level = level;
      this.info(`Log level updated to: ${level}`);
    } else {
      this.error(`Invalid log level: ${level}`);
    }
  }

  /**
   * Create child logger with additional context
   */
  public child(defaultMeta: any): winston.Logger {
    return this.logger.child(defaultMeta);
  }

  /**
   * Flush all logs and close transports
   */
  public async flush(): Promise<void> {
    return new Promise((resolve) => {
      this.logger.end(() => {
        resolve();
      });
    });
  }
}

// Create singleton logger instance
export const logger = new Logger();

// Export logger class for testing
export { Logger };

// Export logging interfaces
export type { ChainlinkLogContext };

// Default export
export default logger;
