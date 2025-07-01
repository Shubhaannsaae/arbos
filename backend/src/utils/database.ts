import { Pool, PoolClient, PoolConfig } from 'pg';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { logger } from './logger';
import * as crypto from 'crypto';

interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl: boolean | object;
  poolSize: number;
  connectionTimeout: number;
  idleTimeout: number;
  queryTimeout: number;
  statementTimeout: number;
  schema: string;
  applicationName: string;
}

interface QueryOptions {
  timeout?: number;
  retries?: number;
  useTransaction?: boolean;
  isolation?: 'READ UNCOMMITTED' | 'READ COMMITTED' | 'REPEATABLE READ' | 'SERIALIZABLE';
}

interface TransactionCallback<T> {
  (client: PoolClient): Promise<T>;
}

interface HealthCheckResult {
  status: 'healthy' | 'unhealthy';
  latency: number;
  activeConnections: number;
  idleConnections: number;
  totalConnections: number;
  timestamp: Date;
}

interface QueryMetrics {
  query: string;
  duration: number;
  rows: number;
  success: boolean;
  timestamp: Date;
  userId?: string;
}

class DatabaseManager {
  private pool: Pool;
  private config: DatabaseConfig;
  private isConnected: boolean = false;
  private queryMetrics: QueryMetrics[] = [];
  private readonly metricsRetention = 1000; // Keep last 1000 queries

  constructor() {
    this.config = this.loadConfiguration();
    this.pool = this.createPool();
    this.setupEventHandlers();
  }

  /**
   * Load database configuration from environment variables
   */
  private loadConfiguration(): DatabaseConfig {
    return {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'arbos',
      username: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || '',
      ssl: this.parseSSLConfig(),
      poolSize: parseInt(process.env.DB_POOL_SIZE || '20'),
      connectionTimeout: parseInt(process.env.DB_CONNECTION_TIMEOUT || '30000'),
      idleTimeout: parseInt(process.env.DB_IDLE_TIMEOUT || '10000'),
      queryTimeout: parseInt(process.env.DB_QUERY_TIMEOUT || '60000'),
      statementTimeout: parseInt(process.env.DB_STATEMENT_TIMEOUT || '30000'),
      schema: process.env.DB_SCHEMA || 'public',
      applicationName: process.env.DB_APPLICATION_NAME || 'arbos-backend'
    };
  }

  /**
   * Parse SSL configuration from environment
   */
  private parseSSLConfig(): boolean | object {
    if (process.env.DB_SSL === 'false') {
      return false;
    }

    if (process.env.DB_SSL === 'true') {
      return { rejectUnauthorized: false };
    }

    if (process.env.DB_SSL_CERT && process.env.DB_SSL_KEY) {
      return {
        cert: Buffer.from(process.env.DB_SSL_CERT, 'base64'),
        key: Buffer.from(process.env.DB_SSL_KEY, 'base64'),
        ca: process.env.DB_SSL_CA ? Buffer.from(process.env.DB_SSL_CA, 'base64') : undefined,
        rejectUnauthorized: process.env.NODE_ENV === 'production'
      };
    }

    return process.env.NODE_ENV === 'production';
  }

  /**
   * Create PostgreSQL connection pool
   */
  private createPool(): Pool {
    const poolConfig: PoolConfig = {
      host: this.config.host,
      port: this.config.port,
      database: this.config.database,
      user: this.config.username,
      password: this.config.password,
      ssl: this.config.ssl,
      max: this.config.poolSize,
      min: Math.ceil(this.config.poolSize / 4), // 25% of max pool size as minimum
      connectionTimeoutMillis: this.config.connectionTimeout,
      idleTimeoutMillis: this.config.idleTimeout,
      query_timeout: this.config.queryTimeout,
      statement_timeout: this.config.statementTimeout,
      keepAlive: true,
      keepAliveInitialDelayMillis: 10000,
      application_name: this.config.applicationName
    };

    return new Pool(poolConfig);
  }

  /**
   * Setup pool event handlers
   */
  private setupEventHandlers(): void {
    this.pool.on('connect', (client) => {
      logger.info('New database client connected', {
        totalCount: this.pool.totalCount,
        idleCount: this.pool.idleCount,
        waitingCount: this.pool.waitingCount
      });
      
      // Set schema for the client
      client.query(`SET search_path TO ${this.config.schema}`);
    });

    this.pool.on('acquire', (client) => {
      logger.debug('Database client acquired from pool');
    });

    this.pool.on('remove', (client) => {
      logger.debug('Database client removed from pool', {
        totalCount: this.pool.totalCount,
        idleCount: this.pool.idleCount
      });
    });

    this.pool.on('error', (err, client) => {
      logger.error('Database pool error', {
        error: err.message,
        stack: err.stack,
        totalCount: this.pool.totalCount,
        idleCount: this.pool.idleCount
      });
    });
  }

  /**
   * Initialize database connection and run migrations
   */
  public async initialize(): Promise<void> {
    try {
      // Test connection
      await this.testConnection();
      
      // Run migrations
      await this.runMigrations();
      
      // Create indexes
      await this.createIndexes();
      
      // Setup monitoring
      this.setupMonitoring();
      
      this.isConnected = true;
      logger.info('Database initialized successfully', {
        host: this.config.host,
        database: this.config.database,
        poolSize: this.config.poolSize,
        schema: this.config.schema
      });
    } catch (error) {
      logger.error('Failed to initialize database', { error });
      throw error;
    }
  }

  /**
   * Test database connection
   */
  private async testConnection(): Promise<void> {
    const client = await this.pool.connect();
    try {
      const result = await client.query('SELECT NOW() as timestamp, version() as version');
      logger.info('Database connection test successful', {
        timestamp: result.rows[0].timestamp,
        version: result.rows[0].version.split(' ')[0]
      });
    } finally {
      client.release();
    }
  }

  /**
   * Run database migrations using Drizzle ORM
   */
  private async runMigrations(): Promise<void> {
    try {
      const connectionString = this.buildConnectionString();
      const sql = postgres(connectionString, { max: 1 });
      const db = drizzle(sql);

      await migrate(db, { migrationsFolder: './src/database/migrations' });
      
      await sql.end();
      logger.info('Database migrations completed successfully');
    } catch (error) {
      logger.error('Database migration failed', { error });
      throw error;
    }
  }

  /**
   * Create database indexes for performance
   */
  private async createIndexes(): Promise<void> {
    const indexes = [
      // User indexes
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_wallet_address ON users (wallet_address)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users (email) WHERE email IS NOT NULL',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users (created_at)',
      
      // Portfolio indexes
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_id ON portfolios (user_id)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_chain_id ON portfolios (chain_id)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_is_active ON portfolios (is_active)',
      
      // Transaction indexes
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_user_id ON transactions (user_id)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_tx_hash ON transactions (tx_hash)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_chain_id ON transactions (chain_id)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_timestamp ON transactions (timestamp)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_status ON transactions (status)',
      
      // Agent indexes
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_user_id ON agents (user_id)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_type ON agents (type)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status ON agents (status)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_is_enabled ON agents (is_enabled)',
      
      // Arbitrage opportunity indexes
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arbitrage_opportunities_user_id ON arbitrage_opportunities (user_id)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arbitrage_opportunities_token_pair ON arbitrage_opportunities (token_pair)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arbitrage_opportunities_status ON arbitrage_opportunities (status)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arbitrage_opportunities_detected_at ON arbitrage_opportunities (detected_at)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arbitrage_opportunities_net_profit ON arbitrage_opportunities (net_profit)',
      
      // Composite indexes for complex queries
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_arbitrage_user_status_profit ON arbitrage_opportunities (user_id, status, net_profit DESC)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_user_chain_timestamp ON transactions (user_id, chain_id, timestamp DESC)',
      'CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolios_user_active_updated ON portfolios (user_id, is_active, updated_at DESC)'
    ];

    for (const indexQuery of indexes) {
      try {
        await this.query(indexQuery, [], { timeout: 300000 }); // 5 minute timeout for index creation
        logger.debug('Index created successfully', { query: indexQuery.substring(0, 100) + '...' });
      } catch (error: any) {
        if (error.code === '42P07') {
          logger.debug('Index already exists, skipping', { query: indexQuery.substring(0, 100) + '...' });
        } else {
          logger.warn('Failed to create index', { 
            error: error.message, 
            query: indexQuery.substring(0, 100) + '...' 
          });
        }
      }
    }
  }

  /**
   * Setup database monitoring
   */
  private setupMonitoring(): void {
    // Monitor pool statistics every 30 seconds
    setInterval(() => {
      const stats = {
        totalConnections: this.pool.totalCount,
        idleConnections: this.pool.idleCount,
        waitingCount: this.pool.waitingCount,
        timestamp: new Date()
      };
      
      logger.debug('Database pool statistics', stats);
      
      // Alert if pool is running low
      const utilizationRate = (this.pool.totalCount - this.pool.idleCount) / this.config.poolSize;
      if (utilizationRate > 0.8) {
        logger.warn('High database pool utilization', {
          utilizationRate: Math.round(utilizationRate * 100),
          ...stats
        });
      }
    }, 30000);

    // Clean up old query metrics every 5 minutes
    setInterval(() => {
      if (this.queryMetrics.length > this.metricsRetention) {
        this.queryMetrics = this.queryMetrics.slice(-this.metricsRetention);
      }
    }, 5 * 60 * 1000);
  }

  /**
   * Execute a database query
   */
  public async query<T = any>(
    text: string,
    params: any[] = [],
    options: QueryOptions = {}
  ): Promise<T[]> {
    const startTime = Date.now();
    const queryId = crypto.randomUUID();
    let client: PoolClient | undefined;

    try {
      logger.debug('Executing database query', {
        queryId,
        query: text.substring(0, 200) + (text.length > 200 ? '...' : ''),
        paramCount: params.length
      });

      client = await this.pool.connect();
      
      // Set query timeout if specified
      if (options.timeout) {
        await client.query(`SET statement_timeout = ${options.timeout}`);
      }

      const result = await client.query(text, params);
      const duration = Date.now() - startTime;

      // Record metrics
      this.recordQueryMetrics({
        query: text,
        duration,
        rows: result.rowCount || 0,
        success: true,
        timestamp: new Date(),
        userId: options.userId
      });

      logger.debug('Query executed successfully', {
        queryId,
        duration,
        rowCount: result.rowCount
      });

      return result.rows;
    } catch (error: any) {
      const duration = Date.now() - startTime;
      
      // Record metrics for failed query
      this.recordQueryMetrics({
        query: text,
        duration,
        rows: 0,
        success: false,
        timestamp: new Date(),
        userId: options.userId
      });

      logger.error('Database query failed', {
        queryId,
        error: error.message,
        duration,
        query: text.substring(0, 200) + '...',
        code: error.code
      });

      throw error;
    } finally {
      if (client) {
        client.release();
      }
    }
  }

  /**
   * Execute queries within a transaction
   */
  public async transaction<T>(
    callback: TransactionCallback<T>,
    options: QueryOptions = {}
  ): Promise<T> {
    const client = await this.pool.connect();
    const transactionId = crypto.randomUUID();

    try {
      logger.debug('Starting database transaction', { transactionId });

      // Begin transaction with isolation level
      const isolationLevel = options.isolation || 'READ COMMITTED';
      await client.query(`BEGIN ISOLATION LEVEL ${isolationLevel}`);

      const result = await callback(client);

      await client.query('COMMIT');
      
      logger.debug('Database transaction committed', { transactionId });
      return result;
    } catch (error: any) {
      await client.query('ROLLBACK');
      
      logger.error('Database transaction rolled back', {
        transactionId,
        error: error.message
      });
      
      throw error;
    } finally {
      client.release();
    }
  }

  /**
   * Execute a query with automatic retries
   */
  public async queryWithRetry<T = any>(
    text: string,
    params: any[] = [],
    options: QueryOptions = {}
  ): Promise<T[]> {
    const maxRetries = options.retries || 3;
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await this.query<T>(text, params, options);
      } catch (error: any) {
        lastError = error;
        
        // Don't retry on syntax errors or constraint violations
        if (error.code && ['42601', '23505', '23503', '23514'].includes(error.code)) {
          throw error;
        }

        if (attempt < maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000); // Exponential backoff with max 5s
          logger.warn(`Query failed, retrying in ${delay}ms`, {
            attempt,
            maxRetries,
            error: error.message
          });
          
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError!;
  }

  /**
   * Get database health status
   */
  public async healthCheck(): Promise<HealthCheckResult> {
    const startTime = Date.now();
    
    try {
      const client = await this.pool.connect();
      
      try {
        await client.query('SELECT 1');
        const latency = Date.now() - startTime;
        
        return {
          status: 'healthy',
          latency,
          activeConnections: this.pool.totalCount - this.pool.idleCount,
          idleConnections: this.pool.idleCount,
          totalConnections: this.pool.totalCount,
          timestamp: new Date()
        };
      } finally {
        client.release();
      }
    } catch (error: any) {
      logger.error('Database health check failed', { error: error.message });
      
      return {
        status: 'unhealthy',
        latency: Date.now() - startTime,
        activeConnections: 0,
        idleConnections: 0,
        totalConnections: 0,
        timestamp: new Date()
      };
    }
  }

  /**
   * Get query performance metrics
   */
  public getQueryMetrics(): QueryMetrics[] {
    return [...this.queryMetrics];
  }

  /**
   * Get connection pool statistics
   */
  public getPoolStats(): any {
    return {
      totalCount: this.pool.totalCount,
      idleCount: this.pool.idleCount,
      waitingCount: this.pool.waitingCount,
      config: {
        max: this.config.poolSize,
        min: Math.ceil(this.config.poolSize / 4),
        connectionTimeout: this.config.connectionTimeout,
        idleTimeout: this.config.idleTimeout
      }
    };
  }

  /**
   * Record query metrics
   */
  private recordQueryMetrics(metrics: QueryMetrics): void {
    this.queryMetrics.push(metrics);
    
    // Alert on slow queries
    if (metrics.duration > 5000) { // 5 seconds
      logger.warn('Slow database query detected', {
        query: metrics.query.substring(0, 200) + '...',
        duration: metrics.duration,
        success: metrics.success
      });
    }
  }

  /**
   * Build connection string for migrations
   */
  private buildConnectionString(): string {
    const sslParam = this.config.ssl ? '?sslmode=require' : '';
    return `postgresql://${this.config.username}:${this.config.password}@${this.config.host}:${this.config.port}/${this.config.database}${sslParam}`;
  }

  /**
   * Close all database connections
   */
  public async close(): Promise<void> {
    try {
      await this.pool.end();
      this.isConnected = false;
      logger.info('Database connections closed');
    } catch (error: any) {
      logger.error('Error closing database connections', { error: error.message });
      throw error;
    }
  }

  /**
   * Check if database is connected
   */
  public isHealthy(): boolean {
    return this.isConnected && this.pool.totalCount > 0;
  }
}

// Create singleton instance
export const database = new DatabaseManager();

// Export for testing
export { DatabaseManager };

// Export types
export type { DatabaseConfig, QueryOptions, HealthCheckResult, QueryMetrics };

// Default export
export default database;
