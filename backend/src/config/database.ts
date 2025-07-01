import { Pool, PoolConfig } from 'pg';
import { logger } from '../utils/logger';

interface DatabaseConnectionConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl: boolean | object;
  poolConfig: {
    min: number;
    max: number;
    acquireTimeoutMillis: number;
    idleTimeoutMillis: number;
    reapIntervalMillis: number;
    createTimeoutMillis: number;
    destroyTimeoutMillis: number;
    createRetryIntervalMillis: number;
  };
  connectionString?: string;
  schema: string;
  migrationConfig: {
    directory: string;
    tableName: string;
    schemaName: string;
  };
}

class DatabaseConfig {
  private config: DatabaseConnectionConfig;

  constructor() {
    this.config = this.loadDatabaseConfiguration();
    this.validateConfiguration();
  }

  /**
   * Load database configuration from environment variables
   */
  private loadDatabaseConfiguration(): DatabaseConnectionConfig {
    const sslConfig = this.parseSSLConfiguration();
    
    return {
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      database: process.env.DB_NAME || 'arbos',
      username: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || '',
      ssl: sslConfig,
      poolConfig: {
        min: parseInt(process.env.DB_POOL_MIN || '5'),
        max: parseInt(process.env.DB_POOL_MAX || '20'),
        acquireTimeoutMillis: parseInt(process.env.DB_ACQUIRE_TIMEOUT || '60000'),
        idleTimeoutMillis: parseInt(process.env.DB_IDLE_TIMEOUT || '10000'),
        reapIntervalMillis: parseInt(process.env.DB_REAP_INTERVAL || '1000'),
        createTimeoutMillis: parseInt(process.env.DB_CREATE_TIMEOUT || '30000'),
        destroyTimeoutMillis: parseInt(process.env.DB_DESTROY_TIMEOUT || '5000'),
        createRetryIntervalMillis: parseInt(process.env.DB_CREATE_RETRY_INTERVAL || '200')
      },
      connectionString: process.env.DATABASE_URL,
      schema: process.env.DB_SCHEMA || 'public',
      migrationConfig: {
        directory: './src/database/migrations',
        tableName: process.env.DB_MIGRATION_TABLE || 'knex_migrations',
        schemaName: process.env.DB_SCHEMA || 'public'
      }
    };
  }

  /**
   * Parse SSL configuration from environment variables
   */
  private parseSSLConfiguration(): boolean | object {
    const sslMode = process.env.DB_SSL_MODE;
    
    if (sslMode === 'disable') {
      return false;
    }

    if (sslMode === 'require') {
      return { rejectUnauthorized: false };
    }

    if (sslMode === 'verify-ca' || sslMode === 'verify-full') {
      return {
        rejectUnauthorized: true,
        ca: process.env.DB_SSL_CA,
        cert: process.env.DB_SSL_CERT,
        key: process.env.DB_SSL_KEY
      };
    }

    // Default SSL configuration based on environment
    return process.env.NODE_ENV === 'production' ? { rejectUnauthorized: true } : false;
  }

  /**
   * Validate database configuration
   */
  private validateConfiguration(): void {
    const required = ['host', 'port', 'database', 'username'];
    const missing = required.filter(key => !this.config[key as keyof DatabaseConnectionConfig]);

    if (missing.length > 0) {
      throw new Error(`Missing required database configuration: ${missing.join(', ')}`);
    }

    if (this.config.port < 1 || this.config.port > 65535) {
      throw new Error('Database port must be between 1 and 65535');
    }

    if (this.config.poolConfig.min < 0 || this.config.poolConfig.max < this.config.poolConfig.min) {
      throw new Error('Invalid pool configuration: max must be greater than or equal to min');
    }
  }

  /**
   * Get PostgreSQL pool configuration
   */
  public getPoolConfig(): PoolConfig {
    const baseConfig: PoolConfig = {
      host: this.config.host,
      port: this.config.port,
      database: this.config.database,
      user: this.config.username,
      password: this.config.password,
      ssl: this.config.ssl,
      min: this.config.poolConfig.min,
      max: this.config.poolConfig.max,
      acquireTimeoutMillis: this.config.poolConfig.acquireTimeoutMillis,
      idleTimeoutMillis: this.config.poolConfig.idleTimeoutMillis,
      reapIntervalMillis: this.config.poolConfig.reapIntervalMillis,
      createTimeoutMillis: this.config.poolConfig.createTimeoutMillis,
      createRetryIntervalMillis: this.config.poolConfig.createRetryIntervalMillis,
      application_name: 'arbos-backend',
      keepAlive: true,
      keepAliveInitialDelayMillis: 10000
    };

    // Use connection string if provided (overrides individual parameters)
    if (this.config.connectionString) {
      return {
        ...baseConfig,
        connectionString: this.config.connectionString
      };
    }

    return baseConfig;
  }

  /**
   * Get database connection string
   */
  public getConnectionString(): string {
    if (this.config.connectionString) {
      return this.config.connectionString;
    }

    const { host, port, database, username, password } = this.config;
    const sslParam = this.config.ssl ? '?sslmode=require' : '';
    
    return `postgresql://${username}:${password}@${host}:${port}/${database}${sslParam}`;
  }

  /**
   * Get migration configuration
   */
  public getMigrationConfig(): any {
    return {
      client: 'postgresql',
      connection: this.getConnectionString(),
      migrations: {
        directory: this.config.migrationConfig.directory,
        tableName: this.config.migrationConfig.tableName,
        schemaName: this.config.migrationConfig.schemaName
      },
      pool: {
        min: 2,
        max: 10
      }
    };
  }

  /**
   * Get schema name
   */
  public getSchema(): string {
    return this.config.schema;
  }

  /**
   * Check if SSL is enabled
   */
  public isSSLEnabled(): boolean {
    return Boolean(this.config.ssl);
  }

  /**
   * Get database configuration for logging (without sensitive data)
   */
  public getConfigForLogging(): any {
    return {
      host: this.config.host,
      port: this.config.port,
      database: this.config.database,
      username: this.config.username,
      schema: this.config.schema,
      ssl: Boolean(this.config.ssl),
      poolMin: this.config.poolConfig.min,
      poolMax: this.config.poolConfig.max
    };
  }
}

// Create singleton instance
export const databaseConfig = new DatabaseConfig();

// Export configuration class
export { DatabaseConfig };

// Export types
export type { DatabaseConnectionConfig };

// Default export
export default databaseConfig;
