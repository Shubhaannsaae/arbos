import express, { Application, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import { createProxyMiddleware } from 'http-proxy-middleware';
import swaggerUi from 'swagger-ui-express';
import { OpenAPIV3 } from 'openapi-types';

// Internal imports
import { logger } from './utils/logger';
import { database } from './utils/database';
import { databaseConfig } from './config/database';
import { chainlinkConfig } from './config/chainlink';
import { web3Config } from './config/web3';
import { agentConfigService } from './config/agents';

// Middleware imports
import { rateLimitStatus } from './middleware/rateLimiter';
import authMiddleware from './middleware/auth';

// Route imports
import agentRoutes from './routes/agents';
import arbitrageRoutes from './routes/arbitrage';
import portfolioRoutes from './routes/portfolio';
import securityRoutes from './routes/security';
import userRoutes from './routes/users';

// Types
interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: Date;
  version: string;
  environment: string;
  services: {
    database: {
      status: 'healthy' | 'unhealthy';
      latency?: number;
      connections?: any;
    };
    chainlink: {
      status: 'healthy' | 'unhealthy';
      networks: string[];
      services: string[];
    };
    web3: {
      status: 'healthy' | 'unhealthy';
      chains: number[];
      providers: number;
    };
    agents: {
      status: 'healthy' | 'unhealthy';
      types: number;
      models: number;
    };
  };
  uptime: number;
  memory: NodeJS.MemoryUsage;
}

class ArbOSApplication {
  public app: Application;
  private startTime: Date;
  private port: number;
  private environment: string;

  constructor() {
    this.app = express();
    this.startTime = new Date();
    this.port = parseInt(process.env.PORT || '3000');
    this.environment = process.env.NODE_ENV || 'development';

    this.initializeMiddleware();
    this.initializeRoutes();
    this.initializeErrorHandling();
    this.initializeHealthChecks();
  }

  /**
   * Initialize essential middleware
   */
  private initializeMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'", "https://api.coingecko.com", "https://*.alchemy.com"]
        }
      },
      hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
      }
    }));

    // CORS configuration
    this.app.use(cors({
      origin: this.getAllowedOrigins(),
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
      allowedHeaders: [
        'Origin',
        'X-Requested-With',
        'Content-Type',
        'Accept',
        'Authorization',
        'X-API-Key',
        'X-Client-Version'
      ],
      exposedHeaders: [
        'X-RateLimit-Limit',
        'X-RateLimit-Remaining',
        'X-RateLimit-Reset',
        'X-Response-Time'
      ]
    }));

    // Compression
    this.app.use(compression({
      level: 6,
      threshold: 1024,
      filter: (req, res) => {
        if (req.headers['x-no-compression']) {
          return false;
        }
        return compression.filter(req, res);
      }
    }));

    // Body parsing
    this.app.use(express.json({ 
      limit: '10mb',
      verify: (req: any, res, buf) => {
        req.rawBody = buf;
      }
    }));
    this.app.use(express.urlencoded({ 
      extended: true, 
      limit: '10mb' 
    }));

    // Request logging
    this.app.use((req: Request, res: Response, next: NextFunction) => {
      const start = Date.now();
      
      res.on('finish', () => {
        const duration = Date.now() - start;
        logger.http('HTTP Request', {
          method: req.method,
          url: req.url,
          statusCode: res.statusCode,
          duration,
          ip: req.ip,
          userAgent: req.headers['user-agent']
        });
      });

      next();
    });

    // Rate limiting status middleware
    this.app.use(rateLimitStatus());

    // Global rate limiting
    const globalRateLimit = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: this.environment === 'production' ? 1000 : 10000,
      message: {
        success: false,
        error: 'TooManyRequests',
        message: 'Too many requests from this IP, please try again later.'
      },
      standardHeaders: true,
      legacyHeaders: false
    });
    this.app.use(globalRateLimit);

    // Trust proxy in production
    if (this.environment === 'production') {
      this.app.set('trust proxy', 1);
    }

    logger.info('Middleware initialized successfully');
  }

  /**
   * Initialize API routes
   */
  private initializeRoutes(): void {
    // API Documentation
    this.app.use('/api/docs', swaggerUi.serve);
    this.app.get('/api/docs', swaggerUi.setup(this.getSwaggerSpec()));

    // Health check endpoint (no auth required)
    this.app.get('/health', this.healthCheck.bind(this));
    this.app.get('/api/health', this.healthCheck.bind(this));

    // Metrics endpoint (no auth required)
    this.app.get('/metrics', this.getMetrics.bind(this));

    // API version endpoint
    this.app.get('/api/version', (req: Request, res: Response) => {
      res.json({
        version: process.env.API_VERSION || '1.0.0',
        buildDate: process.env.BUILD_DATE || new Date().toISOString(),
        gitCommit: process.env.GIT_COMMIT || 'unknown',
        environment: this.environment
      });
    });

    // Main API routes
    this.app.use('/api/users', userRoutes);
    this.app.use('/api/agents', agentRoutes);
    this.app.use('/api/arbitrage', arbitrageRoutes);
    this.app.use('/api/portfolios', portfolioRoutes);
    this.app.use('/api/security', securityRoutes);

    // Chainlink webhook endpoints (if needed)
    this.app.use('/webhooks/chainlink', this.setupChainlinkWebhooks());

    // Catch-all for unknown routes
    this.app.use('*', (req: Request, res: Response) => {
      res.status(404).json({
        success: false,
        error: 'NotFound',
        message: `Route ${req.method} ${req.originalUrl} not found`,
        timestamp: new Date().toISOString()
      });
    });

    logger.info('Routes initialized successfully');
  }

  /**
   * Initialize error handling
   */
  private initializeErrorHandling(): void {
    // Global error handler
    this.app.use((error: any, req: Request, res: Response, next: NextFunction) => {
      const errorId = require('crypto').randomUUID();
      
      logger.error('Unhandled application error', {
        errorId,
        error: error.message,
        stack: error.stack,
        method: req.method,
        url: req.url,
        ip: req.ip,
        userAgent: req.headers['user-agent'],
        userId: req.user?.id
      });

      // Don't expose internal errors in production
      const message = this.environment === 'production' 
        ? 'Internal server error' 
        : error.message;

      res.status(error.status || 500).json({
        success: false,
        error: 'InternalServerError',
        message,
        errorId,
        timestamp: new Date().toISOString()
      });
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason: any, promise: Promise<any>) => {
      logger.error('Unhandled Promise Rejection', {
        reason: reason?.message || reason,
        stack: reason?.stack,
        promise: promise.toString()
      });
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error: Error) => {
      logger.critical('Uncaught Exception', {
        error: error.message,
        stack: error.stack
      });

      // Graceful shutdown
      this.gracefulShutdown('UNCAUGHT_EXCEPTION');
    });

    // Handle process termination
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, starting graceful shutdown');
      this.gracefulShutdown('SIGTERM');
    });

    process.on('SIGINT', () => {
      logger.info('SIGINT received, starting graceful shutdown');
      this.gracefulShutdown('SIGINT');
    });

    logger.info('Error handling initialized successfully');
  }

  /**
   * Initialize health checks and monitoring
   */
  private initializeHealthChecks(): void {
    // Periodic health checks
    setInterval(async () => {
      try {
        const health = await this.performHealthCheck();
        if (health.status !== 'healthy') {
          logger.warn('System health check failed', health);
        }
      } catch (error) {
        logger.error('Health check error:', error);
      }
    }, 30000); // Every 30 seconds

    logger.info('Health checks initialized successfully');
  }

  /**
   * Health check endpoint handler
   */
  private async healthCheck(req: Request, res: Response): Promise<void> {
    try {
      const health = await this.performHealthCheck();
      const statusCode = health.status === 'healthy' ? 200 : 
                        health.status === 'degraded' ? 200 : 503;
      
      res.status(statusCode).json(health);
    } catch (error) {
      logger.error('Health check failed:', error);
      res.status(503).json({
        status: 'unhealthy',
        timestamp: new Date(),
        error: 'Health check failed'
      });
    }
  }

  /**
   * Perform comprehensive health check
   */
  private async performHealthCheck(): Promise<HealthCheckResult> {
    const health: HealthCheckResult = {
      status: 'healthy',
      timestamp: new Date(),
      version: process.env.API_VERSION || '1.0.0',
      environment: this.environment,
      services: {
        database: { status: 'unhealthy' },
        chainlink: { status: 'unhealthy', networks: [], services: [] },
        web3: { status: 'unhealthy', chains: [], providers: 0 },
        agents: { status: 'unhealthy', types: 0, models: 0 }
      },
      uptime: Date.now() - this.startTime.getTime(),
      memory: process.memoryUsage()
    };

    try {
      // Database health check
      const dbHealth = await database.healthCheck();
      health.services.database = {
        status: dbHealth.status === 'healthy' ? 'healthy' : 'unhealthy',
        latency: dbHealth.latency,
        connections: database.getPoolStats()
      };

      // Chainlink health check
      const chainlinkHealth = await chainlinkConfig.validateConfiguration();
      health.services.chainlink = {
        status: chainlinkHealth ? 'healthy' : 'unhealthy',
        networks: chainlinkConfig.getAllSupportedChains().map(id => id.toString()),
        services: Object.entries(chainlinkConfig.getChainlinkConfig().services)
          .filter(([, service]) => service.enabled)
          .map(([name]) => name)
      };

      // Web3 health check
      const web3Health = await web3Config.healthCheck();
      const healthyChains = Object.entries(web3Health).filter(([, healthy]) => healthy);
      health.services.web3 = {
        status: healthyChains.length > 0 ? 'healthy' : 'unhealthy',
        chains: healthyChains.map(([chainId]) => parseInt(chainId)),
        providers: web3Config.getSupportedChains().length
      };

      // Agent configuration health check
      const agentConfig = agentConfigService.getConfigSummary();
      health.services.agents = {
        status: 'healthy',
        types: agentConfig.agentTypes.length,
        models: agentConfig.models.length
      };

      // Determine overall status
      const servicesHealthy = Object.values(health.services).every(
        service => service.status === 'healthy'
      );
      
      if (!servicesHealthy) {
        const criticalServicesDown = health.services.database.status === 'unhealthy' ||
                                   health.services.web3.status === 'unhealthy';
        health.status = criticalServicesDown ? 'unhealthy' : 'degraded';
      }

    } catch (error) {
      logger.error('Health check error:', error);
      health.status = 'unhealthy';
    }

    return health;
  }

  /**
   * Get application metrics
   */
  private async getMetrics(req: Request, res: Response): Promise<void> {
    try {
      const metrics = {
        uptime: Date.now() - this.startTime.getTime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage(),
        database: database.getPoolStats(),
        environment: this.environment,
        nodeVersion: process.version,
        platform: process.platform,
        timestamp: new Date().toISOString()
      };

      res.json(metrics);
    } catch (error) {
      logger.error('Error getting metrics:', error);
      res.status(500).json({ error: 'Failed to get metrics' });
    }
  }

  /**
   * Setup Chainlink webhook endpoints
   */
  private setupChainlinkWebhooks(): express.Router {
    const router = express.Router();

    // Automation upkeep webhook
    router.post('/automation/:upkeepId', (req: Request, res: Response) => {
      logger.info('Chainlink automation webhook received', {
        upkeepId: req.params.upkeepId,
        body: req.body
      });
      res.json({ success: true });
    });

    // Functions fulfillment webhook
    router.post('/functions/:requestId', (req: Request, res: Response) => {
      logger.info('Chainlink Functions webhook received', {
        requestId: req.params.requestId,
        body: req.body
      });
      res.json({ success: true });
    });

    // VRF fulfillment webhook
    router.post('/vrf/:requestId', (req: Request, res: Response) => {
      logger.info('Chainlink VRF webhook received', {
        requestId: req.params.requestId,
        body: req.body
      });
      res.json({ success: true });
    });

    return router;
  }

  /**
   * Get allowed CORS origins
   */
  private getAllowedOrigins(): string[] {
    const origins = process.env.ALLOWED_ORIGINS?.split(',') || [];
    
    if (this.environment === 'development') {
      origins.push('http://localhost:3000', 'http://localhost:3001', 'http://127.0.0.1:3000');
    }

    return origins;
  }

  /**
   * Get Swagger API specification
   */
  private getSwaggerSpec(): OpenAPIV3.Document {
    return {
      openapi: '3.0.0',
      info: {
        title: 'ArbOS API',
        version: process.env.API_VERSION || '1.0.0',
        description: 'AI-Powered Arbitrage and Portfolio Management System with Chainlink Integration',
        contact: {
          name: 'ArbOS Support',
          email: 'support@arbos.ai'
        },
        license: {
          name: 'MIT'
        }
      },
      servers: [
        {
          url: process.env.API_BASE_URL || `http://localhost:${this.port}`,
          description: this.environment
        }
      ],
      components: {
        securitySchemes: {
          bearerAuth: {
            type: 'http',
            scheme: 'bearer',
            bearerFormat: 'JWT'
          },
          apiKeyAuth: {
            type: 'apiKey',
            in: 'header',
            name: 'X-API-Key'
          }
        }
      },
      paths: {
        '/health': {
          get: {
            summary: 'Health Check',
            tags: ['System'],
            responses: {
              '200': {
                description: 'System is healthy'
              }
            }
          }
        }
      }
    };
  }

  /**
   * Initialize the application
   */
  public async initialize(): Promise<void> {
    try {
      logger.info('Initializing ArbOS application...');

      // Initialize database
      await database.initialize();
      logger.info('Database initialized successfully');

      // Validate configurations
      const configValidations = await Promise.allSettled([
        chainlinkConfig.validateConfiguration(),
        web3Config.healthCheck()
      ]);

      configValidations.forEach((result, index) => {
        const serviceName = ['Chainlink', 'Web3'][index];
        if (result.status === 'rejected') {
          logger.warn(`${serviceName} configuration validation failed:`, result.reason);
        } else {
          logger.info(`${serviceName} configuration validated successfully`);
        }
      });

      logger.info('ArbOS application initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize ArbOS application:', error);
      throw error;
    }
  }

  /**
   * Start the server
   */
  public async start(): Promise<void> {
    try {
      await this.initialize();

      this.app.listen(this.port, () => {
        logger.info(`ArbOS server started successfully`, {
          port: this.port,
          environment: this.environment,
          nodeVersion: process.version,
          pid: process.pid,
          startTime: this.startTime.toISOString()
        });
      });
    } catch (error) {
      logger.error('Failed to start ArbOS server:', error);
      process.exit(1);
    }
  }

  /**
   * Graceful shutdown
   */
  private async gracefulShutdown(signal: string): Promise<void> {
    logger.info(`Graceful shutdown initiated by ${signal}`);

    try {
      // Close database connections
      await database.close();
      logger.info('Database connections closed');

      // Flush logs
      await logger.flush();

      logger.info('Graceful shutdown completed');
      process.exit(0);
    } catch (error) {
      logger.error('Error during graceful shutdown:', error);
      process.exit(1);
    }
  }
}

// Create and export application instance
const arbOSApp = new ArbOSApplication();

// Start the application if this file is run directly
if (require.main === module) {
  arbOSApp.start().catch((error) => {
    console.error('Failed to start application:', error);
    process.exit(1);
  });
}

export default arbOSApp;
export { ArbOSApplication };
