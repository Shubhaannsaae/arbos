import { Request, Response, NextFunction } from 'express';
import { agentService } from '../services/agentService';
import { chainlinkService } from '../services/chainlinkService';
import { logger } from '../utils/logger';
import { 
  CreateAgentDto, 
  UpdateAgentDto, 
  AgentType, 
  AgentStatus 
} from '../models/Agent';
import { validateRequest } from '../utils/validators';
import { ApiResponse, PaginationParams } from '../types/api';

class AgentController {
  /**
   * Create a new AI agent instance
   */
  public async createAgent(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const createAgentDto: CreateAgentDto = req.body;
      
      // Validate input
      const validation = validateRequest(createAgentDto, {
        name: { required: true, type: 'string', minLength: 3, maxLength: 50 },
        type: { required: true, type: 'string', enum: Object.values(AgentType) },
        configuration: { required: true, type: 'object' },
        permissions: { required: true, type: 'object' }
      });

      if (!validation.isValid) {
        res.status(400).json({
          success: false,
          error: 'Validation Error',
          message: 'Invalid agent configuration',
          details: validation.errors
        });
        return;
      }

      // Initialize Chainlink services for the agent
      await chainlinkService.validateAgentConfiguration(createAgentDto.type);

      const agent = await agentService.createAgent(userId, createAgentDto);

      logger.info(`Agent created successfully`, {
        userId,
        agentId: agent.id,
        agentType: agent.type
      });

      const response: ApiResponse<typeof agent> = {
        success: true,
        data: agent,
        message: 'Agent created successfully'
      };

      res.status(201).json(response);
    } catch (error) {
      logger.error('Error creating agent:', error);
      next(error);
    }
  }

  /**
   * Get agent by ID
   */
  public async getAgent(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      const agent = await agentService.getAgentById(agentId, userId);

      if (!agent) {
        res.status(404).json({
          success: false,
          error: 'Not Found',
          message: 'Agent not found'
        });
        return;
      }

      const response: ApiResponse<typeof agent> = {
        success: true,
        data: agent
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting agent:', error);
      next(error);
    }
  }

  /**
   * Get all agents for a user
   */
  public async getUserAgents(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Unauthorized',
          message: 'User authentication required'
        });
        return;
      }

      const pagination: PaginationParams = {
        page: parseInt(req.query.page as string) || 1,
        limit: parseInt(req.query.limit as string) || 10,
        sortBy: req.query.sortBy as string || 'createdAt',
        sortOrder: (req.query.sortOrder as 'asc' | 'desc') || 'desc'
      };

      const filters = {
        type: req.query.type as AgentType,
        status: req.query.status as AgentStatus,
        enabled: req.query.enabled === 'true' ? true : req.query.enabled === 'false' ? false : undefined
      };

      const result = await agentService.getUserAgents(userId, filters, pagination);

      const response: ApiResponse<typeof result.agents> = {
        success: true,
        data: result.agents,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: result.total,
          totalPages: Math.ceil(result.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting user agents:', error);
      next(error);
    }
  }

  /**
   * Update agent configuration
   */
  public async updateAgent(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;
      const updateData: UpdateAgentDto = req.body;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      // Validate update data
      if (Object.keys(updateData).length === 0) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'No update data provided'
        });
        return;
      }

      const updatedAgent = await agentService.updateAgent(agentId, userId, updateData);

      logger.info(`Agent updated successfully`, {
        userId,
        agentId,
        updates: Object.keys(updateData)
      });

      const response: ApiResponse<typeof updatedAgent> = {
        success: true,
        data: updatedAgent,
        message: 'Agent updated successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error updating agent:', error);
      next(error);
    }
  }

  /**
   * Start an agent
   */
  public async startAgent(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      await agentService.startAgent(agentId, userId);

      logger.info(`Agent started`, { userId, agentId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'Agent started successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error starting agent:', error);
      next(error);
    }
  }

  /**
   * Stop an agent
   */
  public async stopAgent(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      await agentService.stopAgent(agentId, userId);

      logger.info(`Agent stopped`, { userId, agentId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'Agent stopped successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error stopping agent:', error);
      next(error);
    }
  }

  /**
   * Get agent performance metrics
   */
  public async getAgentPerformance(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      const timeRange = {
        start: req.query.start ? new Date(req.query.start as string) : new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        end: req.query.end ? new Date(req.query.end as string) : new Date()
      };

      const performance = await agentService.getAgentPerformance(agentId, userId, timeRange);

      const response: ApiResponse<typeof performance> = {
        success: true,
        data: performance
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting agent performance:', error);
      next(error);
    }
  }

  /**
   * Get agent logs
   */
  public async getAgentLogs(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      const pagination: PaginationParams = {
        page: parseInt(req.query.page as string) || 1,
        limit: parseInt(req.query.limit as string) || 50,
        sortBy: 'timestamp',
        sortOrder: 'desc'
      };

      const filters = {
        level: req.query.level as string,
        startDate: req.query.startDate ? new Date(req.query.startDate as string) : undefined,
        endDate: req.query.endDate ? new Date(req.query.endDate as string) : undefined
      };

      const logs = await agentService.getAgentLogs(agentId, userId, filters, pagination);

      const response: ApiResponse<typeof logs.logs> = {
        success: true,
        data: logs.logs,
        pagination: {
          page: pagination.page,
          limit: pagination.limit,
          total: logs.total,
          totalPages: Math.ceil(logs.total / pagination.limit)
        }
      };

      res.json(response);
    } catch (error) {
      logger.error('Error getting agent logs:', error);
      next(error);
    }
  }

  /**
   * Delete an agent
   */
  public async deleteAgent(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;

      if (!agentId) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID is required'
        });
        return;
      }

      await agentService.deleteAgent(agentId, userId);

      logger.info(`Agent deleted`, { userId, agentId });

      const response: ApiResponse<null> = {
        success: true,
        data: null,
        message: 'Agent deleted successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error deleting agent:', error);
      next(error);
    }
  }

  /**
   * Execute agent action manually
   */
  public async executeAgentAction(
    req: Request,
    res: Response,
    next: NextFunction
  ): Promise<void> {
    try {
      const userId = req.user?.id;
      const agentId = req.params.agentId;
      const { action, parameters } = req.body;

      if (!agentId || !action) {
        res.status(400).json({
          success: false,
          error: 'Bad Request',
          message: 'Agent ID and action are required'
        });
        return;
      }

      const result = await agentService.executeAgentAction(agentId, userId, action, parameters);

      logger.info(`Agent action executed`, { 
        userId, 
        agentId, 
        action,
        success: result.success 
      });

      const response: ApiResponse<typeof result> = {
        success: true,
        data: result,
        message: 'Agent action executed successfully'
      };

      res.json(response);
    } catch (error) {
      logger.error('Error executing agent action:', error);
      next(error);
    }
  }
}

export const agentController = new AgentController();
export default agentController;
