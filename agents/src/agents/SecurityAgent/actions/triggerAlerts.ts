import { ethers, BigNumber } from 'ethers';
import { logger } from '../../../shared/utils/logger';
import { AgentContext } from '../../../shared/types/agent';
import { SecurityAlert, ThreatLevel } from '../../../shared/types/blockchain';
import { SecurityProvider } from '../providers/securityProvider';

export interface AlertConfig {
  alertType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  affectedAddresses: string[];
  evidenceData: any;
  recommendedActions: string[];
  emergencyStop?: boolean;
  notificationChannels?: string[];
  escalationRules?: EscalationRule[];
}

export interface EscalationRule {
  condition: string;
  threshold: number;
  action: 'notify' | 'pause' | 'emergency_stop';
  recipients: string[];
  timeWindow: number;
}

export interface AlertResult {
  success: boolean;
  alertId: string;
  alertsTriggered: number;
  notificationsSent: number;
  emergencyActionsTaken: string[];
  escalationsTriggered: EscalationRule[];
  processingTime: number;
  errors: string[];
}

export interface NotificationChannel {
  type: 'email' | 'webhook' | 'sms' | 'slack' | 'discord' | 'telegram';
  endpoint: string;
  apiKey?: string;
  enabled: boolean;
  severity_filter: ('low' | 'medium' | 'high' | 'critical')[];
}

export async function triggerAlerts(
  securityProvider: SecurityProvider,
  context: AgentContext,
  config: AlertConfig
): Promise<AlertResult> {
  const startTime = Date.now();
  const alertId = `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  logger.info('Triggering security alert', {
    agentId: context.agentId,
    alertId,
    alertType: config.alertType,
    severity: config.severity,
    affectedAddresses: config.affectedAddresses.length
  });

  const result: AlertResult = {
    success: false,
    alertId,
    alertsTriggered: 0,
    notificationsSent: 0,
    emergencyActionsTaken: [],
    escalationsTriggered: [],
    processingTime: 0,
    errors: []
  };

  try {
    // Step 1: Create security alert record
    const alert = await createSecurityAlert(alertId, config, context);
    result.alertsTriggered = 1;

    // Step 2: Determine notification channels
    const notificationChannels = await determineNotificationChannels(config, securityProvider);

    // Step 3: Send notifications
    const notificationResults = await sendNotifications(alert, notificationChannels, securityProvider);
    result.notificationsSent = notificationResults.successCount;
    result.errors.push(...notificationResults.errors);

    // Step 4: Execute emergency actions if required
    if (config.emergencyStop) {
      const emergencyActions = await executeEmergencyActions(alert, config, securityProvider);
      result.emergencyActionsTaken = emergencyActions.actionsExecuted;
      result.errors.push(...emergencyActions.errors);
    }

    // Step 5: Check escalation rules
    if (config.escalationRules) {
      const escalationResults = await processEscalationRules(alert, config.escalationRules, securityProvider);
      result.escalationsTriggered = escalationResults.triggeredRules;
      result.errors.push(...escalationResults.errors);
    }

    // Step 6: Log alert to blockchain (using Chainlink Functions)
    try {
      await logAlertToBlockchain(alert, securityProvider);
    } catch (error) {
      logger.warn('Failed to log alert to blockchain', {
        alertId,
        error: error instanceof Error ? error.message : String(error)
      });
    }

    // Step 7: Update threat intelligence
    await updateThreatIntelligence(alert, config, securityProvider);

    result.success = true;
    result.processingTime = Date.now() - startTime;

    logger.info('Security alert triggered successfully', {
      agentId: context.agentId,
      alertId,
      notificationsSent: result.notificationsSent,
      emergencyActions: result.emergencyActionsTaken.length,
      duration: result.processingTime
    });

    return result;

  } catch (error) {
    result.errors.push(error instanceof Error ? error.message : String(error));
    result.processingTime = Date.now() - startTime;

    logger.error('Failed to trigger security alert', {
      agentId: context.agentId,
      alertId,
      error: error instanceof Error ? error.message : String(error),
      duration: result.processingTime
    });

    return result;
  }
}

async function createSecurityAlert(
  alertId: string,
  config: AlertConfig,
  context: AgentContext
): Promise<SecurityAlert> {
  const alert: SecurityAlert = {
    id: alertId,
    type: config.alertType,
    severity: config.severity,
    title: generateAlertTitle(config.alertType, config.severity),
    description: config.description,
    timestamp: Date.now(),
    chainId: context.networkIds[0] || 0,
    transactionHash: extractTransactionHash(config.evidenceData),
    contractAddress: extractContractAddress(config.evidenceData),
    affectedAddresses: config.affectedAddresses,
    status: 'active',
    source: context.agentId,
    metadata: {
      evidenceData: config.evidenceData,
      recommendedActions: config.recommendedActions,
      contextData: {
        userId: context.userId,
        sessionId: context.sessionId,
        timestamp: context.timestamp
      }
    }
  };

  // Store alert in security provider
  try {
    await securityProvider.storeAlert(alert);
    
    logger.debug('Security alert created and stored', {
      alertId,
      type: alert.type,
      severity: alert.severity
    });

  } catch (error) {
    logger.error('Failed to store security alert', {
      alertId,
      error: error instanceof Error ? error.message : String(error)
    });
  }

  return alert;
}

async function determineNotificationChannels(
  config: AlertConfig,
  securityProvider: SecurityProvider
): Promise<NotificationChannel[]> {
  try {
    // Get configured notification channels
    const allChannels = await securityProvider.getNotificationChannels();
    
    // Filter channels based on severity and configuration
    const relevantChannels = allChannels.filter(channel => {
      // Check if channel is enabled
      if (!channel.enabled) return false;
      
      // Check severity filter
      if (!channel.severity_filter.includes(config.severity)) return false;
      
      // Check if specific channels are requested
      if (config.notificationChannels && config.notificationChannels.length > 0) {
        return config.notificationChannels.includes(channel.type);
      }
      
      return true;
    });

    logger.debug('Notification channels determined', {
      totalChannels: allChannels.length,
      relevantChannels: relevantChannels.length,
      severity: config.severity
    });

    return relevantChannels;

  } catch (error) {
    logger.error('Failed to determine notification channels', {
      error: error instanceof Error ? error.message : String(error)
    });

    return [];
  }
}

async function sendNotifications(
  alert: SecurityAlert,
  channels: NotificationChannel[],
  securityProvider: SecurityProvider
): Promise<{ successCount: number; errors: string[] }> {
  const results = { successCount: 0, errors: [] };

  for (const channel of channels) {
    try {
      await sendNotificationToChannel(alert, channel, securityProvider);
      results.successCount++;
      
      logger.debug('Notification sent successfully', {
        alertId: alert.id,
        channelType: channel.type,
        endpoint: channel.endpoint
      });

    } catch (error) {
      const errorMessage = `Failed to send ${channel.type} notification: ${error instanceof Error ? error.message : String(error)}`;
      results.errors.push(errorMessage);
      
      logger.error('Notification failed', {
        alertId: alert.id,
        channelType: channel.type,
        error: errorMessage
      });
    }
  }

  return results;
}

async function sendNotificationToChannel(
  alert: SecurityAlert,
  channel: NotificationChannel,
  securityProvider: SecurityProvider
): Promise<void> {
  const notificationPayload = formatNotificationForChannel(alert, channel);

  switch (channel.type) {
    case 'webhook':
      await sendWebhookNotification(channel.endpoint, notificationPayload, channel.apiKey);
      break;
      
    case 'email':
      await sendEmailNotification(channel.endpoint, notificationPayload, channel.apiKey);
      break;
      
    case 'slack':
      await sendSlackNotification(channel.endpoint, notificationPayload, channel.apiKey);
      break;
      
    case 'discord':
      await sendDiscordNotification(channel.endpoint, notificationPayload);
      break;
      
    case 'telegram':
      await sendTelegramNotification(channel.endpoint, notificationPayload, channel.apiKey);
      break;
      
    case 'sms':
      await sendSMSNotification(channel.endpoint, notificationPayload, channel.apiKey);
      break;
      
    default:
      throw new Error(`Unsupported notification channel type: ${channel.type}`);
  }
}

async function sendWebhookNotification(endpoint: string, payload: any, apiKey?: string): Promise<void> {
  const headers: any = {
    'Content-Type': 'application/json',
    'User-Agent': 'ArbOS-SecurityAgent/1.0'
  };

  if (apiKey) {
    headers['Authorization'] = `Bearer ${apiKey}`;
  }

  const response = await fetch(endpoint, {
    method: 'POST',
    headers,
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`Webhook request failed: ${response.status} ${response.statusText}`);
  }
}

async function sendEmailNotification(email: string, payload: any, apiKey?: string): Promise<void> {
  // Implementation would use email service provider (SendGrid, AWS SES, etc.)
  // For now, we'll create a structured email format
  
  const emailContent = {
    to: email,
    subject: `Security Alert: ${payload.title}`,
    html: formatEmailContent(payload),
    text: formatTextContent(payload)
  };

  // This would integrate with actual email service
  logger.debug('Email notification prepared', {
    recipient: email,
    subject: emailContent.subject
  });
}

async function sendSlackNotification(webhook: string, payload: any, apiKey?: string): Promise<void> {
  const slackPayload = {
    channel: '#security-alerts',
    username: 'Security Agent',
    icon_emoji: ':warning:',
    attachments: [
      {
        color: getSeverityColor(payload.severity),
        title: payload.title,
        text: payload.description,
        fields: [
          {
            title: 'Severity',
            value: payload.severity.toUpperCase(),
            short: true
          },
          {
            title: 'Alert ID',
            value: payload.id,
            short: true
          },
          {
            title: 'Timestamp',
            value: new Date(payload.timestamp).toISOString(),
            short: true
          },
          {
            title: 'Affected Addresses',
            value: payload.affectedAddresses.join('\n'),
            short: false
          }
        ],
        footer: 'ArbOS Security Agent',
        ts: Math.floor(payload.timestamp / 1000)
      }
    ]
  };

  await sendWebhookNotification(webhook, slackPayload);
}

async function sendDiscordNotification(webhook: string, payload: any): Promise<void> {
  const discordPayload = {
    embeds: [
      {
        title: payload.title,
        description: payload.description,
        color: getSeverityColorDiscord(payload.severity),
        fields: [
          {
            name: 'Severity',
            value: payload.severity.toUpperCase(),
            inline: true
          },
          {
            name: 'Alert ID',
            value: payload.id,
            inline: true
          },
          {
            name: 'Timestamp',
            value: new Date(payload.timestamp).toISOString(),
            inline: true
          }
        ],
        footer: {
          text: 'ArbOS Security Agent'
        },
        timestamp: new Date(payload.timestamp).toISOString()
      }
    ]
  };

  await sendWebhookNotification(webhook, discordPayload);
}

async function sendTelegramNotification(chatId: string, payload: any, botToken?: string): Promise<void> {
  if (!botToken) {
    throw new Error('Telegram bot token required');
  }

  const message = formatTelegramMessage(payload);
  const telegramEndpoint = `https://api.telegram.org/bot${botToken}/sendMessage`;
  
  const telegramPayload = {
    chat_id: chatId,
    text: message,
    parse_mode: 'Markdown'
  };

  await sendWebhookNotification(telegramEndpoint, telegramPayload);
}

async function sendSMSNotification(phoneNumber: string, payload: any, apiKey?: string): Promise<void> {
  // Implementation would use SMS service provider (Twilio, AWS SNS, etc.)
  const smsContent = formatSMSContent(payload);
  
  logger.debug('SMS notification prepared', {
    recipient: phoneNumber,
    content: smsContent.substring(0, 50) + '...'
  });
}

async function executeEmergencyActions(
  alert: SecurityAlert,
  config: AlertConfig,
  securityProvider: SecurityProvider
): Promise<{ actionsExecuted: string[]; errors: string[] }> {
  const results = { actionsExecuted: [], errors: [] };

  try {
    logger.warn('Executing emergency actions', {
      alertId: alert.id,
      severity: alert.severity,
      affectedAddresses: alert.affectedAddresses?.length || 0
    });

    // Action 1: Pause affected contracts if possible
    for (const address of config.affectedAddresses) {
      try {
        const pauseResult = await securityProvider.emergencyPauseContract(address, alert.chainId || 1);
        if (pauseResult.success) {
          results.actionsExecuted.push(`Paused contract ${address}`);
        }
      } catch (error) {
        results.errors.push(`Failed to pause contract ${address}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    // Action 2: Block suspicious addresses
    for (const address of config.affectedAddresses) {
      try {
        await securityProvider.addToBlocklist(address, alert.type, alert.severity);
        results.actionsExecuted.push(`Added ${address} to blocklist`);
      } catch (error) {
        results.errors.push(`Failed to blocklist ${address}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    // Action 3: Trigger circuit breaker for high-risk situations
    if (alert.severity === 'critical') {
      try {
        await securityProvider.triggerCircuitBreaker(alert.chainId || 1, 'security_emergency', alert.id);
        results.actionsExecuted.push('Circuit breaker activated');
      } catch (error) {
        results.errors.push(`Failed to trigger circuit breaker: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    // Action 4: Notify external security services
    try {
      await securityProvider.notifyExternalSecurityServices(alert);
      results.actionsExecuted.push('External security services notified');
    } catch (error) {
      results.errors.push(`Failed to notify external services: ${error instanceof Error ? error.message : String(error)}`);
    }

    logger.info('Emergency actions completed', {
      alertId: alert.id,
      actionsExecuted: results.actionsExecuted.length,
      errors: results.errors.length
    });

  } catch (error) {
    results.errors.push(`Emergency action execution failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  return results;
}

async function processEscalationRules(
  alert: SecurityAlert,
  escalationRules: EscalationRule[],
  securityProvider: SecurityProvider
): Promise<{ triggeredRules: EscalationRule[]; errors: string[] }> {
  const results = { triggeredRules: [], errors: [] };

  for (const rule of escalationRules) {
    try {
      const shouldEscalate = await evaluateEscalationCondition(alert, rule, securityProvider);
      
      if (shouldEscalate) {
        await executeEscalationAction(alert, rule, securityProvider);
        results.triggeredRules.push(rule);
        
        logger.info('Escalation rule triggered', {
          alertId: alert.id,
          condition: rule.condition,
          action: rule.action
        });
      }

    } catch (error) {
      results.errors.push(`Escalation rule failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return results;
}

async function logAlertToBlockchain(alert: SecurityAlert, securityProvider: SecurityProvider): Promise<void> {
  try {
    // Use Chainlink Functions to log security alert to blockchain
    const chainlinkFunctionsSource = `
      const alertData = {
        id: args[0],
        type: args[1],
        severity: args[2],
        timestamp: args[3],
        chainId: args[4]
      };
      
      // Log to immutable storage
      return Functions.encodeString(JSON.stringify({
        alert: alertData,
        hash: Functions.keccak256(JSON.stringify(alertData)),
        logged_at: Date.now()
      }));
    `;

    await securityProvider.executeChainlinkFunction(
      chainlinkFunctionsSource,
      [alert.id, alert.type, alert.severity, alert.timestamp.toString(), (alert.chainId || 0).toString()],
      alert.chainId || 1
    );

    logger.debug('Alert logged to blockchain', {
      alertId: alert.id,
      chainId: alert.chainId
    });

  } catch (error) {
    logger.error('Failed to log alert to blockchain', {
      alertId: alert.id,
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

async function updateThreatIntelligence(
  alert: SecurityAlert,
  config: AlertConfig,
  securityProvider: SecurityProvider
): Promise<void> {
  try {
    const threatIntelligence = {
      alertId: alert.id,
      threatType: alert.type,
      severity: alert.severity,
      indicators: {
        addresses: config.affectedAddresses,
        patterns: extractPatterns(config.evidenceData),
        timestamps: [alert.timestamp],
        chains: [alert.chainId]
      },
      attribution: {
        source: alert.source,
        confidence: calculateThreatConfidence(config.evidenceData),
        ttps: extractTTPs(config.evidenceData) // Tactics, Techniques, and Procedures
      }
    };

    await securityProvider.updateThreatIntelligence(threatIntelligence);
    
    logger.debug('Threat intelligence updated', {
      alertId: alert.id,
      threatType: alert.type
    });

  } catch (error) {
    logger.error('Failed to update threat intelligence', {
      alertId: alert.id,
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

// Helper functions
function generateAlertTitle(alertType: string, severity: string): string {
  const typeMap: Record<string, string> = {
    'suspicious_transaction': 'Suspicious Transaction Detected',
    'price_manipulation': 'Price Manipulation Alert',
    'fraud_detected': 'Fraudulent Activity Detected',
    'rug_pull': 'Potential Rug Pull Detected',
    'phishing': 'Phishing Activity Detected',
    'contract_vulnerability': 'Smart Contract Vulnerability',
    'anomaly_detected': 'Behavioral Anomaly Detected'
  };

  const title = typeMap[alertType] || 'Security Alert';
  return `${severity.toUpperCase()}: ${title}`;
}

function extractTransactionHash(evidenceData: any): string | undefined {
  if (evidenceData?.transactionHash) return evidenceData.transactionHash;
  if (evidenceData?.transaction?.hash) return evidenceData.transaction.hash;
  if (evidenceData?.hash) return evidenceData.hash;
  return undefined;
}

function extractContractAddress(evidenceData: any): string | undefined {
  if (evidenceData?.contractAddress) return evidenceData.contractAddress;
  if (evidenceData?.contract?.address) return evidenceData.contract.address;
  if (evidenceData?.to) return evidenceData.to;
  return undefined;
}

function formatNotificationForChannel(alert: SecurityAlert, channel: NotificationChannel): any {
  const basePayload = {
    id: alert.id,
    type: alert.type,
    severity: alert.severity,
    title: alert.title,
    description: alert.description,
    timestamp: alert.timestamp,
    chainId: alert.chainId,
    affectedAddresses: alert.affectedAddresses || [],
    metadata: alert.metadata
  };

  // Channel-specific formatting can be added here
  switch (channel.type) {
    case 'email':
      return {
        ...basePayload,
        htmlContent: formatEmailContent(basePayload),
        textContent: formatTextContent(basePayload)
      };
    
    default:
      return basePayload;
  }
}

function formatEmailContent(payload: any): string {
  return `
    <html>
      <body>
        <h2 style="color: ${getSeverityColor(payload.severity)};">${payload.title}</h2>
        <p><strong>Alert ID:</strong> ${payload.id}</p>
        <p><strong>Severity:</strong> ${payload.severity.toUpperCase()}</p>
        <p><strong>Timestamp:</strong> ${new Date(payload.timestamp).toISOString()}</p>
        <p><strong>Description:</strong> ${payload.description}</p>
        ${payload.affectedAddresses.length > 0 ? `
          <h3>Affected Addresses:</h3>
          <ul>
            ${payload.affectedAddresses.map((addr: string) => `<li><code>${addr}</code></li>`).join('')}
          </ul>
        ` : ''}
        <hr>
        <p><small>Generated by ArbOS Security Agent</small></p>
      </body>
    </html>
  `;
}

function formatTextContent(payload: any): string {
  return `
SECURITY ALERT: ${payload.title}

Alert ID: ${payload.id}
Severity: ${payload.severity.toUpperCase()}
Timestamp: ${new Date(payload.timestamp).toISOString()}

Description: ${payload.description}

${payload.affectedAddresses.length > 0 ? `
Affected Addresses:
${payload.affectedAddresses.map((addr: string) => `- ${addr}`).join('\n')}
` : ''}

Generated by ArbOS Security Agent
  `.trim();
}

function formatTelegramMessage(payload: any): string {
  const severityEmoji = {
    'low': '🔵',
    'medium': '🟡',
    'high': '🟠',
    'critical': '🔴'
  };

  return `
${severityEmoji[payload.severity] || '⚠️'} *${payload.title}*

*Alert ID:* \`${payload.id}\`
*Severity:* ${payload.severity.toUpperCase()}
*Time:* ${new Date(payload.timestamp).toISOString()}

*Description:* ${payload.description}

${payload.affectedAddresses.length > 0 ? `
*Affected Addresses:*
${payload.affectedAddresses.map((addr: string) => `\`${addr}\``).join('\n')}
` : ''}

_Generated by ArbOS Security Agent_
  `.trim();
}

function formatSMSContent(payload: any): string {
  return `SECURITY ALERT: ${payload.severity.toUpperCase()} - ${payload.type}. Alert ID: ${payload.id}. Check dashboard for details.`;
}

function getSeverityColor(severity: string): string {
  const colors = {
    'low': '#36a64f',      // Green
    'medium': '#ffb347',   // Orange
    'high': '#ff6b47',     // Red-Orange
    'critical': '#ff0000'  // Red
  };
  return colors[severity] || '#666666';
}

function getSeverityColorDiscord(severity: string): number {
  const colors = {
    'low': 0x36a64f,      // Green
    'medium': 0xffb347,   // Orange
    'high': 0xff6b47,     // Red-Orange
    'critical': 0xff0000  // Red
  };
  return colors[severity] || 0x666666;
}

async function evaluateEscalationCondition(
  alert: SecurityAlert,
  rule: EscalationRule,
  securityProvider: SecurityProvider
): Promise<boolean> {
  // Evaluate escalation conditions
  switch (rule.condition) {
    case 'severity_threshold':
      const severityOrder = { low: 1, medium: 2, high: 3, critical: 4 };
      return severityOrder[alert.severity] >= rule.threshold;
    
    case 'repeat_alert':
      const recentAlerts = await securityProvider.getRecentAlerts(alert.type, rule.timeWindow);
      return recentAlerts.length >= rule.threshold;
    
    case 'multiple_addresses':
      return (alert.affectedAddresses?.length || 0) >= rule.threshold;
    
    default:
      return false;
  }
}

async function executeEscalationAction(
  alert: SecurityAlert,
  rule: EscalationRule,
  securityProvider: SecurityProvider
): Promise<void> {
  switch (rule.action) {
    case 'notify':
      // Send notifications to specified recipients
      for (const recipient of rule.recipients) {
        // Implementation would send escalation notification
      }
      break;
    
    case 'pause':
      // Pause affected services/contracts
      if (alert.contractAddress) {
        await securityProvider.emergencyPauseContract(alert.contractAddress, alert.chainId || 1);
      }
      break;
    
    case 'emergency_stop':
      // Trigger emergency stop procedures
      await securityProvider.triggerCircuitBreaker(alert.chainId || 1, 'escalation_emergency', alert.id);
      break;
  }
}

function extractPatterns(evidenceData: any): string[] {
  const patterns: string[] = [];
  
  if (evidenceData?.suspicionReasons) {
    patterns.push(...evidenceData.suspicionReasons);
  }
  
  if (evidenceData?.riskCategories) {
    patterns.push(...evidenceData.riskCategories);
  }
  
  if (evidenceData?.patterns) {
    patterns.push(...evidenceData.patterns);
  }
  
  return [...new Set(patterns)];
}

function calculateThreatConfidence(evidenceData: any): number {
  if (evidenceData?.confidence) return evidenceData.confidence;
  if (evidenceData?.suspicionScore) return evidenceData.suspicionScore / 100;
  return 0.7; // Default confidence
}

function extractTTPs(evidenceData: any): string[] {
  const ttps: string[] = [];
  
  if (evidenceData?.tactics) ttps.push(...evidenceData.tactics);
  if (evidenceData?.techniques) ttps.push(...evidenceData.techniques);
  if (evidenceData?.procedures) ttps.push(...evidenceData.procedures);
  
  return ttps;
}
