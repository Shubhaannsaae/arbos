import { ethers } from 'ethers';
import { logger } from './logger';

interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: any;
}

interface ValidationRule {
  required?: boolean;
  type?: 'string' | 'number' | 'boolean' | 'array' | 'object';
  minLength?: number;
  maxLength?: number;
  min?: number;
  max?: number;
  pattern?: RegExp;
  enum?: any[];
  custom?: (value: any) => boolean | string;
}

interface ValidationSchema {
  [key: string]: ValidationRule;
}

class ValidatorService {
  private readonly ethereumAddressPattern = /^0x[a-fA-F0-9]{40}$/;
  private readonly transactionHashPattern = /^0x[a-fA-F0-9]{64}$/;
  private readonly emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  private readonly uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  private readonly hexPattern = /^0x[a-fA-F0-9]+$/;

  /**
   * Validate data against schema
   */
  public validateSchema(data: any, schema: ValidationSchema): ValidationResult {
    const errors: ValidationError[] = [];

    try {
      for (const [field, rule] of Object.entries(schema)) {
        const value = data[field];
        const fieldErrors = this.validateField(field, value, rule);
        errors.push(...fieldErrors);
      }

      const result = {
        isValid: errors.length === 0,
        errors
      };

      if (errors.length > 0) {
        logger.debug('Validation failed', {
          errorCount: errors.length,
          fields: errors.map(e => e.field)
        });
      }

      return result;
    } catch (error: any) {
      logger.error('Schema validation error', { error: error.message });
      return {
        isValid: false,
        errors: [{
          field: 'schema',
          message: 'Validation schema error',
          code: 'SCHEMA_ERROR'
        }]
      };
    }
  }

  /**
   * Validate individual field
   */
  private validateField(field: string, value: any, rule: ValidationRule): ValidationError[] {
    const errors: ValidationError[] = [];

    // Required validation
    if (rule.required && (value === undefined || value === null || value === '')) {
      errors.push({
        field,
        message: `${field} is required`,
        code: 'REQUIRED',
        value
      });
      return errors; // Return early if required field is missing
    }

    // Skip further validation if value is not provided and not required
    if (value === undefined || value === null) {
      return errors;
    }

    // Type validation
    if (rule.type && !this.validateType(value, rule.type)) {
      errors.push({
        field,
        message: `${field} must be of type ${rule.type}`,
        code: 'INVALID_TYPE',
        value
      });
    }

    // String validations
    if (rule.type === 'string' && typeof value === 'string') {
      if (rule.minLength && value.length < rule.minLength) {
        errors.push({
          field,
          message: `${field} must be at least ${rule.minLength} characters long`,
          code: 'MIN_LENGTH',
          value
        });
      }

      if (rule.maxLength && value.length > rule.maxLength) {
        errors.push({
          field,
          message: `${field} must be at most ${rule.maxLength} characters long`,
          code: 'MAX_LENGTH',
          value
        });
      }

      if (rule.pattern && !rule.pattern.test(value)) {
        errors.push({
          field,
          message: `${field} format is invalid`,
          code: 'INVALID_FORMAT',
          value
        });
      }
    }

    // Number validations
    if (rule.type === 'number' && typeof value === 'number') {
      if (rule.min !== undefined && value < rule.min) {
        errors.push({
          field,
          message: `${field} must be at least ${rule.min}`,
          code: 'MIN_VALUE',
          value
        });
      }

      if (rule.max !== undefined && value > rule.max) {
        errors.push({
          field,
          message: `${field} must be at most ${rule.max}`,
          code: 'MAX_VALUE',
          value
        });
      }
    }

    // Array validations
    if (rule.type === 'array' && Array.isArray(value)) {
      if (rule.minLength && value.length < rule.minLength) {
        errors.push({
          field,
          message: `${field} must contain at least ${rule.minLength} items`,
          code: 'MIN_ARRAY_LENGTH',
          value
        });
      }

      if (rule.maxLength && value.length > rule.maxLength) {
        errors.push({
          field,
          message: `${field} must contain at most ${rule.maxLength} items`,
          code: 'MAX_ARRAY_LENGTH',
          value
        });
      }
    }

    // Enum validation
    if (rule.enum && !rule.enum.includes(value)) {
      errors.push({
        field,
        message: `${field} must be one of: ${rule.enum.join(', ')}`,
        code: 'INVALID_ENUM',
        value
      });
    }

    // Custom validation
    if (rule.custom) {
      const customResult = rule.custom(value);
      if (customResult !== true) {
        errors.push({
          field,
          message: typeof customResult === 'string' ? customResult : `${field} is invalid`,
          code: 'CUSTOM_VALIDATION',
          value
        });
      }
    }

    return errors;
  }

  /**
   * Validate value type
   */
  private validateType(value: any, type: string): boolean {
    switch (type) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'array':
        return Array.isArray(value);
      case 'object':
        return typeof value === 'object' && value !== null && !Array.isArray(value);
      default:
        return false;
    }
  }

  /**
   * Validate Ethereum address
   */
  public isValidEthereumAddress(address: string): boolean {
    if (!address || typeof address !== 'string') {
      return false;
    }

    // Check basic format
    if (!this.ethereumAddressPattern.test(address)) {
      return false;
    }

    // Use ethers.js for checksum validation
    try {
      return ethers.utils.isAddress(address);
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate transaction hash
   */
  public isValidTransactionHash(hash: string): boolean {
    if (!hash || typeof hash !== 'string') {
      return false;
    }

    return this.transactionHashPattern.test(hash);
  }

  /**
   * Validate email address
   */
  public isValidEmail(email: string): boolean {
    if (!email || typeof email !== 'string') {
      return false;
    }

    return this.emailPattern.test(email) && email.length <= 254;
  }

  /**
   * Validate UUID
   */
  public isValidUUID(uuid: string): boolean {
    if (!uuid || typeof uuid !== 'string') {
      return false;
    }

    return this.uuidPattern.test(uuid);
  }

  /**
   * Validate chain ID
   */
  public isValidChainId(chainId: number): boolean {
    if (!Number.isInteger(chainId) || chainId <= 0) {
      return false;
    }

    // List of known chain IDs
    const knownChainIds = [
      1,      // Ethereum Mainnet
      3,      // Ropsten
      4,      // Rinkeby
      5,      // Goerli
      10,     // Optimism
      56,     // BSC
      137,    // Polygon
      250,    // Fantom
      1337,   // Localhost
      42161,  // Arbitrum One
      43114,  // Avalanche
      43113,  // Avalanche Fuji
      80001,  // Mumbai (Polygon Testnet)
      421613, // Arbitrum Goerli
      11155111 // Sepolia
    ];

    return knownChainIds.includes(chainId);
  }

  /**
   * Validate private key
   */
  public isValidPrivateKey(privateKey: string): boolean {
    if (!privateKey || typeof privateKey !== 'string') {
      return false;
    }

    try {
      // Remove 0x prefix if present
      const cleanKey = privateKey.startsWith('0x') ? privateKey.slice(2) : privateKey;
      
      // Check length (64 hex characters = 32 bytes)
      if (cleanKey.length !== 64) {
        return false;
      }

      // Check if valid hex
      if (!/^[a-fA-F0-9]{64}$/.test(cleanKey)) {
        return false;
      }

      // Validate with ethers.js
      new ethers.Wallet(privateKey);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate mnemonic phrase
   */
  public isValidMnemonic(mnemonic: string): boolean {
    if (!mnemonic || typeof mnemonic !== 'string') {
      return false;
    }

    try {
      ethers.utils.HDNode.fromMnemonic(mnemonic);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate percentage value
   */
  public isValidPercentage(value: number): boolean {
    return typeof value === 'number' && value >= 0 && value <= 100 && !isNaN(value);
  }

  /**
   * Validate URL
   */
  public isValidUrl(url: string): boolean {
    if (!url || typeof url !== 'string') {
      return false;
    }

    try {
      new URL(url);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate JSON string
   */
  public isValidJson(jsonString: string): boolean {
    if (!jsonString || typeof jsonString !== 'string') {
      return false;
    }

    try {
      JSON.parse(jsonString);
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate hex string
   */
  public isValidHex(hex: string): boolean {
    if (!hex || typeof hex !== 'string') {
      return false;
    }

    return this.hexPattern.test(hex);
  }

  /**
   * Validate gas limit
   */
  public isValidGasLimit(gasLimit: number): boolean {
    return Number.isInteger(gasLimit) && gasLimit >= 21000 && gasLimit <= 10000000;
  }

  /**
   * Validate gas price (in wei)
   */
  public isValidGasPrice(gasPrice: string | number): boolean {
    try {
      const gasPriceBN = ethers.BigNumber.from(gasPrice);
      return gasPriceBN.gt(0) && gasPriceBN.lt(ethers.utils.parseUnits('1000', 'gwei'));
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate token amount
   */
  public isValidTokenAmount(amount: string, decimals: number = 18): boolean {
    try {
      const amountBN = ethers.utils.parseUnits(amount, decimals);
      return amountBN.gt(0);
    } catch (error) {
      return false;
    }
  }

  /**
   * Validate signature
   */
  public isValidSignature(signature: string): boolean {
    if (!signature || typeof signature !== 'string') {
      return false;
    }

    // Standard Ethereum signature is 132 characters (0x + 65 bytes in hex)
    return /^0x[a-fA-F0-9]{130}$/.test(signature);
  }

  /**
   * Validate block number
   */
  public isValidBlockNumber(blockNumber: number | string): boolean {
    if (typeof blockNumber === 'string') {
      if (blockNumber === 'latest' || blockNumber === 'pending' || blockNumber === 'earliest') {
        return true;
      }
      blockNumber = parseInt(blockNumber);
    }

    return Number.isInteger(blockNumber) && blockNumber >= 0;
  }

  /**
   * Validate timestamp
   */
  public isValidTimestamp(timestamp: number | string): boolean {
    const ts = typeof timestamp === 'string' ? parseInt(timestamp) : timestamp;
    
    if (!Number.isInteger(ts)) {
      return false;
    }

    // Check if timestamp is reasonable (between 2009 and 2100)
    const minTimestamp = 1230768000; // January 1, 2009
    const maxTimestamp = 4102444800; // January 1, 2100
    
    return ts >= minTimestamp && ts <= maxTimestamp;
  }

  /**
   * Validate date string
   */
  public isValidDateString(dateString: string): boolean {
    if (!dateString || typeof dateString !== 'string') {
      return false;
    }

    const date = new Date(dateString);
    return !isNaN(date.getTime());
  }

  /**
   * Validate password strength
   */
  public isStrongPassword(password: string): { isValid: boolean; score: number; feedback: string[] } {
    if (!password || typeof password !== 'string') {
      return {
        isValid: false,
        score: 0,
        feedback: ['Password is required']
      };
    }

    const feedback: string[] = [];
    let score = 0;

    // Length check
    if (password.length >= 8) {
      score += 1;
    } else {
      feedback.push('Password must be at least 8 characters long');
    }

    // Uppercase check
    if (/[A-Z]/.test(password)) {
      score += 1;
    } else {
      feedback.push('Password must contain at least one uppercase letter');
    }

    // Lowercase check
    if (/[a-z]/.test(password)) {
      score += 1;
    } else {
      feedback.push('Password must contain at least one lowercase letter');
    }

    // Number check
    if (/\d/.test(password)) {
      score += 1;
    } else {
      feedback.push('Password must contain at least one number');
    }

    // Special character check
    if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
      score += 1;
    } else {
      feedback.push('Password must contain at least one special character');
    }

    // Common patterns check
    const commonPatterns = [
      /password/i,
      /123456/,
      /qwerty/i,
      /admin/i,
      /letmein/i
    ];

    if (commonPatterns.some(pattern => pattern.test(password))) {
      score -= 1;
      feedback.push('Password contains common patterns and is easily guessable');
    }

    return {
      isValid: score >= 4 && feedback.length === 0,
      score: Math.max(0, Math.min(5, score)),
      feedback
    };
  }

  /**
   * Sanitize input string
   */
  public sanitizeString(input: string): string {
    if (!input || typeof input !== 'string') {
      return '';
    }

    return input
      .trim()
      .replace(/[<>'"]/g, '') // Remove potential XSS characters
      .replace(/\x00/g, ''); // Remove null bytes
  }

  /**
   * Validate and sanitize user input
   */
  public validateAndSanitize(input: any, rules: ValidationRule & { sanitize?: boolean }): {
    isValid: boolean;
    value: any;
    errors: ValidationError[];
  } {
    const errors: ValidationError[] = [];
    let value = input;

    // Sanitize if requested
    if (rules.sanitize && typeof value === 'string') {
      value = this.sanitizeString(value);
    }

    // Validate
    const validation = this.validateField('input', value, rules);
    errors.push(...validation);

    return {
      isValid: errors.length === 0,
      value,
      errors
    };
  }

  /**
   * Validate Chainlink job ID
   */
  public isValidChainlinkJobId(jobId: string): boolean {
    if (!jobId || typeof jobId !== 'string') {
      return false;
    }

    // Chainlink job IDs are typically UUIDs or 32-byte hex strings
    return this.isValidUUID(jobId) || /^[a-fA-F0-9]{64}$/.test(jobId);
  }

  /**
   * Validate Chainlink request ID
   */
  public isValidRequestId(requestId: string): boolean {
    if (!requestId || typeof requestId !== 'string') {
      return false;
    }

    // Request IDs are typically 32-byte hex strings
    return /^0x[a-fA-F0-9]{64}$/.test(requestId);
  }

  /**
   * Create validation schema for common Chainlink data
   */
  public createChainlinkSchema(): ValidationSchema {
    return {
      chainId: {
        required: true,
        type: 'number',
        custom: (value) => this.isValidChainId(value)
      },
      contractAddress: {
        required: true,
        type: 'string',
        custom: (value) => this.isValidEthereumAddress(value)
      },
      transactionHash: {
        required: false,
        type: 'string',
        custom: (value) => this.isValidTransactionHash(value)
      },
      gasLimit: {
        required: false,
        type: 'number',
        custom: (value) => this.isValidGasLimit(value)
      },
      requestId: {
        required: false,
        type: 'string',
        custom: (value) => this.isValidRequestId(value)
      }
    };
  }
}

// Create singleton instance
export const validator = new ValidatorService();

// Export types
export type { ValidationResult, ValidationError, ValidationRule, ValidationSchema };

// Export class for testing
export { ValidatorService };

// Default export
export default validator;
