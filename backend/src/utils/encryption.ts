import * as crypto from 'crypto';
import * as bcrypt from 'bcrypt';
import { promisify } from 'util';
import { logger } from './logger';

interface EncryptionConfig {
  algorithm: string;
  keyLength: number;
  ivLength: number;
  tagLength: number;
  saltLength: number;
  iterations: number;
  hashAlgorithm: string;
  bcryptRounds: number;
}

interface EncryptedData {
  encrypted: string;
  iv: string;
  tag: string;
  salt?: string;
}

interface KeyPair {
  publicKey: string;
  privateKey: string;
}

interface SignatureResult {
  signature: string;
  publicKey: string;
}

class EncryptionService {
  private config: EncryptionConfig;
  private masterKey: Buffer;
  private keyCache: Map<string, Buffer> = new Map();

  constructor() {
    this.config = this.loadConfiguration();
    this.masterKey = this.initializeMasterKey();
  }

  /**
   * Load encryption configuration
   */
  private loadConfiguration(): EncryptionConfig {
    return {
      algorithm: 'aes-256-gcm',
      keyLength: 32, // 256 bits
      ivLength: 16,  // 128 bits
      tagLength: 16, // 128 bits
      saltLength: 32, // 256 bits
      iterations: 100000, // PBKDF2 iterations
      hashAlgorithm: 'sha256',
      bcryptRounds: parseInt(process.env.BCRYPT_ROUNDS || '12')
    };
  }

  /**
   * Initialize master encryption key
   */
  private initializeMasterKey(): Buffer {
    const masterKeyHex = process.env.MASTER_ENCRYPTION_KEY;
    
    if (masterKeyHex) {
      if (masterKeyHex.length !== 64) { // 32 bytes = 64 hex chars
        throw new Error('Master encryption key must be 64 hex characters (32 bytes)');
      }
      return Buffer.from(masterKeyHex, 'hex');
    }

    // Generate new master key if not provided (development only)
    if (process.env.NODE_ENV === 'development') {
      const newKey = crypto.randomBytes(this.config.keyLength);
      logger.warn('Generated new master encryption key for development', {
        key: newKey.toString('hex')
      });
      return newKey;
    }

    throw new Error('MASTER_ENCRYPTION_KEY environment variable is required');
  }

  /**
   * Encrypt data using AES-256-GCM
   */
  public encrypt(data: string, context?: string): EncryptedData {
    try {
      const iv = crypto.randomBytes(this.config.ivLength);
      const key = this.deriveKey(context);
      
      const cipher = crypto.createCipherGCM(this.config.algorithm, key, iv);
      
      let encrypted = cipher.update(data, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      const tag = cipher.getAuthTag();

      const result: EncryptedData = {
        encrypted,
        iv: iv.toString('hex'),
        tag: tag.toString('hex')
      };

      logger.debug('Data encrypted successfully', {
        algorithm: this.config.algorithm,
        dataLength: data.length,
        context: context || 'default'
      });

      return result;
    } catch (error: any) {
      logger.error('Encryption failed', { error: error.message, context });
      throw new Error('Encryption failed');
    }
  }

  /**
   * Decrypt data using AES-256-GCM
   */
  public decrypt(encryptedData: EncryptedData, context?: string): string {
    try {
      const { encrypted, iv, tag } = encryptedData;
      const key = this.deriveKey(context);
      
      const decipher = crypto.createDecipherGCM(
        this.config.algorithm,
        key,
        Buffer.from(iv, 'hex')
      );
      
      decipher.setAuthTag(Buffer.from(tag, 'hex'));
      
      let decrypted = decipher.update(encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');

      logger.debug('Data decrypted successfully', {
        algorithm: this.config.algorithm,
        context: context || 'default'
      });

      return decrypted;
    } catch (error: any) {
      logger.error('Decryption failed', { error: error.message, context });
      throw new Error('Decryption failed');
    }
  }

  /**
   * Derive encryption key from master key and context
   */
  private deriveKey(context?: string): Buffer {
    const keyId = context || 'default';
    
    // Check cache first
    if (this.keyCache.has(keyId)) {
      return this.keyCache.get(keyId)!;
    }

    // Derive key using HKDF
    const salt = crypto.createHash(this.config.hashAlgorithm)
      .update(keyId)
      .digest();

    const derivedKey = crypto.hkdfSync(
      this.config.hashAlgorithm,
      this.masterKey,
      salt,
      Buffer.from(keyId, 'utf8'),
      this.config.keyLength
    );

    // Cache the derived key
    this.keyCache.set(keyId, derivedKey);

    return derivedKey;
  }

  /**
   * Hash password using bcrypt
   */
  public async hashPassword(password: string): Promise<string> {
    try {
      const hash = await bcrypt.hash(password, this.config.bcryptRounds);
      
      logger.debug('Password hashed successfully', {
        rounds: this.config.bcryptRounds
      });

      return hash;
    } catch (error: any) {
      logger.error('Password hashing failed', { error: error.message });
      throw new Error('Password hashing failed');
    }
  }

  /**
   * Verify password against hash
   */
  public async verifyPassword(password: string, hash: string): Promise<boolean> {
    try {
      const isValid = await bcrypt.compare(password, hash);
      
      logger.debug('Password verification completed', { isValid });

      return isValid;
    } catch (error: any) {
      logger.error('Password verification failed', { error: error.message });
      return false;
    }
  }

  /**
   * Generate secure random token
   */
  public generateToken(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Generate API key with metadata
   */
  public generateApiKey(userId: string, keyName: string): string {
    const timestamp = Date.now().toString();
    const randomPart = crypto.randomBytes(16).toString('hex');
    const payload = `${userId}:${keyName}:${timestamp}:${randomPart}`;
    
    return Buffer.from(payload).toString('base64url');
  }

  /**
   * Parse API key to extract metadata
   */
  public parseApiKey(apiKey: string): { userId: string; keyName: string; timestamp: number } | null {
    try {
      const payload = Buffer.from(apiKey, 'base64url').toString('utf8');
      const parts = payload.split(':');
      
      if (parts.length !== 4) {
        return null;
      }

      return {
        userId: parts[0],
        keyName: parts[1],
        timestamp: parseInt(parts[2])
      };
    } catch (error) {
      return null;
    }
  }

  /**
   * Create HMAC signature
   */
  public createSignature(data: string, secret: string): string {
    return crypto.createHmac(this.config.hashAlgorithm, secret)
      .update(data)
      .digest('hex');
  }

  /**
   * Verify HMAC signature
   */
  public verifySignature(data: string, signature: string, secret: string): boolean {
    const expectedSignature = this.createSignature(data, secret);
    return crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expectedSignature, 'hex')
    );
  }

  /**
   * Generate RSA key pair for digital signatures
   */
  public generateKeyPair(): KeyPair {
    const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
      modulusLength: 2048,
      publicKeyEncoding: {
        type: 'spki',
        format: 'pem'
      },
      privateKeyEncoding: {
        type: 'pkcs8',
        format: 'pem'
      }
    });

    logger.debug('RSA key pair generated successfully');

    return { publicKey, privateKey };
  }

  /**
   * Sign data with RSA private key
   */
  public signData(data: string, privateKey: string): string {
    try {
      const signature = crypto.sign(this.config.hashAlgorithm, Buffer.from(data), privateKey);
      
      logger.debug('Data signed successfully');

      return signature.toString('base64');
    } catch (error: any) {
      logger.error('Data signing failed', { error: error.message });
      throw new Error('Data signing failed');
    }
  }

  /**
   * Verify RSA signature
   */
  public verifySignatureRSA(data: string, signature: string, publicKey: string): boolean {
    try {
      const isValid = crypto.verify(
        this.config.hashAlgorithm,
        Buffer.from(data),
        publicKey,
        Buffer.from(signature, 'base64')
      );

      logger.debug('RSA signature verification completed', { isValid });

      return isValid;
    } catch (error: any) {
      logger.error('RSA signature verification failed', { error: error.message });
      return false;
    }
  }

  /**
   * Encrypt large data with hybrid encryption (RSA + AES)
   */
  public encryptLarge(data: string, publicKey: string): { encryptedData: EncryptedData; encryptedKey: string } {
    try {
      // Generate random AES key
      const aesKey = crypto.randomBytes(this.config.keyLength);
      
      // Encrypt data with AES
      const iv = crypto.randomBytes(this.config.ivLength);
      const cipher = crypto.createCipherGCM(this.config.algorithm, aesKey, iv);
      
      let encrypted = cipher.update(data, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      const tag = cipher.getAuthTag();

      // Encrypt AES key with RSA
      const encryptedKey = crypto.publicEncrypt(publicKey, aesKey).toString('base64');

      const encryptedData: EncryptedData = {
        encrypted,
        iv: iv.toString('hex'),
        tag: tag.toString('hex')
      };

      logger.debug('Large data encrypted with hybrid encryption');

      return { encryptedData, encryptedKey };
    } catch (error: any) {
      logger.error('Large data encryption failed', { error: error.message });
      throw new Error('Large data encryption failed');
    }
  }

  /**
   * Decrypt large data with hybrid encryption
   */
  public decryptLarge(
    encryptedData: EncryptedData,
    encryptedKey: string,
    privateKey: string
  ): string {
    try {
      // Decrypt AES key with RSA
      const aesKey = crypto.privateDecrypt(privateKey, Buffer.from(encryptedKey, 'base64'));
      
      // Decrypt data with AES
      const { encrypted, iv, tag } = encryptedData;
      const decipher = crypto.createDecipherGCM(
        this.config.algorithm,
        aesKey,
        Buffer.from(iv, 'hex')
      );
      
      decipher.setAuthTag(Buffer.from(tag, 'hex'));
      
      let decrypted = decipher.update(encrypted, 'hex', 'utf8');
      decrypted += decipher.final('utf8');

      logger.debug('Large data decrypted with hybrid encryption');

      return decrypted;
    } catch (error: any) {
      logger.error('Large data decryption failed', { error: error.message });
      throw new Error('Large data decryption failed');
    }
  }

  /**
   * Generate deterministic hash
   */
  public hash(data: string, salt?: string): string {
    const hasher = crypto.createHash(this.config.hashAlgorithm);
    hasher.update(data);
    
    if (salt) {
      hasher.update(salt);
    }

    return hasher.digest('hex');
  }

  /**
   * Generate secure random bytes
   */
  public randomBytes(size: number): Buffer {
    return crypto.randomBytes(size);
  }

  /**
   * Constant-time string comparison
   */
  public safeCompare(a: string, b: string): boolean {
    if (a.length !== b.length) {
      return false;
    }

    return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
  }

  /**
   * Encrypt wallet private key for storage
   */
  public encryptWalletKey(privateKey: string, userPassword: string): EncryptedData {
    const context = `wallet:${this.hash(userPassword)}`;
    return this.encrypt(privateKey, context);
  }

  /**
   * Decrypt wallet private key
   */
  public decryptWalletKey(encryptedData: EncryptedData, userPassword: string): string {
    const context = `wallet:${this.hash(userPassword)}`;
    return this.decrypt(encryptedData, context);
  }

  /**
   * Encrypt API credentials
   */
  public encryptCredentials(credentials: object): EncryptedData {
    const context = 'credentials';
    return this.encrypt(JSON.stringify(credentials), context);
  }

  /**
   * Decrypt API credentials
   */
  public decryptCredentials(encryptedData: EncryptedData): object {
    const context = 'credentials';
    const decrypted = this.decrypt(encryptedData, context);
    return JSON.parse(decrypted);
  }

  /**
   * Create secure session token
   */
  public createSessionToken(userId: string, expiresIn: number = 3600000): string {
    const payload = {
      userId,
      expiresAt: Date.now() + expiresIn,
      nonce: this.generateToken(16)
    };

    const token = Buffer.from(JSON.stringify(payload)).toString('base64url');
    const signature = this.createSignature(token, this.masterKey.toString('hex'));

    return `${token}.${signature}`;
  }

  /**
   * Verify session token
   */
  public verifySessionToken(token: string): { userId: string; valid: boolean } {
    try {
      const [payload, signature] = token.split('.');
      
      if (!this.verifySignature(payload, signature, this.masterKey.toString('hex'))) {
        return { userId: '', valid: false };
      }

      const decoded = JSON.parse(Buffer.from(payload, 'base64url').toString());
      
      if (Date.now() > decoded.expiresAt) {
        return { userId: decoded.userId, valid: false };
      }

      return { userId: decoded.userId, valid: true };
    } catch (error) {
      return { userId: '', valid: false };
    }
  }

  /**
   * Clear encryption cache
   */
  public clearCache(): void {
    this.keyCache.clear();
    logger.debug('Encryption cache cleared');
  }

  /**
   * Get encryption statistics
   */
  public getStats(): any {
    return {
      algorithm: this.config.algorithm,
      keyLength: this.config.keyLength,
      cacheSize: this.keyCache.size,
      bcryptRounds: this.config.bcryptRounds
    };
  }
}

// Create singleton instance
export const encryption = new EncryptionService();

// Export types
export type { EncryptedData, KeyPair, SignatureResult };

// Export class for testing
export { EncryptionService };

// Default export
export default encryption;
