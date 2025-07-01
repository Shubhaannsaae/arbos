import { ethers } from 'ethers';
import { isChainSupported, getCCIPSelector } from './chainUtils';

export interface ValidationResult {
  isValid: boolean;
  error?: string;
}

/**
 * Validate Ethereum address
 */
export function validateAddress(address: string): ValidationResult {
  try {
    if (!address || typeof address !== 'string') {
      return { isValid: false, error: 'Address is required and must be a string' };
    }
    
    if (!ethers.isAddress(address)) {
      return { isValid: false, error: 'Invalid Ethereum address format' };
    }
    
    return { isValid: true };
  } catch (error) {
    return { isValid: false, error: 'Invalid address format' };
  }
}

/**
 * Validate transaction hash
 */
export function validateTxHash(hash: string): ValidationResult {
  if (!hash || typeof hash !== 'string') {
    return { isValid: false, error: 'Transaction hash is required' };
  }
  
  if (!ethers.isHexString(hash, 32)) {
    return { isValid: false, error: 'Invalid transaction hash format' };
  }
  
  return { isValid: true };
}

/**
 * Validate BigNumber amount
 */
export function validateAmount(amount: bigint, min?: bigint, max?: bigint): ValidationResult {
  if (amount < 0n) {
    return { isValid: false, error: 'Amount cannot be negative' };
  }
  
  if (amount === 0n) {
    return { isValid: false, error: 'Amount must be greater than zero' };
  }
  
  if (min && amount < min) {
    return { isValid: false, error: `Amount must be at least ${ethers.formatEther(min)} ETH` };
  }
  
  if (max && amount > max) {
    return { isValid: false, error: `Amount cannot exceed ${ethers.formatEther(max)} ETH` };
  }
  
  return { isValid: true };
}

/**
 * Validate chain ID
 */
export function validateChainId(chainId: number): ValidationResult {
  if (!Number.isInteger(chainId) || chainId <= 0) {
    return { isValid: false, error: 'Chain ID must be a positive integer' };
  }
  
  if (!isChainSupported(chainId)) {
    return { isValid: false, error: `Chain ID ${chainId} is not supported` };
  }
  
  return { isValid: true };
}

/**
 * Validate cross-chain route
 */
export function validateCrossChainRoute(sourceChain: number, destChain: number): ValidationResult {
  const sourceValidation = validateChainId(sourceChain);
  if (!sourceValidation.isValid) {
    return { isValid: false, error: `Source chain error: ${sourceValidation.error}` };
  }
  
  const destValidation = validateChainId(destChain);
  if (!destValidation.isValid) {
    return { isValid: false, error: `Destination chain error: ${destValidation.error}` };
  }
  
  if (sourceChain === destChain) {
    return { isValid: false, error: 'Source and destination chains cannot be the same' };
  }
  
  return { isValid: true };
}

/**
 * Validate CCIP message data
 */
export function validateCCIPMessage(
  receiver: string,
  data: string,
  tokenAmounts: { token: string; amount: bigint }[]
): ValidationResult {
  // Validate receiver
  const receiverValidation = validateAddress(receiver);
  if (!receiverValidation.isValid) {
    return { isValid: false, error: `Receiver validation failed: ${receiverValidation.error}` };
  }
  
  // Validate data format
  if (!ethers.isHexString(data)) {
    return { isValid: false, error: 'Message data must be a valid hex string' };
  }
  
  // Validate data size (CCIP has limits)
  const dataSize = ethers.getBytes(data).length;
  if (dataSize > 10000) { // 10KB limit
    return { isValid: false, error: 'Message data exceeds 10KB limit' };
  }
  
  // Validate token amounts
  for (let i = 0; i < tokenAmounts.length; i++) {
    const tokenAmount = tokenAmounts[i];
    
    const tokenValidation = validateAddress(tokenAmount.token);
    if (!tokenValidation.isValid) {
      return { isValid: false, error: `Token ${i} address invalid: ${tokenValidation.error}` };
    }
    
    const amountValidation = validateAmount(tokenAmount.amount);
    if (!amountValidation.isValid) {
      return { isValid: false, error: `Token ${i} amount invalid: ${amountValidation.error}` };
    }
  }
  
  return { isValid: true };
}

/**
 * Validate gas parameters
 */
export function validateGasParams(
  gasLimit: number,
  gasPrice?: bigint,
  maxFeePerGas?: bigint,
  maxPriorityFeePerGas?: bigint
): ValidationResult {
  if (!Number.isInteger(gasLimit) || gasLimit <= 0) {
    return { isValid: false, error: 'Gas limit must be a positive integer' };
  }
  
  if (gasLimit > 10000000) {
    return { isValid: false, error: 'Gas limit too high (max 10M)' };
  }
  
  if (gasPrice && gasPrice <= 0n) {
    return { isValid: false, error: 'Gas price must be positive' };
  }
  
  if (maxFeePerGas && maxFeePerGas <= 0n) {
    return { isValid: false, error: 'Max fee per gas must be positive' };
  }
  
  if (maxPriorityFeePerGas && maxPriorityFeePerGas <= 0n) {
    return { isValid: false, error: 'Max priority fee per gas must be positive' };
  }
  
  if (maxFeePerGas && maxPriorityFeePerGas && maxPriorityFeePerGas > maxFeePerGas) {
    return { isValid: false, error: 'Priority fee cannot exceed max fee per gas' };
  }
  
  return { isValid: true };
}

/**
 * Validate slippage tolerance
 */
export function validateSlippage(slippage: number): ValidationResult {
  if (typeof slippage !== 'number' || isNaN(slippage)) {
    return { isValid: false, error: 'Slippage must be a number' };
  }
  
  if (slippage < 0 || slippage > 10000) {
    return { isValid: false, error: 'Slippage must be between 0 and 10000 basis points' };
  }
  
  if (slippage > 1000) { // 10%
    return { isValid: false, error: 'Slippage tolerance too high (max 10%)' };
  }
  
  return { isValid: true };
}

/**
 * Validate deadline timestamp
 */
export function validateDeadline(deadline: number): ValidationResult {
  if (!Number.isInteger(deadline) || deadline <= 0) {
    return { isValid: false, error: 'Deadline must be a positive integer timestamp' };
  }
  
  const now = Math.floor(Date.now() / 1000);
  if (deadline <= now) {
    return { isValid: false, error: 'Deadline must be in the future' };
  }
  
  const maxDeadline = now + (24 * 60 * 60); // 24 hours
  if (deadline > maxDeadline) {
    return { isValid: false, error: 'Deadline too far in the future (max 24 hours)' };
  }
  
  return { isValid: true };
}

/**
 * Validate hex string with optional length check
 */
export function validateHexString(hex: string, expectedLength?: number): ValidationResult {
  if (!hex || typeof hex !== 'string') {
    return { isValid: false, error: 'Hex string is required' };
  }
  
  if (!ethers.isHexString(hex)) {
    return { isValid: false, error: 'Invalid hex string format' };
  }
  
  if (expectedLength) {
    const actualLength = ethers.getBytes(hex).length;
    if (actualLength !== expectedLength) {
      return { isValid: false, error: `Expected ${expectedLength} bytes, got ${actualLength}` };
    }
  }
  
  return { isValid: true };
}

/**
 * Validate URL format
 */
export function validateURL(url: string): ValidationResult {
  if (!url || typeof url !== 'string') {
    return { isValid: false, error: 'URL is required' };
  }
  
  try {
    new URL(url);
    return { isValid: true };
  } catch {
    return { isValid: false, error: 'Invalid URL format' };
  }
}

/**
 * Validate email format
 */
export function validateEmail(email: string): ValidationResult {
  if (!email || typeof email !== 'string') {
    return { isValid: false, error: 'Email is required' };
  }
  
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return { isValid: false, error: 'Invalid email format' };
  }
  
  return { isValid: true };
}

/**
 * Validate contract bytecode
 */
export function validateBytecode(bytecode: string): ValidationResult {
  if (!bytecode || typeof bytecode !== 'string') {
    return { isValid: false, error: 'Bytecode is required' };
  }
  
  if (!ethers.isHexString(bytecode)) {
    return { isValid: false, error: 'Bytecode must be a hex string' };
  }
  
  if (bytecode === '0x') {
    return { isValid: false, error: 'Bytecode cannot be empty' };
  }
  
  // Check for basic contract structure
  if (bytecode.length < 10) {
    return { isValid: false, error: 'Bytecode too short to be valid contract' };
  }
  
  return { isValid: true };
}

/**
 * Validate multiple conditions with aggregated results
 */
export function validateMultiple(validations: (() => ValidationResult)[]): ValidationResult {
  const errors: string[] = [];
  
  for (const validation of validations) {
    const result = validation();
    if (!result.isValid && result.error) {
      errors.push(result.error);
    }
  }
  
  if (errors.length > 0) {
    return { isValid: false, error: errors.join('; ') };
  }
  
  return { isValid: true };
}

/**
 * Sanitize user input string
 */
export function sanitizeString(input: string, maxLength: number = 1000): string {
  if (!input || typeof input !== 'string') {
    return '';
  }
  
  return input
    .slice(0, maxLength)
    .replace(/[<>]/g, '') // Remove potential XSS characters
    .trim();
}

/**
 * Validate and sanitize user input
 */
export function validateAndSanitizeInput(
  input: string,
  maxLength: number = 1000,
  required: boolean = true
): ValidationResult & { sanitized?: string } {
  if (required && (!input || typeof input !== 'string')) {
    return { isValid: false, error: 'Input is required' };
  }
  
  if (!required && !input) {
    return { isValid: true, sanitized: '' };
  }
  
  if (input.length > maxLength) {
    return { isValid: false, error: `Input too long (max ${maxLength} characters)` };
  }
  
  const sanitized = sanitizeString(input, maxLength);
  return { isValid: true, sanitized };
}
