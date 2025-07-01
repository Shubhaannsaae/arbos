import { ethers } from 'ethers';

/**
 * Format currency values
 */
export function formatCurrency(
  value: number | bigint | string,
  decimals: number = 2,
  symbol: string = '$'
): string {
  let numValue: number;
  
  if (typeof value === 'bigint') {
    numValue = Number(ethers.formatEther(value));
  } else if (typeof value === 'string') {
    numValue = parseFloat(value);
  } else {
    numValue = value;
  }

  if (isNaN(numValue)) return `${symbol}0.00`;

  return `${symbol}${numValue.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  })}`;
}

/**
 * Format percentage values
 */
export function formatPercentage(
  value: number,
  decimals: number = 2,
  showSign: boolean = true
): string {
  if (isNaN(value)) return '0.00%';
  
  const formatted = value.toFixed(decimals);
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${formatted}%`;
}

/**
 * Format token amounts with proper decimals
 */
export function formatTokenAmount(
  amount: bigint | string,
  decimals: number = 18,
  symbol?: string,
  displayDecimals: number = 4
): string {
  let value: bigint;
  
  if (typeof amount === 'string') {
    value = BigInt(amount);
  } else {
    value = amount;
  }

  const formatted = ethers.formatUnits(value, decimals);
  const numValue = parseFloat(formatted);
  
  const displayValue = numValue.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: displayDecimals
  });

  return symbol ? `${displayValue} ${symbol}` : displayValue;
}

/**
 * Format ETH values
 */
export function formatEther(
  value: bigint | string,
  decimals: number = 4,
  showSymbol: boolean = true
): string {
  const formatted = ethers.formatEther(value);
  const numValue = parseFloat(formatted);
  
  const displayValue = numValue.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals
  });

  return showSymbol ? `${displayValue} ETH` : displayValue;
}

/**
 * Format wei to gwei for gas prices
 */
export function formatGwei(weiValue: bigint | string): string {
  const gwei = ethers.formatUnits(weiValue, 'gwei');
  const numValue = parseFloat(gwei);
  
  return `${numValue.toFixed(2)} gwei`;
}

/**
 * Format large numbers with K, M, B suffixes
 */
export function formatCompactNumber(
  value: number,
  decimals: number = 1
): string {
  if (isNaN(value)) return '0';
  
  const absValue = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  
  if (absValue >= 1e9) {
    return `${sign}${(absValue / 1e9).toFixed(decimals)}B`;
  } else if (absValue >= 1e6) {
    return `${sign}${(absValue / 1e6).toFixed(decimals)}M`;
  } else if (absValue >= 1e3) {
    return `${sign}${(absValue / 1e3).toFixed(decimals)}K`;
  }
  
  return `${sign}${absValue.toFixed(decimals)}`;
}

/**
 * Format time duration
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`;
  } else if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m`;
  } else if (seconds < 86400) {
    return `${Math.round(seconds / 3600)}h`;
  } else {
    return `${Math.round(seconds / 86400)}d`;
  }
}

/**
 * Format date and time
 */
export function formatDateTime(
  timestamp: number | Date,
  options: Intl.DateTimeFormatOptions = {}
): string {
  const date = typeof timestamp === 'number' ? new Date(timestamp) : timestamp;
  
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    ...options
  };

  return date.toLocaleDateString('en-US', defaultOptions);
}

/**
 * Format date only
 */
export function formatDate(
  timestamp: number | Date,
  options: Intl.DateTimeFormatOptions = {}
): string {
  const date = typeof timestamp === 'number' ? new Date(timestamp) : timestamp;
  
  const defaultOptions: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    ...options
  };

  return date.toLocaleDateString('en-US', defaultOptions);
}

/**
 * Format time only
 */
export function formatTime(
  timestamp: number | Date,
  options: Intl.DateTimeFormatOptions = {}
): string {
  const date = typeof timestamp === 'number' ? new Date(timestamp) : timestamp;
  
  const defaultOptions: Intl.DateTimeFormatOptions = {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    ...options
  };

  return date.toLocaleTimeString('en-US', defaultOptions);
}

/**
 * Format address for display
 */
export function formatAddress(address: string, startChars: number = 6, endChars: number = 4): string {
  if (!address || address.length < startChars + endChars) {
    return address;
  }
  
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`;
}

/**
 * Format transaction hash
 */
export function formatTxHash(hash: string, chars: number = 8): string {
  return formatAddress(hash, chars, chars);
}

/**
 * Format gas amount
 */
export function formatGas(gasUsed: number | bigint, gasLimit?: number | bigint): string {
  const used = typeof gasUsed === 'bigint' ? Number(gasUsed) : gasUsed;
  
  if (gasLimit) {
    const limit = typeof gasLimit === 'bigint' ? Number(gasLimit) : gasLimit;
    const percentage = (used / limit) * 100;
    return `${used.toLocaleString()} (${percentage.toFixed(1)}%)`;
  }
  
  return used.toLocaleString();
}

/**
 * Format APY/APR values
 */
export function formatAPY(value: number, decimals: number = 2): string {
  if (isNaN(value)) return '0.00%';
  
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format price with appropriate precision
 */
export function formatPrice(
  price: number,
  currency: string = 'USD'
): string {
  if (isNaN(price)) return '$0.00';
  
  let decimals = 2;
  
  // Adjust decimals based on price magnitude
  if (price < 0.01) {
    decimals = 6;
  } else if (price < 1) {
    decimals = 4;
  }
  
  const symbol = currency === 'USD' ? '$' : '';
  return `${symbol}${price.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  })}`;
}

/**
 * Format market cap
 */
export function formatMarketCap(value: number): string {
  return formatCurrency(value, 0);
}

/**
 * Format volume
 */
export function formatVolume(value: number): string {
  return formatCompactNumber(value, 1);
}

/**
 * Format ratio values (like price ratios)
 */
export function formatRatio(value: number, decimals: number = 4): string {
  if (isNaN(value)) return '0.0000';
  
  return value.toFixed(decimals);
}

/**
 * Format basis points to percentage
 */
export function formatBasisPoints(bps: number): string {
  const percentage = bps / 100;
  return formatPercentage(percentage);
}

/**
 * Format slippage
 */
export function formatSlippage(value: number): string {
  return `${value.toFixed(2)}%`;
}

/**
 * Parse and format user input for token amounts
 */
export function parseTokenInput(
  input: string,
  decimals: number = 18
): { value: string; isValid: boolean; error?: string } {
  try {
    // Remove any non-numeric characters except decimal point
    const cleaned = input.replace(/[^\d.]/g, '');
    
    // Check for multiple decimal points
    const decimalCount = (cleaned.match(/\./g) || []).length;
    if (decimalCount > 1) {
      return { value: input, isValid: false, error: 'Invalid decimal format' };
    }
    
    // Check decimal places
    const decimalIndex = cleaned.indexOf('.');
    if (decimalIndex !== -1 && cleaned.length - decimalIndex - 1 > decimals) {
      return { value: input, isValid: false, error: `Too many decimal places (max ${decimals})` };
    }
    
    const numValue = parseFloat(cleaned);
    if (isNaN(numValue) || numValue < 0) {
      return { value: input, isValid: false, error: 'Invalid number' };
    }
    
    return { value: cleaned, isValid: true };
  } catch (error) {
    return { value: input, isValid: false, error: 'Parsing error' };
  }
}
