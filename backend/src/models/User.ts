export interface User {
  id: string;
  walletAddress: string;
  email?: string;
  preferences: UserPreferences;
  riskTolerance: RiskTolerance;
  subscriptionTier: SubscriptionTier;
  apiKeys?: ApiKeys;
  createdAt: Date;
  updatedAt: Date;
  isActive: boolean;
  kycStatus: KYCStatus;
}

export interface UserPreferences {
  notifications: {
    email: boolean;
    push: boolean;
    sms: boolean;
    arbitrageAlerts: boolean;
    portfolioRebalancing: boolean;
    securityAlerts: boolean;
  };
  trading: {
    maxSlippage: number;
    autoRebalance: boolean;
    rebalanceThreshold: number;
    preferredDexes: string[];
    blacklistedTokens: string[];
  };
  risk: {
    maxPositionSize: number;
    stopLossPercentage: number;
    takeProfitPercentage: number;
    allowLeverage: boolean;
    maxLeverage: number;
  };
}

export interface ApiKeys {
  chainlinkNodeUrl?: string;
  exchangeApiKeys?: {
    [exchange: string]: {
      apiKey: string;
      secretKey: string;
      passphrase?: string;
    };
  };
}

export enum RiskTolerance {
  CONSERVATIVE = 'conservative',
  MODERATE = 'moderate',
  AGGRESSIVE = 'aggressive',
  VERY_AGGRESSIVE = 'very_aggressive'
}

export enum SubscriptionTier {
  FREE = 'free',
  BASIC = 'basic',
  PREMIUM = 'premium',
  ENTERPRISE = 'enterprise'
}

export enum KYCStatus {
  PENDING = 'pending',
  VERIFIED = 'verified',
  REJECTED = 'rejected',
  NOT_REQUIRED = 'not_required'
}

export interface CreateUserDto {
  walletAddress: string;
  email?: string;
  riskTolerance?: RiskTolerance;
  preferences?: Partial<UserPreferences>;
}

export interface UpdateUserDto {
  email?: string;
  preferences?: Partial<UserPreferences>;
  riskTolerance?: RiskTolerance;
  subscriptionTier?: SubscriptionTier;
}
