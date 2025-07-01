import { ethers, BigNumber, Contract } from 'ethers';
import axios from 'axios';
import { logger } from '../../../shared/utils/logger';
import { PriceData, TokenInfo } from '../../../shared/types/market';
import { getNetworkConfig, getProvider } from '../../../config/agentConfig';

export interface MarketSummary {
  totalMarketCap: BigNumber;
  total24hVolume: BigNumber;
  btcDominance: number;
  ethDominance: number;
  fearGreedIndex: number;
  volatilityIndex: number;
  activeCoins: number;
  marketTrend: 'bullish' | 'bearish' | 'neutral';
}

export interface AssetMetadata {
  symbol: string;
  name: string;
  marketCap: BigNumber;
  circulatingSupply: BigNumber;
  totalSupply: BigNumber;
  maxSupply?: BigNumber;
  averageDailyVolume: BigNumber;
  sector: string;
  website: string;
  description: string;
  socialLinks: {
    twitter?: string;
    discord?: string;
    telegram?: string;
  };
  technicalIndicators: {
    rsi: number;
    macd: number;
    sma20: BigNumber;
    sma50: BigNumber;
    sma200: BigNumber;
    bollingerBands: {
      upper: BigNumber;
      middle: BigNumber;
      lower: BigNumber;
    };
  };
}

export interface LiquidityMetrics {
  averageDailyVolume: BigNumber;
  bidAskSpread: number;
  marketDepth: BigNumber;
  liquidityScore: number;
  slippageEstimates: {
    '1000': number;    // $1k trade slippage
    '10000': number;   // $10k trade slippage
    '100000': number;  // $100k trade slippage
  };
}

export interface VolatilityData {
  annualizedVolatility: number;
  historicalVolatility: {
    '7d': number;
    '30d': number;
    '90d': number;
    '1y': number;
  };
  impliedVolatility?: number;
  volatilityRank: number; // 0-100 percentile rank
}

export interface BetaData {
  beta: number;
  correlation: number;
  r_squared: number;
  alpha: number;
  benchmark: string;
  period: string;
}

export class MarketDataProvider {
  private providers: Map<number, ethers.JsonRpcProvider> = new Map();
  private dataCache: Map<string, any> = new Map();
  private priceFeeds: Map<string, Contract> = new Map();
  private apiKeys: {
    coingecko: string;
    coinmarketcap: string;
    messari: string;
    dune: string;
    theGraph: string;
  };

  // Chainlink Price Feed ABI
  private readonly PRICE_FEED_ABI = [
    "function latestRoundData() external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound)",
    "function decimals() external view returns (uint8)",
    "function description() external view returns (string memory)"
  ];

  // Chainlink Price Feed Addresses (Production mainnet addresses)
  private readonly PRICE_FEEDS: Record<number, Record<string, string>> = {
    1: { // Ethereum Mainnet
      'ETH/USD': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
      'BTC/USD': '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',
      'LINK/USD': '0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c',
      'USDC/USD': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6',
      'USDT/USD': '0x3E7d1eAB13ad0104d2750B8863b489D65364e32D',
      'UNI/USD': '0x553303d460EE0afB37EdFf9bE42922D8FF63220e',
      'AAVE/USD': '0x547a514d5e3769680Ce22B2361c10Ea13619e8a9',
      'COMP/USD': '0xdbd020CAeF83eFd542f4De03e3cF0C28A4428bd5',
      'MKR/USD': '0xec1D1B3b0443256cc3860e24a46F108e699484Aa'
    },
    137: { // Polygon
      'MATIC/USD': '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0',
      'ETH/USD': '0xF9680D99D6C9589e2a93a78A04A279e509205945',
      'BTC/USD': '0xc907E116054Ad103354f2D350FD2514433D57F6f',
      'LINK/USD': '0xd9FFdb71EbE7496cC440152d43986Aae0AB76665',
      'USDC/USD': '0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7',
      'USDT/USD': '0x0A6513e40db6EB1b165753AD52E80663aeA50545'
    },
    42161: { // Arbitrum
      'ETH/USD': '0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612',
      'BTC/USD': '0x6ce185860a4963106506C203335A2910413708e9',
      'LINK/USD': '0x86E53CF1B870786351Da77A57575e79CB55812CB',
      'USDC/USD': '0x50834F3163758fcC1Df9973b6e91f0F0F0434aD3',
      'USDT/USD': '0x3f3f5dF88dC9F13eac63DF89EC16ef6e7E25DdE7'
    },
    43114: { // Avalanche
      'AVAX/USD': '0x0A77230d17318075983913bC2145DB16C7366156',
      'ETH/USD': '0x976B3D034E162d8bD72D6b9C989d545b839003b0',
      'BTC/USD': '0x2779D32d5166BAaa2B2b658333bA7e6Ec0C65743',
      'LINK/USD': '0x49ccd9ca821EfEab2b98c60dC60F518E765EDe9a',
      'USDC/USD': '0xF096872672F44d6EBA71458D74fe67F9a77a23B9'
    }
  };

  constructor() {
    this.apiKeys = {
      coingecko: process.env.COINGECKO_API_KEY || '',
      coinmarketcap: process.env.COINMARKETCAP_API_KEY || '',
      messari: process.env.MESSARI_API_KEY || '',
      dune: process.env.DUNE_API_KEY || '',
      theGraph: process.env.THEGRAPH_API_KEY || ''
    };

    this.initializeProviders();
  }

  async initialize(): Promise<void> {
    logger.info('Initializing market data provider');

    await this.initializeChainlinkFeeds();
    await this.validateDataSources();

    logger.info('Market data provider initialized successfully');
  }

  private initializeProviders(): void {
    const supportedChains = [1, 137, 42161, 43114]; // ETH, Polygon, Arbitrum, Avalanche

    supportedChains.forEach(chainId => {
      const provider = getProvider(chainId);
      this.providers.set(chainId, provider);
    });
  }

  private async initializeChainlinkFeeds(): Promise<void> {
    for (const [chainId, feeds] of Object.entries(this.PRICE_FEEDS)) {
      const provider = this.providers.get(parseInt(chainId));
      if (!provider) continue;

      for (const [pair, address] of Object.entries(feeds)) {
        try {
          const contract = new Contract(address, this.PRICE_FEED_ABI, provider);
          this.priceFeeds.set(`${pair}_${chainId}`, contract);

          logger.debug('Chainlink price feed initialized', {
            pair,
            chainId,
            address
          });

        } catch (error) {
          logger.error('Failed to initialize price feed', {
            pair,
            chainId,
            address,
            error: error instanceof Error ? error.message : String(error)
          });
        }
      }
    }
  }

  private async validateDataSources(): Promise<void> {
    const validationPromises = [
      this.testCoingeckoAPI(),
      this.testChainlinkFeeds(),
      this.testTheGraphAPI()
    ];

    const results = await Promise.allSettled(validationPromises);
    
    results.forEach((result, index) => {
      const sources = ['CoinGecko', 'Chainlink', 'TheGraph'];
      if (result.status === 'rejected') {
        logger.warn(`${sources[index]} validation failed`, {
          error: result.reason
        });
      }
    });
  }

  async getCurrentPrices(symbols: string[]): Promise<Record<string, BigNumber>> {
    const prices: Record<string, BigNumber> = {};

    try {
      // First try Chainlink price feeds for accurate prices
      for (const symbol of symbols) {
        const chainlinkPrice = await this.getChainlinkPrice(symbol);
        if (chainlinkPrice) {
          prices[symbol] = chainlinkPrice.answer;
          continue;
        }

        // Fallback to CoinGecko API
        const geckoPrice = await this.getCoinGeckoPrice(symbol);
        if (geckoPrice) {
          prices[symbol] = ethers.utils.parseEther(geckoPrice.toString());
        }
      }

      logger.debug('Current prices fetched', {
        symbols,
        priceCount: Object.keys(prices).length
      });

      return prices;

    } catch (error) {
      logger.error('Failed to get current prices', {
        symbols,
        error: error instanceof Error ? error.message : String(error)
      });

      return {};
    }
  }

  async getChainlinkPrice(symbol: string, chainId: number = 1): Promise<{
    answer: BigNumber;
    decimals: number;
    updatedAt: number;
  } | null> {
    try {
      const pairKey = `${symbol}/USD`;
      const feedKey = `${pairKey}_${chainId}`;
      const priceFeed = this.priceFeeds.get(feedKey);

      if (!priceFeed) {
        logger.debug('Chainlink price feed not found', { symbol, chainId });
        return null;
      }

      const [roundData, decimals] = await Promise.all([
        priceFeed.latestRoundData(),
        priceFeed.decimals()
      ]);

      return {
        answer: roundData.answer,
        decimals,
        updatedAt: roundData.updatedAt.toNumber()
      };

    } catch (error) {
      logger.error('Failed to get Chainlink price', {
        symbol,
        chainId,
        error: error instanceof Error ? error.message : String(error)
      });

      return null;
    }
  }

  async getCoinGeckoPrice(symbol: string): Promise<number | null> {
    try {
      const cacheKey = `gecko_price_${symbol}`;
      const cached = this.dataCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < 60000) { // 1 minute cache
        return cached.price;
      }

      const coinId = this.getCoinGeckoId(symbol);
      if (!coinId) return null;

      const url = `https://api.coingecko.com/api/v3/simple/price`;
      const params = {
        ids: coinId,
        vs_currencies: 'usd',
        include_24hr_change: 'true',
        include_market_cap: 'true',
        include_24hr_vol: 'true'
      };

      const headers: any = {
        'Accept': 'application/json',
        'User-Agent': 'ArbOS-Agents/1.0'
      };

      if (this.apiKeys.coingecko) {
        headers['x-cg-pro-api-key'] = this.apiKeys.coingecko;
      }

      const response = await axios.get(url, { params, headers, timeout: 10000 });
      const price = response.data[coinId]?.usd;

      if (price) {
        this.dataCache.set(cacheKey, {
          price,
          timestamp: Date.now(),
          data: response.data[coinId]
        });
      }

      return price;

    } catch (error) {
      logger.error('Failed to get CoinGecko price', {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      return null;
    }
  }

  async getHistoricalPrices(symbol: string, days: number): Promise<PriceData[]> {
    try {
      const cacheKey = `historical_${symbol}_${days}`;
      const cached = this.dataCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < 3600000) { // 1 hour cache
        return cached.data;
      }

      const coinId = this.getCoinGeckoId(symbol);
      if (!coinId) return [];

      const url = `https://api.coingecko.com/api/v3/coins/${coinId}/market_chart`;
      const params = {
        vs_currency: 'usd',
        days: days.toString(),
        interval: days > 90 ? 'daily' : 'hourly'
      };

      const headers: any = {
        'Accept': 'application/json'
      };

      if (this.apiKeys.coingecko) {
        headers['x-cg-pro-api-key'] = this.apiKeys.coingecko;
      }

      const response = await axios.get(url, { params, headers, timeout: 15000 });
      
      const priceData: PriceData[] = response.data.prices.map(([timestamp, price]: [number, number]) => ({
        token: symbol,
        price: ethers.utils.parseEther(price.toString()),
        priceUsd: price,
        timestamp: Math.floor(timestamp / 1000),
        source: 'coingecko',
        confidence: 1.0,
        volume24h: BigNumber.from(0), // Would need separate call for volume
        change24h: 0 // Would need calculation
      }));

      this.dataCache.set(cacheKey, {
        data: priceData,
        timestamp: Date.now()
      });

      return priceData;

    } catch (error) {
      logger.error('Failed to get historical prices', {
        symbol,
        days,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  async getHistoricalReturns(symbol: string, days: number): Promise<number[]> {
    try {
      const priceData = await this.getHistoricalPrices(symbol, days);
      
      if (priceData.length < 2) return [];

      const returns: number[] = [];
      for (let i = 1; i < priceData.length; i++) {
        const prevPrice = parseFloat(ethers.utils.formatEther(priceData[i - 1].price));
        const currentPrice = parseFloat(ethers.utils.formatEther(priceData[i].price));
        const return_ = (currentPrice - prevPrice) / prevPrice;
        returns.push(return_);
      }

      return returns;

    } catch (error) {
      logger.error('Failed to calculate historical returns', {
        symbol,
        days,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  async getCorrelationMatrix(symbols: string[]): Promise<Record<string, Record<string, number>>> {
    try {
      const cacheKey = `correlation_${symbols.sort().join('_')}`;
      const cached = this.dataCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < 3600000) { // 1 hour cache
        return cached.data;
      }

      // Get historical returns for all symbols
      const allReturns: Record<string, number[]> = {};
      
      for (const symbol of symbols) {
        const returns = await this.getHistoricalReturns(symbol, 90); // 90 days
        allReturns[symbol] = returns;
      }

      // Calculate correlation matrix
      const correlationMatrix: Record<string, Record<string, number>> = {};
      
      for (const symbol1 of symbols) {
        correlationMatrix[symbol1] = {};
        
        for (const symbol2 of symbols) {
          if (symbol1 === symbol2) {
            correlationMatrix[symbol1][symbol2] = 1.0;
          } else {
            const correlation = this.calculateCorrelation(
              allReturns[symbol1] || [],
              allReturns[symbol2] || []
            );
            correlationMatrix[symbol1][symbol2] = correlation;
          }
        }
      }

      this.dataCache.set(cacheKey, {
        data: correlationMatrix,
        timestamp: Date.now()
      });

      return correlationMatrix;

    } catch (error) {
      logger.error('Failed to get correlation matrix', {
        symbols,
        error: error instanceof Error ? error.message : String(error)
      });

      return {};
    }
  }

  async getMarketSummary(): Promise<MarketSummary> {
    try {
      const cacheKey = 'market_summary';
      const cached = this.dataCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
        return cached.data;
      }

      const url = 'https://api.coingecko.com/api/v3/global';
      const headers: any = { 'Accept': 'application/json' };

      if (this.apiKeys.coingecko) {
        headers['x-cg-pro-api-key'] = this.apiKeys.coingecko;
      }

      const response = await axios.get(url, { headers, timeout: 10000 });
      const data = response.data.data;

      // Get Fear & Greed Index
      const fearGreedIndex = await this.getFearGreedIndex();

      const summary: MarketSummary = {
        totalMarketCap: ethers.utils.parseEther(data.total_market_cap.usd.toString()),
        total24hVolume: ethers.utils.parseEther(data.total_volume.usd.toString()),
        btcDominance: data.market_cap_percentage.btc,
        ethDominance: data.market_cap_percentage.eth,
        fearGreedIndex: fearGreedIndex || 50,
        volatilityIndex: this.calculateVolatilityIndex(data),
        activeCoins: data.active_cryptocurrencies,
        marketTrend: this.determineMarketTrend(data)
      };

      this.dataCache.set(cacheKey, {
        data: summary,
        timestamp: Date.now()
      });

      return summary;

    } catch (error) {
      logger.error('Failed to get market summary', {
        error: error instanceof Error ? error.message : String(error)
      });

      // Return default values
      return {
        totalMarketCap: ethers.utils.parseEther('2000000000000'), // $2T
        total24hVolume: ethers.utils.parseEther('100000000000'), // $100B
        btcDominance: 45,
        ethDominance: 18,
        fearGreedIndex: 50,
        volatilityIndex: 50,
        activeCoins: 10000,
        marketTrend: 'neutral'
      };
    }
  }

  async getAssetMetadata(symbol: string): Promise<AssetMetadata> {
    try {
      const cacheKey = `metadata_${symbol}`;
      const cached = this.dataCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < 3600000) { // 1 hour cache
        return cached.data;
      }

      const coinId = this.getCoinGeckoId(symbol);
      if (!coinId) throw new Error(`Unknown coin ID for ${symbol}`);

      const url = `https://api.coingecko.com/api/v3/coins/${coinId}`;
      const params = {
        localization: 'false',
        tickers: 'false',
        market_data: 'true',
        community_data: 'true',
        developer_data: 'false',
        sparkline: 'false'
      };

      const headers: any = { 'Accept': 'application/json' };
      if (this.apiKeys.coingecko) {
        headers['x-cg-pro-api-key'] = this.apiKeys.coingecko;
      }

      const response = await axios.get(url, { params, headers, timeout: 15000 });
      const data = response.data;

      // Get technical indicators
      const technicalIndicators = await this.getTechnicalIndicators(symbol);

      const metadata: AssetMetadata = {
        symbol: data.symbol.toUpperCase(),
        name: data.name,
        marketCap: ethers.utils.parseEther((data.market_data.market_cap?.usd || 0).toString()),
        circulatingSupply: ethers.utils.parseEther((data.market_data.circulating_supply || 0).toString()),
        totalSupply: ethers.utils.parseEther((data.market_data.total_supply || 0).toString()),
        maxSupply: data.market_data.max_supply ? 
          ethers.utils.parseEther(data.market_data.max_supply.toString()) : undefined,
        averageDailyVolume: ethers.utils.parseEther((data.market_data.total_volume?.usd || 0).toString()),
        sector: this.categorizeAsset(symbol),
        website: data.links?.homepage?.[0] || '',
        description: data.description?.en || '',
        socialLinks: {
          twitter: data.links?.twitter_screen_name ? `https://twitter.com/${data.links.twitter_screen_name}` : undefined,
          discord: data.links?.chat_url?.find((url: string) => url.includes('discord')) || undefined,
          telegram: data.links?.telegram_channel_identifier ? 
            `https://t.me/${data.links.telegram_channel_identifier}` : undefined
        },
        technicalIndicators
      };

      this.dataCache.set(cacheKey, {
        data: metadata,
        timestamp: Date.now()
      });

      return metadata;

    } catch (error) {
      logger.error('Failed to get asset metadata', {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      throw error;
    }
  }

  async getLiquidityMetrics(symbol: string): Promise<LiquidityMetrics> {
    try {
      const cacheKey = `liquidity_${symbol}`;
      const cached = this.dataCache.get(cacheKey);

      if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
        return cached.data;
      }

      // Get orderbook data from DEX aggregators
      const liquidityData = await this.getOrderbookLiquidity(symbol);
      
      const metrics: LiquidityMetrics = {
        averageDailyVolume: liquidityData.volume24h,
        bidAskSpread: liquidityData.spread,
        marketDepth: liquidityData.depth,
        liquidityScore: this.calculateLiquidityScore(liquidityData),
        slippageEstimates: liquidityData.slippageEstimates
      };

      this.dataCache.set(cacheKey, {
        data: metrics,
        timestamp: Date.now()
      });

      return metrics;

    } catch (error) {
      logger.error('Failed to get liquidity metrics', {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      // Return default metrics
      return {
        averageDailyVolume: BigNumber.from(0),
        bidAskSpread: 0.5,
        marketDepth: BigNumber.from(0),
        liquidityScore: 50,
        slippageEstimates: {
          '1000': 0.1,
          '10000': 0.5,
          '100000': 2.0
        }
      };
    }
  }

  async getAssetVolatility(symbol: string): Promise<VolatilityData> {
    try {
      const returns = await this.getHistoricalReturns(symbol, 252); // 1 year
      
      if (returns.length < 30) {
        throw new Error('Insufficient data for volatility calculation');
      }

      // Calculate different period volatilities
      const volatilities = {
        '7d': this.calculateVolatility(returns.slice(-7)),
        '30d': this.calculateVolatility(returns.slice(-30)),
        '90d': this.calculateVolatility(returns.slice(-90)),
        '1y': this.calculateVolatility(returns)
      };

      const annualizedVolatility = volatilities['1y'];
      
      // Calculate volatility rank (percentile compared to other assets)
      const volatilityRank = await this.calculateVolatilityRank(symbol, annualizedVolatility);

      return {
        annualizedVolatility,
        historicalVolatility: volatilities,
        volatilityRank
      };

    } catch (error) {
      logger.error('Failed to get asset volatility', {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        annualizedVolatility: 0.15, // Default 15%
        historicalVolatility: {
          '7d': 0.15,
          '30d': 0.15,
          '90d': 0.15,
          '1y': 0.15
        },
        volatilityRank: 50
      };
    }
  }

  async getAssetBeta(symbol: string, benchmark: string = 'ETH'): Promise<BetaData> {
    try {
      const [assetReturns, benchmarkReturns] = await Promise.all([
        this.getHistoricalReturns(symbol, 252),
        this.getHistoricalReturns(benchmark, 252)
      ]);

      const minLength = Math.min(assetReturns.length, benchmarkReturns.length);
      
      if (minLength < 50) {
        throw new Error('Insufficient data for beta calculation');
      }

      const asset = assetReturns.slice(-minLength);
      const bench = benchmarkReturns.slice(-minLength);

      const beta = this.calculateBeta(asset, bench);
      const correlation = this.calculateCorrelation(asset, bench);
      const rSquared = correlation * correlation;
      
      // Calculate alpha (Jensen's Alpha)
      const riskFreeRate = 0.03 / 252; // Daily risk-free rate
      const assetMean = asset.reduce((sum, r) => sum + r, 0) / asset.length;
      const benchMean = bench.reduce((sum, r) => sum + r, 0) / bench.length;
      const alpha = (assetMean - riskFreeRate) - beta * (benchMean - riskFreeRate);

      return {
        beta,
        correlation,
        r_squared: rSquared,
        alpha: alpha * 252, // Annualized
        benchmark,
        period: '1Y'
      };

    } catch (error) {
      logger.error('Failed to calculate asset beta', {
        symbol,
        benchmark,
        error: error instanceof Error ? error.message : String(error)
      });

      return {
        beta: 1.0,
        correlation: 0.8,
        r_squared: 0.64,
        alpha: 0,
        benchmark,
        period: '1Y'
      };
    }
  }

  async getTopAssetsByMarketCap(limit: number = 50): Promise<Array<{
    symbol: string;
    name: string;
    marketCap: BigNumber;
    price: number;
    change24h: number;
    volume24h: BigNumber;
    sector: string;
    averageDailyVolume: BigNumber;
  }>> {
    try {
      const url = 'https://api.coingecko.com/api/v3/coins/markets';
      const params = {
        vs_currency: 'usd',
        order: 'market_cap_desc',
        per_page: limit.toString(),
        page: '1',
        sparkline: 'false',
        price_change_percentage: '24h'
      };

      const headers: any = { 'Accept': 'application/json' };
      if (this.apiKeys.coingecko) {
        headers['x-cg-pro-api-key'] = this.apiKeys.coingecko;
      }

      const response = await axios.get(url, { params, headers, timeout: 15000 });
      
      return response.data.map((coin: any) => ({
        symbol: coin.symbol.toUpperCase(),
        name: coin.name,
        marketCap: ethers.utils.parseEther((coin.market_cap || 0).toString()),
        price: coin.current_price || 0,
        change24h: coin.price_change_percentage_24h || 0,
        volume24h: ethers.utils.parseEther((coin.total_volume || 0).toString()),
        sector: this.categorizeAsset(coin.symbol.toUpperCase()),
        averageDailyVolume: ethers.utils.parseEther((coin.total_volume || 0).toString())
      }));

    } catch (error) {
      logger.error('Failed to get top assets by market cap', {
        limit,
        error: error instanceof Error ? error.message : String(error)
      });

      return [];
    }
  }

  async getTokenInfo(symbol: string): Promise<TokenInfo | null> {
    try {
      const metadata = await this.getAssetMetadata(symbol);
      
      return {
        address: '', // Would need to be fetched from token registry
        symbol: metadata.symbol,
        name: metadata.name,
        decimals: 18, // Default, would need actual contract call
        chainId: 1, // Default to Ethereum
        tags: [metadata.sector],
        isStable: this.isStablecoin(symbol),
        isNative: this.isNativeToken(symbol)
      };

    } catch (error) {
      logger.error('Failed to get token info', {
        symbol,
        error: error instanceof Error ? error.message : String(error)
      });

      return null;
    }
  }

  async getSwapQuote(fromToken: string, toToken: string, amount: BigNumber): Promise<BigNumber> {
    try {
      // This would integrate with DEX aggregators like 1inch, 0x, etc.
      // For now, use price ratio calculation
      const [fromPrice, toPrice] = await Promise.all([
        this.getCoinGeckoPrice(fromToken),
        this.getCoinGeckoPrice(toToken)
      ]);

      if (!fromPrice || !toPrice) {
        throw new Error('Price data unavailable');
      }

      const amountNum = parseFloat(ethers.utils.formatEther(amount));
      const outputAmount = (amountNum * fromPrice) / toPrice;
      
      return ethers.utils.parseEther(outputAmount.toString());

    } catch (error) {
      logger.error('Failed to get swap quote', {
        fromToken,
        toToken,
        amount: ethers.utils.formatEther(amount),
        error: error instanceof Error ? error.message : String(error)
      });

      return BigNumber.from(0);
    }
  }

  // Private helper methods
  private getCoinGeckoId(symbol: string): string | null {
    const mapping: Record<string, string> = {
      'ETH': 'ethereum',
      'BTC': 'bitcoin',
      'LINK': 'chainlink',
      'USDC': 'usd-coin',
      'USDT': 'tether',
      'MATIC': 'matic-network',
      'AVAX': 'avalanche-2',
      'UNI': 'uniswap',
      'AAVE': 'aave',
      'COMP': 'compound-governance-token',
      'MKR': 'maker',
      'SNX': 'havven',
      'CRV': 'curve-dao-token',
      'BAL': 'balancer',
      'YFI': 'yearn-finance'
    };

    return mapping[symbol.toUpperCase()] || null;
  }

  private calculateCorrelation(returns1: number[], returns2: number[]): number {
    const n = Math.min(returns1.length, returns2.length);
    if (n < 2) return 0;

    const mean1 = returns1.slice(0, n).reduce((sum, r) => sum + r, 0) / n;
    const mean2 = returns2.slice(0, n).reduce((sum, r) => sum + r, 0) / n;

    let numerator = 0;
    let sum1Sq = 0;
    let sum2Sq = 0;

    for (let i = 0; i < n; i++) {
      const diff1 = returns1[i] - mean1;
      const diff2 = returns2[i] - mean2;
      
      numerator += diff1 * diff2;
      sum1Sq += diff1 * diff1;
      sum2Sq += diff2 * diff2;
    }

    const denominator = Math.sqrt(sum1Sq * sum2Sq);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private calculateBeta(assetReturns: number[], benchmarkReturns: number[]): number {
    const correlation = this.calculateCorrelation(assetReturns, benchmarkReturns);
    const assetVol = this.calculateVolatility(assetReturns);
    const benchmarkVol = this.calculateVolatility(benchmarkReturns);
    
    return benchmarkVol === 0 ? 1 : correlation * (assetVol / benchmarkVol);
  }

  private calculateVolatility(returns: number[]): number {
    if (returns.length < 2) return 0;
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance * 252); // Annualized
  }

  private async getFearGreedIndex(): Promise<number | null> {
    try {
      const url = 'https://api.alternative.me/fng/';
      const response = await axios.get(url, { timeout: 5000 });
      return parseInt(response.data.data[0].value);
    } catch {
      return null;
    }
  }

  private calculateVolatilityIndex(globalData: any): number {
    // Simplified volatility index calculation
    const btcChange = Math.abs(globalData.market_cap_change_percentage_24h_usd || 0);
    return Math.min(btcChange * 10, 100); // Scale to 0-100
  }

  private determineMarketTrend(globalData: any): 'bullish' | 'bearish' | 'neutral' {
    const change24h = globalData.market_cap_change_percentage_24h_usd || 0;
    
    if (change24h > 2) return 'bullish';
    if (change24h < -2) return 'bearish';
    return 'neutral';
  }

  private categorizeAsset(symbol: string): string {
    const categories: Record<string, string> = {
      'BTC': 'Store of Value',
      'ETH': 'Smart Contract Platform',
      'LINK': 'Oracle',
      'UNI': 'DEX',
      'AAVE': 'Lending',
      'COMP': 'Lending',
      'MKR': 'Stablecoin',
      'SNX': 'Derivatives',
      'CRV': 'DEX',
      'BAL': 'DEX',
      'YFI': 'Yield Farming',
      'USDC': 'Stablecoin',
      'USDT': 'Stablecoin',
      'MATIC': 'Layer 2',
      'AVAX': 'Smart Contract Platform'
    };

    return categories[symbol.toUpperCase()] || 'Other';
  }

  private isStablecoin(symbol: string): boolean {
    const stablecoins = ['USDC', 'USDT', 'BUSD', 'DAI', 'FRAX', 'LUSD'];
    return stablecoins.includes(symbol.toUpperCase());
  }

  private isNativeToken(symbol: string): boolean {
    const nativeTokens = ['ETH', 'BTC', 'MATIC', 'AVAX', 'BNB', 'SOL'];
    return nativeTokens.includes(symbol.toUpperCase());
  }

  private async getTechnicalIndicators(symbol: string): Promise<AssetMetadata['technicalIndicators']> {
    try {
      const priceData = await this.getHistoricalPrices(symbol, 200);
      
      if (priceData.length < 200) {
        throw new Error('Insufficient data for technical indicators');
      }

      const prices = priceData.map(p => parseFloat(ethers.utils.formatEther(p.price)));
      const latestPrice = prices[prices.length - 1];

      // Simple Moving Averages
      const sma20 = this.calculateSMA(prices, 20);
      const sma50 = this.calculateSMA(prices, 50);
      const sma200 = this.calculateSMA(prices, 200);

      // RSI calculation
      const rsi = this.calculateRSI(prices, 14);

      // MACD calculation
      const macd = this.calculateMACD(prices);

      // Bollinger Bands
      const bollingerBands = this.calculateBollingerBands(prices, 20, 2);

      return {
        rsi,
        macd,
        sma20: ethers.utils.parseEther(sma20.toString()),
        sma50: ethers.utils.parseEther(sma50.toString()),
        sma200: ethers.utils.parseEther(sma200.toString()),
        bollingerBands: {
          upper: ethers.utils.parseEther(bollingerBands.upper.toString()),
          middle: ethers.utils.parseEther(bollingerBands.middle.toString()),
          lower: ethers.utils.parseEther(bollingerBands.lower.toString())
        }
      };

    } catch (error) {
      // Return default values
      return {
        rsi: 50,
        macd: 0,
        sma20: ethers.utils.parseEther('1000'),
        sma50: ethers.utils.parseEther('1000'),
        sma200: ethers.utils.parseEther('1000'),
        bollingerBands: {
          upper: ethers.utils.parseEther('1100'),
          middle: ethers.utils.parseEther('1000'),
          lower: ethers.utils.parseEther('900')
        }
      };
    }
  }

  private calculateSMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1] || 0;
    
    const recentPrices = prices.slice(-period);
    return recentPrices.reduce((sum, price) => sum + price, 0) / period;
  }

  private calculateRSI(prices: number[], period: number = 14): number {
    if (prices.length < period + 1) return 50;

    const changes = [];
    for (let i = 1; i < prices.length; i++) {
      changes.push(prices[i] - prices[i - 1]);
    }

    const recentChanges = changes.slice(-period);
    const gains = recentChanges.filter(change => change > 0);
    const losses = recentChanges.filter(change => change < 0).map(loss => Math.abs(loss));

    const avgGain = gains.length > 0 ? gains.reduce((sum, gain) => sum + gain, 0) / period : 0;
    const avgLoss = losses.length > 0 ? losses.reduce((sum, loss) => sum + loss, 0) / period : 0;

    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(prices: number[]): number {
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    return ema12 - ema26;
  }

  private calculateEMA(prices: number[], period: number): number {
    if (prices.length < period) return prices[prices.length - 1] || 0;

    const multiplier = 2 / (period + 1);
    let ema = prices[0];

    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }

    return ema;
  }

  private calculateBollingerBands(prices: number[], period: number, stdDev: number): {
    upper: number;
    middle: number;
    lower: number;
  } {
    const sma = this.calculateSMA(prices, period);
    
    if (prices.length < period) {
      return { upper: sma * 1.1, middle: sma, lower: sma * 0.9 };
    }

    const recentPrices = prices.slice(-period);
    const variance = recentPrices.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / period;
    const standardDeviation = Math.sqrt(variance);

    return {
      upper: sma + (standardDeviation * stdDev),
      middle: sma,
      lower: sma - (standardDeviation * stdDev)
    };
  }

  private async getOrderbookLiquidity(symbol: string): Promise<{
    volume24h: BigNumber;
    spread: number;
    depth: BigNumber;
    slippageEstimates: Record<string, number>;
  }> {
    try {
      // In production, this would query DEX aggregators or orderbook APIs
      // For now, return estimated values based on market cap and volume
      const metadata = await this.getAssetMetadata(symbol);
      
      const volume24h = metadata.averageDailyVolume;
      const marketCap = parseFloat(ethers.utils.formatEther(metadata.marketCap));
      
      // Estimate spread based on market cap
      let spread = 0.5; // Default 0.5%
      if (marketCap > 10000000000) spread = 0.1; // >$10B = tight spread
      else if (marketCap > 1000000000) spread = 0.2; // >$1B
      else if (marketCap > 100000000) spread = 0.3; // >$100M

      // Estimate depth as 10x of 24h volume
      const depth = volume24h.mul(10);

      // Estimate slippage based on trade size vs daily volume
      const volumeNum = parseFloat(ethers.utils.formatEther(volume24h));
      
      return {
        volume24h,
        spread,
        depth,
        slippageEstimates: {
          '1000': Math.min(1000 / volumeNum * 100, 10), // Max 10%
          '10000': Math.min(10000 / volumeNum * 100, 20), // Max 20%
          '100000': Math.min(100000 / volumeNum * 100, 50) // Max 50%
        }
      };

    } catch (error) {
      return {
        volume24h: BigNumber.from(0),
        spread: 1.0,
        depth: BigNumber.from(0),
        slippageEstimates: {
          '1000': 0.5,
          '10000': 2.0,
          '100000': 10.0
        }
      };
    }
  }

  private calculateLiquidityScore(liquidityData: any): number {
    // Score from 0-100 based on various liquidity factors
    let score = 50; // Base score

    const volume24h = parseFloat(ethers.utils.formatEther(liquidityData.volume24h));
    
    // Volume score
    if (volume24h > 100000000) score += 30; // >$100M
    else if (volume24h > 10000000) score += 20; // >$10M
    else if (volume24h > 1000000) score += 10; // >$1M

    // Spread score
    if (liquidityData.spread < 0.1) score += 20;
    else if (liquidityData.spread < 0.5) score += 10;
    else if (liquidityData.spread > 2.0) score -= 20;

    return Math.min(Math.max(score, 0), 100);
  }

  private async calculateVolatilityRank(symbol: string, volatility: number): Promise<number> {
    try {
      // In production, this would compare against a database of all asset volatilities
      // For now, use simplified ranking
      if (volatility < 0.15) return 25; // Low volatility
      if (volatility < 0.30) return 50; // Medium volatility
      if (volatility < 0.60) return 75; // High volatility
      return 90; // Very high volatility
    } catch {
      return 50; // Default median rank
    }
  }

  // Testing methods
  private async testCoingeckoAPI(): Promise<void> {
    await axios.get('https://api.coingecko.com/api/v3/ping', { timeout: 5000 });
  }

  private async testChainlinkFeeds(): Promise<void> {
    const ethFeed = this.priceFeeds.get('ETH/USD_1');
    if (ethFeed) {
      await ethFeed.latestRoundData();
    }
  }

  private async testTheGraphAPI(): Promise<void> {
    // Test The Graph API if needed
  }

  // Cache management
  clearCache(): void {
    this.dataCache.clear();
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.dataCache.size,
      keys: Array.from(this.dataCache.keys())
    };
  }
}
