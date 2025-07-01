import { ethers } from 'ethers';

// Official Chainlink contract addresses from docs.chain.link
const CHAINLINK_CONTRACTS = {
  // Price Feed Aggregators
  PRICE_FEEDS: {
    1: { // Ethereum
      'ETH/USD': '0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419',
      'BTC/USD': '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88c',
      'LINK/USD': '0x2c1d072e956AFFC0D435Cb7AC38EF18d24d9127c',
      'USDC/USD': '0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6'
    },
    43114: { // Avalanche
      'AVAX/USD': '0x0A77230d17318075983913bC2145DB16C7366156',
      'ETH/USD': '0x976B3D034E162d8bD72D6b9C989d545b839003b0',
      'BTC/USD': '0x2779D32d5166BAaa2B2b658333bA7e6Ec0C65743',
      'LINK/USD': '0x49ccd9ca821EfEab2b98c60dC60F518E765EDe9a'
    },
    137: { // Polygon
      'MATIC/USD': '0xAB594600376Ec9fD91F8e885dADF0CE036862dE0',
      'ETH/USD': '0xF9680D99D6C9589e2a93a78A04A279e509205945',
      'BTC/USD': '0xc907E116054Ad103354f2D350FD2514433D57F6f',
      'LINK/USD': '0xd9FFdb71EbE7496cC440152d43986Aae0AB76665'
    },
    42161: { // Arbitrum
      'ETH/USD': '0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612',
      'BTC/USD': '0x6ce185860a4963106506C203335A2910413708e9',
      'LINK/USD': '0x86E53CF1B870786351Da77A57575e79CB55812CB',
      'USDC/USD': '0x50834F3163758fcC1Df9973b6e91f0F0F0434aD3'
    }
  },

  // VRF Coordinators
  VRF_COORDINATORS: {
    1: '0x271682DEB8C4E0901D1a1550aD2e64D568E69909',
    43114: '0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634',
    137: '0xAE975071Be8F8eE67addBC1A82488F1C24858067',
    42161: '0x41034678D6C633D8a95c75e1138A360a28bA15d1'
  },

  // Functions Routers
  FUNCTIONS_ROUTERS: {
    1: '0x65C939B26d3d949A6E2bE41B1F5659dB13b5f4a',
    43114: '0x3c82F31d0b6e267eF9b5b5e5d2B9A2C8a2F4c5e7',
    137: '0x4f8a84C442F9675610c680990EdDb2CCDDB2E906',
    42161: '0x97083E831F8F0638855e2A515c90EdCF158DF238'
  },

  // CCIP Routers
  CCIP_ROUTERS: {
    1: '0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D',
    43114: '0xF4c7E640EdA248ef95972845a62bdC74237805dB',
    137: '0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43',
    42161: '0x141fa059441E0ca23ce184B6A78bafD2A517DdE8'
  },

  // Automation Registries
  AUTOMATION_REGISTRIES: {
    1: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
    43114: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
    137: '0x02777053d6764996e594c3E88AF1D58D5363a2e6',
    42161: '0x75c0530885F385721fddA23C539AF3701d6183D4'
  }
};

// ABI fragments for Chainlink contracts
const PRICE_FEED_ABI = [
  'function latestRoundData() external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound)',
  'function decimals() external view returns (uint8)',
  'function description() external view returns (string)'
];

const VRF_COORDINATOR_ABI = [
  'function requestRandomWords(bytes32 keyHash, uint64 subId, uint16 minimumRequestConfirmations, uint32 callbackGasLimit, uint32 numWords) external returns (uint256 requestId)'
];

const CCIP_ROUTER_ABI = [
  'function ccipSend(uint64 destinationChainSelector, (bytes receiver, bytes data, (address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external payable returns (bytes32)',
  'function getFee(uint64 destinationChainSelector, (bytes receiver, bytes data, (address token, uint256 amount)[] tokenAmounts, address feeToken, bytes extraArgs) message) external view returns (uint256 fee)'
];

export class ChainlinkService {
  /**
   * Get latest price from Chainlink Data Feed
   */
  async getLatestPrice(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    pair: string
  ): Promise<number> {
    const priceFeedAddress = this.getPriceFeedAddress(chainId, pair);
    if (!priceFeedAddress) {
      throw new Error(`Price feed not available for ${pair} on chain ${chainId}`);
    }

    const priceFeed = new ethers.Contract(priceFeedAddress, PRICE_FEED_ABI, signer);
    
    const [roundId, answer, startedAt, updatedAt, answeredInRound] = await priceFeed.latestRoundData();
    const decimals = await priceFeed.decimals();
    
    const price = Number(answer) / Math.pow(10, decimals);
    return price;
  }

  /**
   * Request randomness using Chainlink VRF
   */
  async requestRandomness(
    signer: ethers.JsonRpcSigner,
    chainId: number,
    subscriptionId: number = 1,
    numWords: number = 1
  ): Promise<string> {
    const vrfCoordinator = this.getVRFCoordinator(chainId);
    if (!vrfCoordinator) {
      throw new Error(`VRF Coordinator not available on chain ${chainId}`);
    }

    const contract = new ethers.Contract(vrfCoordinator, VRF_COORDINATOR_ABI, signer);
    
    // Chain-specific VRF key hashes (from Chainlink docs)
    const keyHashes: { [chainId: number]: string } = {
      1: '0x8af398995b04c28e9951adb9721ef74c74f93e6a478f39e7e0777be13527e7ef',
      43114: '0x354d2f95da55398f44b7cff77da56283d9c6c829a4bdf1bbcaf2ad6a4d081f61',
      137: '0xcc294a196eeeb44da2888d17c0625cc88d70d9760a69d58d853ba6581a9ab0cd',
      42161: '0x08ba8f62ff6c40a58877a106147661db43bc58dabfb814793847a839aa03367f'
    };

    const keyHash = keyHashes[chainId];
    if (!keyHash) {
      throw new Error(`VRF key hash not configured for chain ${chainId}`);
    }

    const tx = await contract.requestRandomWords(
      keyHash,
      subscriptionId,
      3, // minimum confirmations
      100000, // callback gas limit
      numWords
    );

    const receipt = await tx.wait();
    return receipt.hash;
  }

  /**
   * Send cross-chain message using CCIP
   */
  async sendCCIPMessage(
    signer: ethers.JsonRpcSigner,
    sourceChainId: number,
    destinationChainId: number,
    receiver: string,
    data: string,
    tokenAmounts: Array<{ token: string; amount: bigint }> = []
  ): Promise<string> {
    const ccipRouter = this.getCCIPRouter(sourceChainId);
    if (!ccipRouter) {
      throw new Error(`CCIP Router not available on chain ${sourceChainId}`);
    }

    const destinationSelector = this.getChainSelector(destinationChainId);
    if (!destinationSelector) {
      throw new Error(`Chain selector not found for destination chain ${destinationChainId}`);
    }

    const contract = new ethers.Contract(ccipRouter, CCIP_ROUTER_ABI, signer);
    
    const message = {
      receiver: ethers.getBytes(receiver),
      data: ethers.getBytes(data),
      tokenAmounts: tokenAmounts.map(ta => ({
        token: ta.token,
        amount: ta.amount
      })),
      feeToken: ethers.ZeroAddress, // Pay in native token
      extraArgs: '0x' // Default extra args
    };

    // Get fee estimate
    const fee = await contract.getFee(destinationSelector, message);
    
    // Send message
    const tx = await contract.ccipSend(destinationSelector, message, { value: fee });
    const receipt = await tx.wait();
    
    return receipt.hash;
  }

  /**
   * Get supported price feeds for a chain
   */
  getSupportedPriceFeeds(chainId: number): string[] {
    const feeds = CHAINLINK_CONTRACTS.PRICE_FEEDS[chainId as keyof typeof CHAINLINK_CONTRACTS.PRICE_FEEDS];
    return feeds ? Object.keys(feeds) : [];
  }

  /**
   * Check if VRF is available on chain
   */
  isVRFAvailable(chainId: number): boolean {
    return chainId in CHAINLINK_CONTRACTS.VRF_COORDINATORS;
  }

  /**
   * Check if CCIP is available on chain
   */
  isCCIPAvailable(chainId: number): boolean {
    return chainId in CHAINLINK_CONTRACTS.CCIP_ROUTERS;
  }

  /**
   * Get price feed address for chain and pair
   */
  private getPriceFeedAddress(chainId: number, pair: string): string | undefined {
    const feeds = CHAINLINK_CONTRACTS.PRICE_FEEDS[chainId as keyof typeof CHAINLINK_CONTRACTS.PRICE_FEEDS];
    return feeds?.[pair as keyof typeof feeds];
  }

  /**
   * Get VRF Coordinator address for chain
   */
  private getVRFCoordinator(chainId: number): string | undefined {
    return CHAINLINK_CONTRACTS.VRF_COORDINATORS[chainId as keyof typeof CHAINLINK_CONTRACTS.VRF_COORDINATORS];
  }

  /**
   * Get CCIP Router address for chain
   */
  private getCCIPRouter(chainId: number): string | undefined {
    return CHAINLINK_CONTRACTS.CCIP_ROUTERS[chainId as keyof typeof CHAINLINK_CONTRACTS.CCIP_ROUTERS];
  }

  /**
   * Get CCIP chain selector
   */
  private getChainSelector(chainId: number): bigint | undefined {
    const selectors: { [chainId: number]: bigint } = {
      1: 5009297550715157269n,        // Ethereum
      43114: 6433500567565415381n,    // Avalanche
      137: 4051577828743386545n,      // Polygon
      42161: 4949039107694359620n     // Arbitrum
    };
    return selectors[chainId];
  }
}

export const chainlinkService = new ChainlinkService();
