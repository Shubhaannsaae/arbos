// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {AggregatorV3Interface} from "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {IChainlink} from "../interfaces/IChainlink.sol";

/**
 * @title ChainlinkConsumer
 * @dev Main Chainlink services consumer for ArbOS ecosystem
 * @notice Integrates Data Feeds, CCIP, Automation, VRF, and Functions
 */
contract ChainlinkConsumer is IChainlink, Ownable {
    
    // Price feed mappings
    mapping(address => PriceFeedData) public priceFeeds;
    mapping(address => bool) public supportedTokens;
    
    // Automation configuration
    mapping(uint256 => AutomationConfig) public automationConfigs;
    
    // VRF configuration
    VRFConfig public vrfConfig;
    
    // Service contract addresses
    address public ccipReceiver;
    address public automationConsumer;
    address public vrfConsumer;
    address public functionsConsumer;
    
    // Constants
    uint256 public constant PRICE_FRESHNESS_THRESHOLD = 3600; // 1 hour
    
    // Events from interface
    
    modifier validPriceFeed(address token) {
        require(priceFeeds[token].active, "ChainlinkConsumer: Price feed not active");
        _;
    }

    constructor() {}

    /**
     * @dev Add price feed for token
     * @param token Token address
     * @param priceFeed Chainlink price feed address
     * @param heartbeat Maximum time between price updates
     */
    function addPriceFeed(
        address token,
        address priceFeed,
        uint256 heartbeat
    ) external override onlyOwner {
        require(token != address(0), "ChainlinkConsumer: Invalid token");
        require(priceFeed != address(0), "ChainlinkConsumer: Invalid price feed");
        
        AggregatorV3Interface feed = AggregatorV3Interface(priceFeed);
        uint8 decimals = feed.decimals();
        
        priceFeeds[token] = PriceFeedData({
            priceFeed: feed,
            decimals: decimals,
            heartbeat: heartbeat,
            active: true
        });
        
        supportedTokens[token] = true;
        
        emit PriceFeedUpdated(token, priceFeed, 0);
    }

    /**
     * @dev Get latest price for token
     * @param token Token address
     * @return price Latest price
     * @return timestamp Price timestamp
     */
    function getLatestPrice(address token) 
        external 
        view 
        override 
        validPriceFeed(token) 
        returns (int256 price, uint256 timestamp) 
    {
        PriceFeedData memory feedData = priceFeeds[token];
        
        try feedData.priceFeed.latestRoundData() returns (
            uint80 /* roundId */,
            int256 answer,
            uint256 /* startedAt */,
            uint256 updatedAt,
            uint80 /* answeredInRound */
        ) {
            require(answer > 0, "ChainlinkConsumer: Invalid price");
            require(
                block.timestamp - updatedAt <= PRICE_FRESHNESS_THRESHOLD,
                "ChainlinkConsumer: Price too stale"
            );
            
            return (answer, updatedAt);
        } catch {
            revert("ChainlinkConsumer: Price feed error");
        }
    }

    /**
     * @dev Request randomness from VRF
     * @return requestId VRF request ID
     */
    function requestRandomness() external override returns (uint256 requestId) {
        require(vrfConsumer != address(0), "ChainlinkConsumer: VRF not configured");
        
        // Delegate to VRF consumer
        (bool success, bytes memory data) = vrfConsumer.call(
            abi.encodeWithSignature("requestRandomWords()")
        );
        
        require(success, "ChainlinkConsumer: VRF request failed");
        requestId = abi.decode(data, (uint256));
        
        emit VRFRequested(requestId, msg.sender);
        return requestId;
    }

    /**
     * @dev Send cross-chain message via CCIP
     * @param destinationChain Destination chain selector
     * @param receiver Receiver contract address
     * @param data Message data
     * @return messageId CCIP message ID
     */
    function sendCrossChainMessage(
        uint64 destinationChain,
        address receiver,
        bytes calldata data
    ) external override returns (bytes32 messageId) {
        require(ccipReceiver != address(0), "ChainlinkConsumer: CCIP not configured");
        
        // Delegate to CCIP receiver
        (bool success, bytes memory returnData) = ccipReceiver.call(
            abi.encodeWithSignature(
                "sendMessage(uint64,address,bytes)",
                destinationChain,
                receiver,
                data
            )
        );
        
        require(success, "ChainlinkConsumer: CCIP send failed");
        messageId = abi.decode(returnData, (bytes32));
        
        emit CCIPMessageSent(messageId, destinationChain, receiver);
        return messageId;
    }

    /**
     * @dev Register automation upkeep
     * @param name Upkeep name
     * @param interval Execution interval
     * @return upkeepId Automation upkeep ID
     */
    function registerUpkeep(
        string calldata name,
        uint256 interval
    ) external override returns (uint256 upkeepId) {
        require(automationConsumer != address(0), "ChainlinkConsumer: Automation not configured");
        
        // Generate unique upkeep ID
        upkeepId = uint256(keccak256(abi.encodePacked(name, block.timestamp)));
        
        automationConfigs[upkeepId] = AutomationConfig({
            upkeepId: upkeepId,
            interval: interval,
            lastPerformed: block.timestamp,
            active: true
        });
        
        return upkeepId;
    }

    /**
     * @dev Get price feed decimals
     * @param token Token address
     * @return decimals Price feed decimals
     */
    function getPriceFeedDecimals(address token) 
        external 
        view 
        validPriceFeed(token) 
        returns (uint8 decimals) 
    {
        return priceFeeds[token].decimals;
    }

    /**
     * @dev Check if price feed is stale
     * @param token Token address
     * @return isStale True if price is stale
     */
    function isPriceFeedStale(address token) 
        external 
        view 
        validPriceFeed(token) 
        returns (bool isStale) 
    {
        PriceFeedData memory feedData = priceFeeds[token];
        
        try feedData.priceFeed.latestRoundData() returns (
            uint80 /* roundId */,
            int256 /* answer */,
            uint256 /* startedAt */,
            uint256 updatedAt,
            uint80 /* answeredInRound */
        ) {
            return block.timestamp - updatedAt > feedData.heartbeat;
        } catch {
            return true;
        }
    }

    /**
     * @dev Update service contract addresses
     */
    function updateServiceContracts(
        address _ccipReceiver,
        address _automationConsumer,
        address _vrfConsumer,
        address _functionsConsumer
    ) external onlyOwner {
        ccipReceiver = _ccipReceiver;
        automationConsumer = _automationConsumer;
        vrfConsumer = _vrfConsumer;
        functionsConsumer = _functionsConsumer;
    }

    /**
     * @dev Update VRF configuration
     */
    function updateVRFConfig(
        uint64 subscriptionId,
        bytes32 keyHash,
        uint32 callbackGasLimit,
        uint16 requestConfirmations,
        uint32 numWords
    ) external onlyOwner {
        vrfConfig = VRFConfig({
            subscriptionId: subscriptionId,
            keyHash: keyHash,
            callbackGasLimit: callbackGasLimit,
            requestConfirmations: requestConfirmations,
            numWords: numWords
        });
    }

    /**
     * @dev Disable price feed
     */
    function disablePriceFeed(address token) external onlyOwner {
        priceFeeds[token].active = false;
        supportedTokens[token] = false;
    }

    /**
     * @dev Get supported tokens list
     */
    function getSupportedTokens() external view returns (address[] memory tokens) {
        // use EnumerableSet for efficiency
        // This is a simplified implementation
        uint256 count = 0;
        
        tokens = new address[](count);
        return tokens;
    }
}
