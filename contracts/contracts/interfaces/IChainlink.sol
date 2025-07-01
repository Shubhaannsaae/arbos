// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {AggregatorV3Interface} from "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title IChainlink
 * @dev Interface for Chainlink service integrations
 */
interface IChainlink {
    struct PriceFeedData {
        AggregatorV3Interface priceFeed;
        uint8 decimals;
        uint256 heartbeat;
        bool active;
    }

    struct AutomationConfig {
        uint256 upkeepId;
        uint256 interval;
        uint256 lastPerformed;
        bool active;
    }

    struct VRFConfig {
        uint64 subscriptionId;
        bytes32 keyHash;
        uint32 callbackGasLimit;
        uint16 requestConfirmations;
        uint32 numWords;
    }

    event PriceFeedUpdated(
        address indexed token,
        address indexed priceFeed,
        int256 price
    );

    event AutomationPerformed(
        uint256 indexed upkeepId,
        uint256 timestamp
    );

    event VRFRequested(
        uint256 indexed requestId,
        address indexed requester
    );

    event CCIPMessageSent(
        bytes32 indexed messageId,
        uint64 indexed destinationChain,
        address receiver
    );

    function addPriceFeed(
        address token,
        address priceFeed,
        uint256 heartbeat
    ) external;

    function getLatestPrice(address token) external view returns (int256, uint256);
    
    function requestRandomness() external returns (uint256 requestId);
    
    function sendCrossChainMessage(
        uint64 destinationChain,
        address receiver,
        bytes calldata data
    ) external returns (bytes32 messageId);
    
    function registerUpkeep(
        string calldata name,
        uint256 interval
    ) external returns (uint256 upkeepId);
}
