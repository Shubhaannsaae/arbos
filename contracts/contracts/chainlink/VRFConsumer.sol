// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {VRFCoordinatorV2Interface} from "@chainlink/contracts/src/v0.8/interfaces/VRFCoordinatorV2Interface.sol";
import {VRFConsumerBaseV2} from "@chainlink/contracts/src/v0.8/vrf/VRFConsumerBaseV2.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title VRFConsumer
 * @dev Chainlink VRF consumer for ArbOS random number generation
 * @notice Provides verifiable randomness for portfolio strategies and security
 */
contract VRFConsumer is VRFConsumerBaseV2, Ownable {
    
    // VRF Coordinator
    VRFCoordinatorV2Interface private immutable i_vrfCoordinator;
    
    // VRF Configuration
    uint64 private s_subscriptionId;
    bytes32 private s_keyHash;
    uint32 private s_callbackGasLimit = 100000;
    uint16 private s_requestConfirmations = 3;
    uint32 private s_numWords = 1;
    
    // State variables
    mapping(uint256 => address) public requestIdToRequester;
    mapping(uint256 => uint256[]) public requestIdToRandomWords;
    mapping(address => uint256[]) public userRandomNumbers;
    mapping(address => uint256) public userRequestCounts;
    
    uint256 public totalRequests;
    address public arbOSCore;
    
    // Request types
    enum RequestType {
        PORTFOLIO_REBALANCING,
        SECURITY_AUDIT,
        ARBITRAGE_SELECTION,
        GENERAL
    }
    
    mapping(uint256 => RequestType) public requestTypes;
    
    // Events
    event RandomnessRequested(
        uint256 indexed requestId,
        address indexed requester,
        RequestType requestType
    );
    
    event RandomnessFulfilled(
        uint256 indexed requestId,
        address indexed requester,
        uint256[] randomWords
    );

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "VRFConsumer: Only core");
        _;
    }

    constructor(
        address vrfCoordinator,
        uint64 subscriptionId,
        bytes32 keyHash
    ) VRFConsumerBaseV2(vrfCoordinator) {
        i_vrfCoordinator = VRFCoordinatorV2Interface(vrfCoordinator);
        s_subscriptionId = subscriptionId;
        s_keyHash = keyHash;
    }

    /**
     * @dev Request random words from Chainlink VRF
     * @param numWords Number of random words to request
     * @param requestType Type of randomness request
     * @return requestId VRF request ID
     */
    function requestRandomWords(
        uint32 numWords,
        RequestType requestType
    ) external returns (uint256 requestId) {
        requestId = i_vrfCoordinator.requestRandomWords(
            s_keyHash,
            s_subscriptionId,
            s_requestConfirmations,
            s_callbackGasLimit,
            numWords
        );
        
        requestIdToRequester[requestId] = msg.sender;
        requestTypes[requestId] = requestType;
        userRequestCounts[msg.sender]++;
        totalRequests++;
        
        emit RandomnessRequested(requestId, msg.sender, requestType);
        
        return requestId;
    }

    /**
     * @dev Request randomness for portfolio rebalancing
     * @return requestId VRF request ID
     */
    function requestPortfolioRandomness() external onlyCore returns (uint256 requestId) {
        return requestRandomWords(s_numWords, RequestType.PORTFOLIO_REBALANCING);
    }

    /**
     * @dev Request randomness for security audits
     * @return requestId VRF request ID
     */
    function requestSecurityRandomness() external onlyCore returns (uint256 requestId) {
        return requestRandomWords(2, RequestType.SECURITY_AUDIT);
    }

    /**
     * @dev Request randomness for arbitrage opportunity selection
     * @return requestId VRF request ID
     */
    function requestArbitrageRandomness() external onlyCore returns (uint256 requestId) {
        return requestRandomWords(1, RequestType.ARBITRAGE_SELECTION);
    }

    /**
     * @dev Callback function used by VRF Coordinator
     * @param requestId VRF request ID
     * @param randomWords Array of random words
     */
    function fulfillRandomWords(
        uint256 requestId,
        uint256[] memory randomWords
    ) internal override {
        address requester = requestIdToRequester[requestId];
        require(requester != address(0), "VRFConsumer: Invalid request");
        
        requestIdToRandomWords[requestId] = randomWords;
        
        // Store random numbers for the user
        for (uint256 i = 0; i < randomWords.length; i++) {
            userRandomNumbers[requester].push(randomWords[i]);
        }
        
        // Process based on request type
        RequestType requestType = requestTypes[requestId];
        _processRandomnessRequest(requestId, randomWords, requestType, requester);
        
        emit RandomnessFulfilled(requestId, requester, randomWords);
    }

    /**
     * @dev Process randomness request based on type
     */
    function _processRandomnessRequest(
        uint256 requestId,
        uint256[] memory randomWords,
        RequestType requestType,
        address requester
    ) internal {
        if (requestType == RequestType.PORTFOLIO_REBALANCING) {
            _processPortfolioRandomness(randomWords, requester);
        } else if (requestType == RequestType.SECURITY_AUDIT) {
            _processSecurityRandomness(randomWords, requester);
        } else if (requestType == RequestType.ARBITRAGE_SELECTION) {
            _processArbitrageRandomness(randomWords, requester);
        }
    }

    /**
     * @dev Process portfolio rebalancing randomness
     */
    function _processPortfolioRandomness(uint256[] memory randomWords, address requester) internal {
        if (arbOSCore != address(0)) {
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processPortfolioRandomness(address,uint256)",
                    requester,
                    randomWords[0]
                )
            );
            // Continue even if call fails
        }
    }

    /**
     * @dev Process security audit randomness
     */
    function _processSecurityRandomness(uint256[] memory randomWords, address requester) internal {
        if (arbOSCore != address(0) && randomWords.length >= 2) {
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processSecurityRandomness(address,uint256,uint256)",
                    requester,
                    randomWords[0],
                    randomWords[1]
                )
            );
            // Continue even if call fails
        }
    }

    /**
     * @dev Process arbitrage selection randomness
     */
    function _processArbitrageRandomness(uint256[] memory randomWords, address requester) internal {
        if (arbOSCore != address(0)) {
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processArbitrageRandomness(address,uint256)",
                    requester,
                    randomWords[0]
                )
            );
            // Continue even if call fails
        }
    }

    /**
     * @dev Get random words for a request
     * @param requestId VRF request ID
     * @return randomWords Array of random words
     */
    function getRandomWords(uint256 requestId) external view returns (uint256[] memory) {
        return requestIdToRandomWords[requestId];
    }

    /**
     * @dev Get user's random numbers
     * @param user User address
     * @return randomNumbers User's random numbers
     */
    function getUserRandomNumbers(address user) external view returns (uint256[] memory) {
        return userRandomNumbers[user];
    }

    /**
     * @dev Get latest random number for user
     * @param user User address
     * @return randomNumber Latest random number
     */
    function getLatestRandomNumber(address user) external view returns (uint256) {
        uint256[] memory numbers = userRandomNumbers[user];
        require(numbers.length > 0, "VRFConsumer: No random numbers");
        return numbers[numbers.length - 1];
    }

    /**
     * @dev Generate normalized random number between 0 and max
     * @param randomWord Raw random word
     * @param max Maximum value (exclusive)
     * @return normalizedRandom Normalized random number
     */
    function normalizeRandomNumber(uint256 randomWord, uint256 max) 
        external 
        pure 
        returns (uint256) 
    {
        require(max > 0, "VRFConsumer: Max must be greater than 0");
        return randomWord % max;
    }

    /**
     * @dev Generate random boolean
     * @param randomWord Raw random word
     * @return randomBool Random boolean
     */
    function getRandomBoolean(uint256 randomWord) external pure returns (bool) {
        return (randomWord % 2) == 1;
    }

    /**
     * @dev Generate random percentage (0-100)
     * @param randomWord Raw random word
     * @return percentage Random percentage
     */
    function getRandomPercentage(uint256 randomWord) external pure returns (uint256) {
        return (randomWord % 101); // 0-100 inclusive
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
        s_subscriptionId = subscriptionId;
        s_keyHash = keyHash;
        s_callbackGasLimit = callbackGasLimit;
        s_requestConfirmations = requestConfirmations;
        s_numWords = numWords;
    }

    /**
     * @dev Update ArbOS core address
     */
    function updateArbOSCore(address _arbOSCore) external onlyOwner {
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Get VRF configuration
     */
    function getVRFConfig() external view returns (
        uint64 subscriptionId,
        bytes32 keyHash,
        uint32 callbackGasLimit,
        uint16 requestConfirmations,
        uint32 numWords
    ) {
        return (
            s_subscriptionId,
            s_keyHash,
            s_callbackGasLimit,
            s_requestConfirmations,
            s_numWords
        );
    }

    /**
     * @dev Get request statistics
     */
    function getRequestStats() external view returns (
        uint256 totalRequestCount,
        uint256 userRequestCount
    ) {
        return (totalRequests, userRequestCounts[msg.sender]);
    }
}
