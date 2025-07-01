// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {FunctionsClient} from "@chainlink/contracts/src/v0.8/functions/dev/v1_0_0/FunctionsClient.sol";
import {FunctionsRequest} from "@chainlink/contracts/src/v0.8/functions/dev/v1_0_0/libraries/FunctionsRequest.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title FunctionsConsumer
 * @dev Chainlink Functions consumer for ArbOS external data and compute
 * @notice Executes serverless functions for ML predictions and external API calls
 */
contract FunctionsConsumer is FunctionsClient, Ownable {
    using FunctionsRequest for FunctionsRequest.Request;

    // Functions configuration
    uint64 private s_subscriptionId;
    uint32 private s_gasLimit = 300000;
    bytes32 private s_donId;
    
    // Request tracking
    mapping(bytes32 => address) public requestIdToRequester;
    mapping(bytes32 => RequestType) public requestTypes;
    mapping(bytes32 => bytes) public requestResults;
    mapping(address => uint256) public userRequestCounts;
    
    uint256 public totalRequests;
    address public arbOSCore;
    
    // Request types
    enum RequestType {
        PRICE_PREDICTION,
        MARKET_SENTIMENT,
        ARBITRAGE_ANALYSIS,
        PORTFOLIO_OPTIMIZATION,
        SECURITY_ANALYSIS
    }
    
    // Predefined JavaScript source codes for different operations
    mapping(RequestType => string) public functionSources;
    
    // Events
    event FunctionRequested(
        bytes32 indexed requestId,
        address indexed requester,
        RequestType requestType
    );
    
    event FunctionFulfilled(
        bytes32 indexed requestId,
        address indexed requester,
        bytes result
    );
    
    event FunctionError(
        bytes32 indexed requestId,
        bytes error
    );

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "FunctionsConsumer: Only core");
        _;
    }

    constructor(
        address router,
        uint64 subscriptionId,
        bytes32 donId
    ) FunctionsClient(router) {
        s_subscriptionId = subscriptionId;
        s_donId = donId;
        
        _initializeFunctionSources();
    }

    /**
     * @dev Initialize predefined function sources
     */
    function _initializeFunctionSources() internal {
        // Price prediction function
        functionSources[RequestType.PRICE_PREDICTION] = 
            "const symbol = args[0];"
            "const apiResponse = await Functions.makeHttpRequest({"
            "  url: `https://api.coindesk.com/v1/bpi/currentprice/${symbol}.json`"
            "});"
            "if (apiResponse.error) {"
            "  throw Error('Request failed');"
            "}"
            "const price = apiResponse.data.bpi[symbol].rate_float;"
            "return Functions.encodeUint256(Math.round(price * 1e8));";

        // Market sentiment function
        functionSources[RequestType.MARKET_SENTIMENT] = 
            "const symbol = args[0];"
            "const apiResponse = await Functions.makeHttpRequest({"
            "  url: `https://api.alternative.me/fng/?limit=1&format=json`"
            "});"
            "if (apiResponse.error) {"
            "  throw Error('Request failed');"
            "}"
            "const sentiment = apiResponse.data.data[0].value;"
            "return Functions.encodeUint256(sentiment);";

        // Arbitrage analysis function
        functionSources[RequestType.ARBITRAGE_ANALYSIS] = 
            "const tokenA = args[0];"
            "const tokenB = args[1];"
            "const exchangeA = args[2];"
            "const exchangeB = args[3];"
            "const priceA = await Functions.makeHttpRequest({"
            "  url: `https://api.${exchangeA}.com/api/v3/ticker/price?symbol=${tokenA}${tokenB}`"
            "});"
            "const priceB = await Functions.makeHttpRequest({"
            "  url: `https://api.${exchangeB}.com/api/v3/ticker/price?symbol=${tokenA}${tokenB}`"
            "});"
            "if (priceA.error || priceB.error) {"
            "  throw Error('Price fetch failed');"
            "}"
            "const priceDiff = Math.abs(parseFloat(priceA.data.price) - parseFloat(priceB.data.price));"
            "const profitPercentage = (priceDiff / parseFloat(priceA.data.price)) * 100;"
            "return Functions.encodeUint256(Math.round(profitPercentage * 1e4));";
    }

    /**
     * @dev Execute function for price prediction
     * @param symbol Token symbol
     * @param apiKey Optional API key
     * @return requestId Functions request ID
     */
    function requestPricePrediction(
        string calldata symbol,
        string calldata apiKey
    ) external returns (bytes32 requestId) {
        string[] memory args = new string[](1);
        args[0] = symbol;
        
        bytes memory secrets;
        if (bytes(apiKey).length > 0) {
            secrets = abi.encodePacked(apiKey);
        }
        
        return _sendRequest(
            functionSources[RequestType.PRICE_PREDICTION],
            args,
            secrets,
            RequestType.PRICE_PREDICTION
        );
    }

    /**
     * @dev Execute function for market sentiment analysis
     * @return requestId Functions request ID
     */
    function requestMarketSentiment() external returns (bytes32 requestId) {
        string[] memory args = new string[](0);
        bytes memory secrets;
        
        return _sendRequest(
            functionSources[RequestType.MARKET_SENTIMENT],
            args,
            secrets,
            RequestType.MARKET_SENTIMENT
        );
    }

    /**
     * @dev Execute function for arbitrage analysis
     * @param tokenA First token symbol
     * @param tokenB Second token symbol
     * @param exchangeA First exchange
     * @param exchangeB Second exchange
     * @return requestId Functions request ID
     */
    function requestArbitrageAnalysis(
        string calldata tokenA,
        string calldata tokenB,
        string calldata exchangeA,
        string calldata exchangeB
    ) external onlyCore returns (bytes32 requestId) {
        string[] memory args = new string[](4);
        args[0] = tokenA;
        args[1] = tokenB;
        args[2] = exchangeA;
        args[3] = exchangeB;
        
        bytes memory secrets;
        
        return _sendRequest(
            functionSources[RequestType.ARBITRAGE_ANALYSIS],
            args,
            secrets,
            RequestType.ARBITRAGE_ANALYSIS
        );
    }

    /**
     * @dev Execute custom function
     * @param source JavaScript source code
     * @param args Function arguments
     * @param secrets Encrypted secrets
     * @param requestType Type of request
     * @return requestId Functions request ID
     */
    function requestCustomFunction(
        string calldata source,
        string[] calldata args,
        bytes calldata secrets,
        RequestType requestType
    ) external onlyCore returns (bytes32 requestId) {
        return _sendRequest(source, args, secrets, requestType);
    }

    /**
     * @dev Internal function to send Functions request
     */
    function _sendRequest(
        string memory source,
        string[] memory args,
        bytes memory secrets,
        RequestType requestType
    ) internal returns (bytes32 requestId) {
        FunctionsRequest.Request memory req;
        req.initializeRequestForInlineJavaScript(source);
        
        if (args.length > 0) {
            req.setArgs(args);
        }
        
        if (secrets.length > 0) {
            req.addSecretsReference(secrets);
        }
        
        requestId = _sendRequest(
            req.encodeCBOR(),
            s_subscriptionId,
            s_gasLimit,
            s_donId
        );
        
        requestIdToRequester[requestId] = msg.sender;
        requestTypes[requestId] = requestType;
        userRequestCounts[msg.sender]++;
        totalRequests++;
        
        emit FunctionRequested(requestId, msg.sender, requestType);
        
        return requestId;
    }

    /**
     * @dev Callback function for Functions response
     * @param requestId Functions request ID
     * @param response Function response
     * @param err Error data
     */
    function fulfillRequest(
        bytes32 requestId,
        bytes memory response,
        bytes memory err
    ) internal override {
        address requester = requestIdToRequester[requestId];
        require(requester != address(0), "FunctionsConsumer: Invalid request");
        
        if (err.length > 0) {
            emit FunctionError(requestId, err);
            return;
        }
        
        requestResults[requestId] = response;
        
        // Process response based on request type
        RequestType requestType = requestTypes[requestId];
        _processResponse(requestId, response, requestType, requester);
        
        emit FunctionFulfilled(requestId, requester, response);
    }

    /**
     * @dev Process function response based on type
     */
    function _processResponse(
        bytes32 requestId,
        bytes memory response,
        RequestType requestType,
        address requester
    ) internal {
        if (requestType == RequestType.PRICE_PREDICTION) {
            _processPricePrediction(response, requester);
        } else if (requestType == RequestType.MARKET_SENTIMENT) {
            _processMarketSentiment(response, requester);
        } else if (requestType == RequestType.ARBITRAGE_ANALYSIS) {
            _processArbitrageAnalysis(response, requester);
        }
    }

    /**
     * @dev Process price prediction response
     */
    function _processPricePrediction(bytes memory response, address requester) internal {
        if (arbOSCore != address(0) && response.length >= 32) {
            uint256 predictedPrice = abi.decode(response, (uint256));
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processPricePrediction(address,uint256)",
                    requester,
                    predictedPrice
                )
            );
            // Continue even if call fails
        }
    }

    /**
     * @dev Process market sentiment response
     */
    function _processMarketSentiment(bytes memory response, address requester) internal {
        if (arbOSCore != address(0) && response.length >= 32) {
            uint256 sentiment = abi.decode(response, (uint256));
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processMarketSentiment(address,uint256)",
                    requester,
                    sentiment
                )
            );
            // Continue even if call fails
        }
    }

    /**
     * @dev Process arbitrage analysis response
     */
    function _processArbitrageAnalysis(bytes memory response, address requester) internal {
        if (arbOSCore != address(0) && response.length >= 32) {
            uint256 profitPercentage = abi.decode(response, (uint256));
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processArbitrageAnalysis(address,uint256)",
                    requester,
                    profitPercentage
                )
            );
            // Continue even if call fails
        }
    }

    /**
     * @dev Get function result
     * @param requestId Functions request ID
     * @return result Function result
     */
    function getResult(bytes32 requestId) external view returns (bytes memory) {
        return requestResults[requestId];
    }

    /**
     * @dev Update function source for a request type
     * @param requestType Type of request
     * @param source New JavaScript source code
     */
    function updateFunctionSource(
        RequestType requestType,
        string calldata source
    ) external onlyOwner {
        functionSources[requestType] = source;
    }

    /**
     * @dev Update Functions configuration
     * @param subscriptionId New subscription ID
     * @param gasLimit New gas limit
     * @param donId New DON ID
     */
    function updateFunctionsConfig(
        uint64 subscriptionId,
        uint32 gasLimit,
        bytes32 donId
    ) external onlyOwner {
        s_subscriptionId = subscriptionId;
        s_gasLimit = gasLimit;
        s_donId = donId;
    }

    /**
     * @dev Update ArbOS core address
     */
    function updateArbOSCore(address _arbOSCore) external onlyOwner {
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Get Functions configuration
     */
    function getFunctionsConfig() external view returns (
        uint64 subscriptionId,
        uint32 gasLimit,
        bytes32 donId
    ) {
        return (s_subscriptionId, s_gasLimit, s_donId);
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
