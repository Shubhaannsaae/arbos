// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

import {IArbitrage} from "../interfaces/IArbitrage.sol";
import {IChainlink} from "../interfaces/IChainlink.sol";

/**
 * @title ArbitrageEngine
 * @dev Core arbitrage execution engine with Chainlink price feed integration
 */
contract ArbitrageEngine is IArbitrage, Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using EnumerableSet for EnumerableSet.UintSet;

    // Constants
    uint256 public constant MAX_SLIPPAGE = 500; // 5%
    uint256 public constant MIN_PROFIT_THRESHOLD = 10; // 0.1%
    uint256 public constant MAX_OPPORTUNITIES = 100;
    
    // State variables
    mapping(uint256 => ArbitrageOpportunity) public opportunities;
    mapping(address => PriceData) public latestPrices;
    mapping(address => bool) public supportedTokens;
    mapping(address => bool) public supportedExchanges;
    
    EnumerableSet.UintSet private activeOpportunities;
    uint256 public nextOpportunityId = 1;
    uint256 public executedOpportunities;
    uint256 public totalProfit;
    
    address public chainlinkConsumer;
    address public arbOSCore;
    
    // Events from interface already declared

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "ArbitrageEngine: Only core");
        _;
    }

    modifier supportedToken(address token) {
        require(supportedTokens[token], "ArbitrageEngine: Token not supported");
        _;
    }

    modifier supportedExchange(address exchange) {
        require(supportedExchanges[exchange], "ArbitrageEngine: Exchange not supported");
        _;
    }

    constructor(address _chainlinkConsumer, address _arbOSCore) {
        chainlinkConsumer = _chainlinkConsumer;
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Detect arbitrage opportunity between two exchanges
     */
    function detectOpportunity(
        address tokenA,
        address tokenB,
        address exchangeA,
        address exchangeB,
        uint256 amount
    ) 
        external 
        view 
        override 
        supportedToken(tokenA)
        supportedToken(tokenB)
        supportedExchange(exchangeA)
        supportedExchange(exchangeB)
        returns (bool profitable, uint256 expectedProfit) 
    {
        // Get latest prices from Chainlink
        PriceData memory priceA = latestPrices[tokenA];
        PriceData memory priceB = latestPrices[tokenB];
        
        require(priceA.price > 0 && priceB.price > 0, "ArbitrageEngine: Invalid prices");
        
        // Calculate exchange rates on both exchanges
        uint256 rateExchangeA = _getExchangeRate(tokenA, tokenB, exchangeA, amount);
        uint256 rateExchangeB = _getExchangeRate(tokenA, tokenB, exchangeB, amount);
        
        // Calculate potential profit
        int256 profit = calculateProfit(tokenA, tokenB, amount, exchangeA, exchangeB);
        
        if (profit > 0) {
            uint256 profitBps = (uint256(profit) * 10000) / amount;
            profitable = profitBps >= MIN_PROFIT_THRESHOLD;
            expectedProfit = profitable ? uint256(profit) : 0;
        }
        
        return (profitable, expectedProfit);
    }

    /**
     * @dev Execute arbitrage opportunity
     */
    function executeArbitrage(uint256 opportunityId) 
        external 
        override 
        onlyCore 
        nonReentrant 
        returns (bool success) 
    {
        require(
            activeOpportunities.contains(opportunityId),
            "ArbitrageEngine: Opportunity not found or executed"
        );
        
        ArbitrageOpportunity storage opportunity = opportunities[opportunityId];
        require(!opportunity.executed, "ArbitrageEngine: Already executed");
        require(
            block.timestamp <= opportunity.timestamp + 300,
            "ArbitrageEngine: Opportunity expired"
        );

        // Re-check profitability
        (bool profitable, uint256 currentProfit) = detectOpportunity(
            opportunity.tokenA,
            opportunity.tokenB,
            opportunity.exchangeA,
            opportunity.exchangeB,
            opportunity.amountIn
        );
        
        require(profitable, "ArbitrageEngine: No longer profitable");
        
        // Execute the arbitrage
        success = _executeArbitrageInternal(opportunity, currentProfit);
        
        if (success) {
            opportunity.executed = true;
            activeOpportunities.remove(opportunityId);
            executedOpportunities++;
            totalProfit += currentProfit;
            
            emit ArbitrageExecuted(
                opportunityId,
                opportunity.tokenA,
                opportunity.tokenB,
                currentProfit,
                block.timestamp
            );
        }
        
        return success;
    }

    /**
     * @dev Get latest price for token from Chainlink
     */
    function getLatestPrice(address token) 
        external 
        view 
        override 
        returns (PriceData memory) 
    {
        return latestPrices[token];
    }

    /**
     * @dev Calculate potential profit from arbitrage
     */
    function calculateProfit(
        address tokenA,
        address tokenB,
        uint256 amountIn,
        address exchangeA,
        address exchangeB
    ) public view override returns (int256 profit) {
        // Get prices on both exchanges
        uint256 priceOnA = _getExchangeRate(tokenA, tokenB, exchangeA, amountIn);
        uint256 priceOnB = _getExchangeRate(tokenB, tokenA, exchangeB, priceOnA);
        
        // Calculate profit (accounting for gas and fees)
        int256 grossProfit = int256(priceOnB) - int256(amountIn);
        uint256 estimatedCosts = _calculateTransactionCosts(amountIn);
        
        profit = grossProfit - int256(estimatedCosts);
        
        return profit;
    }

    /**
     * @dev Create new arbitrage opportunity
     */
    function createOpportunity(
        address tokenA,
        address tokenB,
        address exchangeA,
        address exchangeB,
        uint256 amountIn
    ) external onlyOwner returns (uint256 opportunityId) {
        require(
            activeOpportunities.length() < MAX_OPPORTUNITIES,
            "ArbitrageEngine: Too many active opportunities"
        );
        
        (bool profitable, uint256 expectedProfit) = detectOpportunity(
            tokenA, tokenB, exchangeA, exchangeB, amountIn
        );
        
        require(profitable, "ArbitrageEngine: Not profitable");
        
        opportunityId = nextOpportunityId++;
        
        opportunities[opportunityId] = ArbitrageOpportunity({
            tokenA: tokenA,
            tokenB: tokenB,
            exchangeA: exchangeA,
            exchangeB: exchangeB,
            amountIn: amountIn,
            expectedProfit: expectedProfit,
            timestamp: block.timestamp,
            executed: false
        });
        
        activeOpportunities.add(opportunityId);
        
        emit OpportunityDetected(opportunityId, tokenA, tokenB, expectedProfit);
        
        return opportunityId;
    }

    /**
     * @dev Update price feed data from Chainlink
     */
    function updatePriceData(address token) external {
        require(chainlinkConsumer != address(0), "ArbitrageEngine: Chainlink not set");
        
        try IChainlink(chainlinkConsumer).getLatestPrice(token) returns (
            int256 price,
            uint256 timestamp
        ) {
            latestPrices[token] = PriceData({
                price: price,
                timestamp: timestamp,
                roundId: 0 // Simplified
            });
        } catch {
            revert("ArbitrageEngine: Price update failed");
        }
    }

    /**
     * @dev Add supported token
     */
    function addSupportedToken(address token) external onlyOwner {
        require(token != address(0), "ArbitrageEngine: Invalid token");
        supportedTokens[token] = true;
    }

    /**
     * @dev Add supported exchange
     */
    function addSupportedExchange(address exchange) external onlyOwner {
        require(exchange != address(0), "ArbitrageEngine: Invalid exchange");
        supportedExchanges[exchange] = true;
    }

    /**
     * @dev Remove supported token
     */
    function removeSupportedToken(address token) external onlyOwner {
        supportedTokens[token] = false;
    }

    /**
     * @dev Remove supported exchange
     */
    function removeSupportedExchange(address exchange) external onlyOwner {
        supportedExchanges[exchange] = false;
    }

    /**
     * @dev Get active opportunities
     */
    function getActiveOpportunities() external view returns (uint256[] memory) {
        uint256 length = activeOpportunities.length();
        uint256[] memory ids = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            ids[i] = activeOpportunities.at(i);
        }
        
        return ids;
    }

    /**
     * @dev Internal function to execute arbitrage
     */
    function _executeArbitrageInternal(
        ArbitrageOpportunity memory opportunity,
        uint256 expectedProfit
    ) internal returns (bool) {
        // This would integrate with actual DEX contracts
        // For production, implement actual swap logic with DEX routers
        
        // Step 1: Swap tokenA to tokenB on exchangeA
        bool step1Success = _performSwap(
            opportunity.tokenA,
            opportunity.tokenB,
            opportunity.exchangeA,
            opportunity.amountIn
        );
        
        if (!step1Success) return false;
        
        // Step 2: Swap tokenB back to tokenA on exchangeB
        uint256 tokenBAmount = _getSwapOutput(
            opportunity.tokenA,
            opportunity.tokenB,
            opportunity.exchangeA,
            opportunity.amountIn
        );
        
        bool step2Success = _performSwap(
            opportunity.tokenB,
            opportunity.tokenA,
            opportunity.exchangeB,
            tokenBAmount
        );
        
        return step2Success;
    }

    /**
     * @dev Get exchange rate between two tokens on specific exchange
     */
    function _getExchangeRate(
        address tokenIn,
        address tokenOut,
        address exchange,
        uint256 amountIn
    ) internal view returns (uint256) {
        // In production, this would query actual DEX contracts
        // Using Chainlink price feeds as fallback
        PriceData memory priceIn = latestPrices[tokenIn];
        PriceData memory priceOut = latestPrices[tokenOut];
        
        if (priceIn.price > 0 && priceOut.price > 0) {
            return (amountIn * uint256(priceOut.price)) / uint256(priceIn.price);
        }
        
        return 0;
    }

    /**
     * @dev Calculate transaction costs (gas + fees)
     */
    function _calculateTransactionCosts(uint256 amountIn) internal view returns (uint256) {
        // Simplified cost calculation
        uint256 gasCost = tx.gasprice * 200000; // Estimated gas usage
        uint256 exchangeFees = (amountIn * 30) / 10000; // 0.3% exchange fees
        
        return gasCost + exchangeFees;
    }

    /**
     * @dev Perform token swap (simplified)
     */
    function _performSwap(
        address tokenIn,
        address tokenOut,
        address exchange,
        uint256 amountIn
    ) internal returns (bool) {
        // In production, integrate with actual DEX router contracts
        // This is a simplified implementation
        return true;
    }

    /**
     * @dev Get swap output amount (simplified)
     */
    function _getSwapOutput(
        address tokenIn,
        address tokenOut,
        address exchange,
        uint256 amountIn
    ) internal view returns (uint256) {
        // In production, query actual DEX contracts
        return _getExchangeRate(tokenIn, tokenOut, exchange, amountIn);
    }

    /**
     * @dev Update Chainlink consumer address
     */
    function updateChainlinkConsumer(address _chainlinkConsumer) external onlyOwner {
        chainlinkConsumer = _chainlinkConsumer;
    }

    /**
     * @dev Update ArbOS core address
     */
    function updateArbOSCore(address _arbOSCore) external onlyOwner {
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Emergency withdrawal function
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        if (token == address(0)) {
            payable(owner()).transfer(amount);
        } else {
            IERC20(token).safeTransfer(owner(), amount);
        }
    }
}
