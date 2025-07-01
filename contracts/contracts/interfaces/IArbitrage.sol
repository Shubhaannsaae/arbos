// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IArbitrage
 * @dev Interface for arbitrage operations with Chainlink integration
 */
interface IArbitrage {
    struct ArbitrageOpportunity {
        address tokenA;
        address tokenB;
        address exchangeA;
        address exchangeB;
        uint256 amountIn;
        uint256 expectedProfit;
        uint256 timestamp;
        bool executed;
    }

    struct PriceData {
        int256 price;
        uint256 timestamp;
        uint80 roundId;
    }

    event ArbitrageExecuted(
        uint256 indexed opportunityId,
        address indexed tokenA,
        address indexed tokenB,
        uint256 profit,
        uint256 timestamp
    );

    event OpportunityDetected(
        uint256 indexed opportunityId,
        address indexed tokenA,
        address indexed tokenB,
        uint256 expectedProfit
    );

    function detectOpportunity(
        address tokenA,
        address tokenB,
        address exchangeA,
        address exchangeB,
        uint256 amount
    ) external view returns (bool profitable, uint256 expectedProfit);

    function executeArbitrage(uint256 opportunityId) external returns (bool success);
    
    function getLatestPrice(address token) external view returns (PriceData memory);
    
    function calculateProfit(
        address tokenA,
        address tokenB,
        uint256 amountIn,
        address exchangeA,
        address exchangeB
    ) external view returns (int256 profit);
}
