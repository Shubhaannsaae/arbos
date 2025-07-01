// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IPortfolio
 * @dev Interface for portfolio management operations
 */
interface IPortfolio {
    struct Asset {
        address token;
        uint256 allocation; // Basis points (10000 = 100%)
        uint256 currentValue;
        uint256 targetValue;
        uint256 lastRebalance;
    }

    struct RiskMetrics {
        uint256 totalValue;
        uint256 volatility;
        uint256 sharpeRatio;
        uint256 maxDrawdown;
        uint256 beta;
    }

    event PortfolioRebalanced(
        address indexed user,
        uint256 totalValue,
        uint256 timestamp
    );

    event AssetAdded(
        address indexed user,
        address indexed token,
        uint256 allocation
    );

    event RiskAssessmentUpdated(
        address indexed user,
        uint256 riskScore,
        uint256 timestamp
    );

    function addAsset(address token, uint256 allocation) external;
    
    function removeAsset(address token) external;
    
    function rebalancePortfolio() external;
    
    function calculateRiskMetrics() external view returns (RiskMetrics memory);
    
    function getPortfolioValue() external view returns (uint256);
    
    function getAssetAllocation(address token) external view returns (uint256);
    
    function shouldRebalance() external view returns (bool);
}
