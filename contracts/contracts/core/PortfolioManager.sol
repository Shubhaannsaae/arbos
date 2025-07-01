// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

import {IPortfolio} from "../interfaces/IPortfolio.sol";
import {IChainlink} from "../interfaces/IChainlink.sol";

/**
 * @title PortfolioManager
 * @dev Manages portfolio allocation and rebalancing with Chainlink price feeds
 */
contract PortfolioManager is IPortfolio, Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using EnumerableSet for EnumerableSet.AddressSet;

    // Constants
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MAX_ASSETS = 20;
    uint256 public constant REBALANCE_THRESHOLD = 500; // 5%
    uint256 public constant MIN_REBALANCE_INTERVAL = 1 hours;

    // Portfolio data
    mapping(address => mapping(address => Asset)) public userAssets;
    mapping(address => EnumerableSet.AddressSet) private userAssetsList;
    mapping(address => uint256) public totalAllocations;
    mapping(address => uint256) public lastRebalanceTime;
    mapping(address => RiskMetrics) public userRiskMetrics;

    // System contracts
    address public chainlinkConsumer;
    address public arbOSCore;

    // Events from interface
    
    modifier onlyCore() {
        require(msg.sender == arbOSCore, "PortfolioManager: Only core");
        _;
    }

    modifier validAllocation(uint256 allocation) {
        require(allocation > 0 && allocation <= BASIS_POINTS, "PortfolioManager: Invalid allocation");
        _;
    }

    constructor(address _chainlinkConsumer, address _arbOSCore) {
        chainlinkConsumer = _chainlinkConsumer;
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Add asset to user's portfolio
     * @param token Asset token address
     * @param allocation Allocation in basis points
     */
    function addAsset(address token, uint256 allocation) 
        external 
        override 
        validAllocation(allocation) 
        nonReentrant 
    {
        address user = msg.sender;
        
        require(
            userAssetsList[user].length() < MAX_ASSETS,
            "PortfolioManager: Max assets exceeded"
        );
        
        require(
            totalAllocations[user] + allocation <= BASIS_POINTS,
            "PortfolioManager: Total allocation exceeded"
        );

        if (userAssetsList[user].contains(token)) {
            // Update existing asset
            Asset storage asset = userAssets[user][token];
            totalAllocations[user] = totalAllocations[user] - asset.allocation + allocation;
            asset.allocation = allocation;
            asset.lastRebalance = block.timestamp;
        } else {
            // Add new asset
            userAssetsList[user].add(token);
            userAssets[user][token] = Asset({
                token: token,
                allocation: allocation,
                currentValue: 0,
                targetValue: 0,
                lastRebalance: block.timestamp
            });
            totalAllocations[user] += allocation;
        }

        _updateAssetValues(user);
        emit AssetAdded(user, token, allocation);
    }

    /**
     * @dev Remove asset from user's portfolio
     * @param token Asset token address
     */
    function removeAsset(address token) external override nonReentrant {
        address user = msg.sender;
        
        require(
            userAssetsList[user].contains(token),
            "PortfolioManager: Asset not found"
        );

        Asset memory asset = userAssets[user][token];
        totalAllocations[user] -= asset.allocation;
        
        userAssetsList[user].remove(token);
        delete userAssets[user][token];

        _updateAssetValues(user);
    }

    /**
     * @dev Rebalance user's portfolio
     */
    function rebalancePortfolio() external override nonReentrant {
        address user = msg.sender;
        
        require(shouldRebalance(), "PortfolioManager: Rebalancing not needed");
        require(
            block.timestamp >= lastRebalanceTime[user] + MIN_REBALANCE_INTERVAL,
            "PortfolioManager: Rebalancing too frequent"
        );

        uint256 totalValue = getPortfolioValue();
        require(totalValue > 0, "PortfolioManager: No portfolio value");

        // Update target values based on current allocations
        EnumerableSet.AddressSet storage assetsList = userAssetsList[user];
        uint256 length = assetsList.length();
        
        for (uint256 i = 0; i < length; i++) {
            address token = assetsList.at(i);
            Asset storage asset = userAssets[user][token];
            
            uint256 targetValue = (totalValue * asset.allocation) / BASIS_POINTS;
            asset.targetValue = targetValue;
            asset.lastRebalance = block.timestamp;
        }

        lastRebalanceTime[user] = block.timestamp;
        _updateRiskMetrics(user);
        
        emit PortfolioRebalanced(user, totalValue, block.timestamp);
    }

    /**
     * @dev Calculate risk metrics for user's portfolio
     */
    function calculateRiskMetrics() external view override returns (RiskMetrics memory) {
        address user = msg.sender;
        return userRiskMetrics[user];
    }

    /**
     * @dev Get total portfolio value
     */
    function getPortfolioValue() public view override returns (uint256) {
        address user = msg.sender;
        uint256 totalValue = 0;
        
        EnumerableSet.AddressSet storage assetsList = userAssetsList[user];
        uint256 length = assetsList.length();
        
        for (uint256 i = 0; i < length; i++) {
            address token = assetsList.at(i);
            Asset memory asset = userAssets[user][token];
            totalValue += asset.currentValue;
        }
        
        return totalValue;
    }

    /**
     * @dev Get asset allocation for specific token
     */
    function getAssetAllocation(address token) external view override returns (uint256) {
        return userAssets[msg.sender][token].allocation;
    }

    /**
     * @dev Check if portfolio should be rebalanced
     */
    function shouldRebalance() public view override returns (bool) {
        address user = msg.sender;
        
        if (block.timestamp < lastRebalanceTime[user] + MIN_REBALANCE_INTERVAL) {
            return false;
        }

        uint256 totalValue = getPortfolioValue();
        if (totalValue == 0) return false;

        EnumerableSet.AddressSet storage assetsList = userAssetsList[user];
        uint256 length = assetsList.length();
        
        for (uint256 i = 0; i < length; i++) {
            address token = assetsList.at(i);
            Asset memory asset = userAssets[user][token];
            
            uint256 currentWeight = (asset.currentValue * BASIS_POINTS) / totalValue;
            uint256 targetWeight = asset.allocation;
            
            uint256 drift = currentWeight > targetWeight 
                ? currentWeight - targetWeight 
                : targetWeight - currentWeight;
            
            if (drift >= REBALANCE_THRESHOLD) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * @dev Update asset values using Chainlink prices
     */
    function _updateAssetValues(address user) internal {
        EnumerableSet.AddressSet storage assetsList = userAssetsList[user];
        uint256 length = assetsList.length();
        
        for (uint256 i = 0; i < length; i++) {
            address token = assetsList.at(i);
            Asset storage asset = userAssets[user][token];
            
            // Get price from Chainlink
            if (chainlinkConsumer != address(0)) {
                try IChainlink(chainlinkConsumer).getLatestPrice(token) returns (
                    int256 price, 
                    uint256 /* timestamp */
                ) {
                    if (price > 0) {
                        uint256 balance = IERC20(token).balanceOf(user);
                        asset.currentValue = (balance * uint256(price)) / 1e18;
                    }
                } catch {
                    // Price feed failed, keep current value
                }
            }
        }
    }

    /**
     * @dev Update risk metrics for user
     */
    function _updateRiskMetrics(address user) internal {
        uint256 totalValue = getPortfolioValue();
        
        // Simplified risk calculation
        RiskMetrics storage metrics = userRiskMetrics[user];
        metrics.totalValue = totalValue;
        metrics.volatility = _calculatePortfolioVolatility(user);
        metrics.sharpeRatio = _calculateSharpeRatio(user);
        metrics.maxDrawdown = _calculateMaxDrawdown(user);
        metrics.beta = _calculateBeta(user);
        
        emit RiskAssessmentUpdated(user, metrics.volatility, block.timestamp);
    }

    /**
     * @dev Calculate portfolio volatility (simplified)
     */
    function _calculatePortfolioVolatility(address user) internal view returns (uint256) {
        // Simplified volatility calculation
        EnumerableSet.AddressSet storage assetsList = userAssetsList[user];
        uint256 length = assetsList.length();
        
        if (length == 0) return 0;
        
        // Return normalized volatility based on number of assets (diversification)
        return 1000 / length; // Simplified: more assets = lower volatility
    }

    /**
     * @dev Calculate Sharpe ratio (simplified)
     */
    function _calculateSharpeRatio(address user) internal view returns (uint256) {
        uint256 volatility = _calculatePortfolioVolatility(user);
        if (volatility == 0) return 0;
        
        // Simplified Sharpe ratio calculation
        return 1000 / volatility;
    }

    /**
     * @dev Calculate maximum drawdown (simplified)
     */
    function _calculateMaxDrawdown(address /* user */) internal pure returns (uint256) {
        // Simplified max drawdown - in production, track historical values
        return 500; // 5%
    }

    /**
     * @dev Calculate beta (simplified)
     */
    function _calculateBeta(address /* user */) internal pure returns (uint256) {
        // Simplified beta calculation
        return 1000; // Beta of 1.0
    }

    /**
     * @dev Get user's assets list
     */
    function getUserAssets(address user) external view returns (address[] memory) {
        EnumerableSet.AddressSet storage assetsList = userAssetsList[user];
        uint256 length = assetsList.length();
        address[] memory assets = new address[](length);
        
        for (uint256 i = 0; i < length; i++) {
            assets[i] = assetsList.at(i);
        }
        
        return assets;
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
}
