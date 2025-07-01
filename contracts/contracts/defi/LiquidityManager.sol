// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {AggregatorV3Interface} from "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title LiquidityManager
 * @dev Manages liquidity across DEX pools with dynamic optimization
 * @notice Provides liquidity to AMM pools with automated position management
 */
contract LiquidityManager is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // Liquidity pool struct
    struct LiquidityPool {
        address poolAddress;
        address tokenA;
        address tokenB;
        uint256 feeRate; // Fee rate in basis points
        uint256 totalLiquidity;
        uint256 apr; // Annual Percentage Rate
        bool active;
        uint256 lastUpdated;
    }

    // User position struct
    struct UserPosition {
        uint256 poolId;
        uint256 liquidity;
        uint256 tokenAAmount;
        uint256 tokenBAmount;
        uint256 feesEarned;
        uint256 lastUpdate;
        uint256 positionId; // NFT position ID for concentrated liquidity
    }

    // Pool performance metrics
    struct PoolMetrics {
        uint256 volume24h;
        uint256 tvl;
        uint256 impermanentLoss;
        uint256 feeAPR;
        uint256 utilization;
    }

    // State variables
    mapping(uint256 => LiquidityPool) public pools;
    mapping(address => mapping(uint256 => UserPosition)) public userPositions;
    mapping(address => uint256[]) public userPoolIds;
    mapping(address => AggregatorV3Interface) public priceFeeds;
    mapping(uint256 => PoolMetrics) public poolMetrics;

    uint256 public poolCount;
    uint256 public totalValueLocked;
    uint256 public rebalanceThreshold = 500; // 5% price deviation
    
    address public arbOSCore;
    address public uniswapV3Factory;
    address public positionManager;

    // Events
    event LiquidityAdded(
        address indexed user,
        uint256 indexed poolId,
        uint256 tokenAAmount,
        uint256 tokenBAmount,
        uint256 liquidity
    );
    
    event LiquidityRemoved(
        address indexed user,
        uint256 indexed poolId,
        uint256 liquidity,
        uint256 tokenAAmount,
        uint256 tokenBAmount
    );
    
    event FeesCollected(
        address indexed user,
        uint256 indexed poolId,
        uint256 amount
    );
    
    event PositionRebalanced(
        address indexed user,
        uint256 indexed poolId,
        uint256 newLiquidity
    );

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "LiquidityManager: Only core");
        _;
    }

    constructor(
        address _arbOSCore,
        address _uniswapV3Factory,
        address _positionManager
    ) {
        arbOSCore = _arbOSCore;
        uniswapV3Factory = _uniswapV3Factory;
        positionManager = _positionManager;
    }

    /**
     * @dev Add new liquidity pool
     */
    function addPool(
        address poolAddress,
        address tokenA,
        address tokenB,
        uint256 feeRate,
        address priceFeedA,
        address priceFeedB
    ) external onlyOwner {
        require(poolAddress != address(0), "LiquidityManager: Invalid pool");
        
        uint256 poolId = poolCount++;
        
        pools[poolId] = LiquidityPool({
            poolAddress: poolAddress,
            tokenA: tokenA,
            tokenB: tokenB,
            feeRate: feeRate,
            totalLiquidity: 0,
            apr: 0,
            active: true,
            lastUpdated: block.timestamp
        });

        // Set price feeds
        if (priceFeedA != address(0)) {
            priceFeeds[tokenA] = AggregatorV3Interface(priceFeedA);
        }
        if (priceFeedB != address(0)) {
            priceFeeds[tokenB] = AggregatorV3Interface(priceFeedB);
        }
    }

    /**
     * @dev Add liquidity to pool
     */
    function addLiquidity(
        uint256 poolId,
        uint256 tokenAAmount,
        uint256 tokenBAmount,
        uint256 minTokenAAmount,
        uint256 minTokenBAmount
    ) external nonReentrant {
        require(poolId < poolCount, "LiquidityManager: Invalid pool");
        require(pools[poolId].active, "LiquidityManager: Pool inactive");
        
        LiquidityPool storage pool = pools[poolId];
        
        // Transfer tokens from user
        IERC20(pool.tokenA).safeTransferFrom(msg.sender, address(this), tokenAAmount);
        IERC20(pool.tokenB).safeTransferFrom(msg.sender, address(this), tokenBAmount);
        
        // Calculate optimal amounts and add liquidity
        (uint256 actualAmountA, uint256 actualAmountB, uint256 liquidity) = 
            _addLiquidityToPool(poolId, tokenAAmount, tokenBAmount);
        
        require(actualAmountA >= minTokenAAmount, "LiquidityManager: Insufficient A amount");
        require(actualAmountB >= minTokenBAmount, "LiquidityManager: Insufficient B amount");
        
        // Update user position
        UserPosition storage position = userPositions[msg.sender][poolId];
        if (position.liquidity == 0) {
            userPoolIds[msg.sender].push(poolId);
        }
        
        position.poolId = poolId;
        position.liquidity += liquidity;
        position.tokenAAmount += actualAmountA;
        position.tokenBAmount += actualAmountB;
        position.lastUpdate = block.timestamp;
        
        // Update pool metrics
        pool.totalLiquidity += liquidity;
        totalValueLocked += _calculatePositionValue(poolId, actualAmountA, actualAmountB);
        
        // Return excess tokens
        if (tokenAAmount > actualAmountA) {
            IERC20(pool.tokenA).safeTransfer(msg.sender, tokenAAmount - actualAmountA);
        }
        if (tokenBAmount > actualAmountB) {
            IERC20(pool.tokenB).safeTransfer(msg.sender, tokenBAmount - actualAmountB);
        }
        
        emit LiquidityAdded(msg.sender, poolId, actualAmountA, actualAmountB, liquidity);
    }

    /**
     * @dev Remove liquidity from pool
     */
    function removeLiquidity(
        uint256 poolId,
        uint256 liquidity,
        uint256 minTokenAAmount,
        uint256 minTokenBAmount
    ) external nonReentrant {
        UserPosition storage position = userPositions[msg.sender][poolId];
        require(position.liquidity >= liquidity, "LiquidityManager: Insufficient liquidity");
        
        // Remove liquidity from pool
        (uint256 tokenAAmount, uint256 tokenBAmount) = 
            _removeLiquidityFromPool(poolId, liquidity);
        
        require(tokenAAmount >= minTokenAAmount, "LiquidityManager: Insufficient A amount");
        require(tokenBAmount >= minTokenBAmount, "LiquidityManager: Insufficient B amount");
        
        // Update position
        position.liquidity -= liquidity;
        position.tokenAAmount -= tokenAAmount;
        position.tokenBAmount -= tokenBAmount;
        
        // Update pool metrics
        pools[poolId].totalLiquidity -= liquidity;
        totalValueLocked -= _calculatePositionValue(poolId, tokenAAmount, tokenBAmount);
        
        // Transfer tokens to user
        LiquidityPool memory pool = pools[poolId];
        IERC20(pool.tokenA).safeTransfer(msg.sender, tokenAAmount);
        IERC20(pool.tokenB).safeTransfer(msg.sender, tokenBAmount);
        
        emit LiquidityRemoved(msg.sender, poolId, liquidity, tokenAAmount, tokenBAmount);
    }

    /**
     * @dev Collect fees from position
     */
    function collectFees(uint256 poolId) external nonReentrant {
        UserPosition storage position = userPositions[msg.sender][poolId];
        require(position.liquidity > 0, "LiquidityManager: No position");
        
        uint256 fees = _collectFeesFromPool(poolId, msg.sender);
        
        if (fees > 0) {
            position.feesEarned += fees;
            emit FeesCollected(msg.sender, poolId, fees);
        }
    }

    /**
     * @dev Rebalance position based on current market conditions
     */
    function rebalancePosition(uint256 poolId) external nonReentrant {
        UserPosition storage position = userPositions[msg.sender][poolId];
        require(position.liquidity > 0, "LiquidityManager: No position");
        
        // Check if rebalancing is needed
        require(_shouldRebalance(poolId), "LiquidityManager: Rebalancing not needed");
        
        // Remove current liquidity
        (uint256 tokenAAmount, uint256 tokenBAmount) = 
            _removeLiquidityFromPool(poolId, position.liquidity);
        
        // Calculate optimal new range and amounts
        (uint256 newTokenAAmount, uint256 newTokenBAmount) = 
            _calculateOptimalAmounts(poolId, tokenAAmount, tokenBAmount);
        
        // Add liquidity with new parameters
        (uint256 actualAmountA, uint256 actualAmountB, uint256 newLiquidity) = 
            _addLiquidityToPool(poolId, newTokenAAmount, newTokenBAmount);
        
        // Update position
        position.liquidity = newLiquidity;
        position.tokenAAmount = actualAmountA;
        position.tokenBAmount = actualAmountB;
        position.lastUpdate = block.timestamp;
        
        emit PositionRebalanced(msg.sender, poolId, newLiquidity);
    }

    /**
     * @dev Get optimal pool for token pair
     */
    function getOptimalPool(address tokenA, address tokenB) 
        external 
        view 
        returns (uint256 poolId, uint256 expectedAPR) 
    {
        uint256 bestPool = 0;
        uint256 bestAPR = 0;
        
        for (uint256 i = 0; i < poolCount; i++) {
            LiquidityPool memory pool = pools[i];
            if (pool.active && 
                ((pool.tokenA == tokenA && pool.tokenB == tokenB) ||
                 (pool.tokenA == tokenB && pool.tokenB == tokenA))) {
                
                uint256 poolAPR = _calculatePoolAPR(i);
                if (poolAPR > bestAPR) {
                    bestAPR = poolAPR;
                    bestPool = i;
                }
            }
        }
        
        return (bestPool, bestAPR);
    }

    /**
     * @dev Calculate position value in USD
     */
    function getPositionValue(address user, uint256 poolId) 
        external 
        view 
        returns (uint256 totalValue, uint256 impermanentLoss) 
    {
        UserPosition memory position = userPositions[user][poolId];
        
        if (position.liquidity == 0) {
            return (0, 0);
        }
        
        totalValue = _calculatePositionValue(poolId, position.tokenAAmount, position.tokenBAmount);
        impermanentLoss = _calculateImpermanentLoss(poolId, position);
        
        return (totalValue, impermanentLoss);
    }

    /**
     * @dev Internal function to add liquidity to pool
     */
    function _addLiquidityToPool(
        uint256 poolId,
        uint256 tokenAAmount,
        uint256 tokenBAmount
    ) internal returns (uint256 actualAmountA, uint256 actualAmountB, uint256 liquidity) {
        LiquidityPool memory pool = pools[poolId];
        
        // Approve tokens for position manager
        IERC20(pool.tokenA).safeApprove(positionManager, tokenAAmount);
        IERC20(pool.tokenB).safeApprove(positionManager, tokenBAmount);
        
        // In production, call actual Uniswap V3 position manager
        // This is a simplified implementation
        actualAmountA = tokenAAmount;
        actualAmountB = tokenBAmount;
        liquidity = _calculateLiquidity(tokenAAmount, tokenBAmount);
        
        return (actualAmountA, actualAmountB, liquidity);
    }

    /**
     * @dev Internal function to remove liquidity from pool
     */
    function _removeLiquidityFromPool(
        uint256 poolId,
        uint256 liquidity
    ) internal returns (uint256 tokenAAmount, uint256 tokenBAmount) {
        // In production, call actual Uniswap V3 position manager
        // This is a simplified implementation
        
        UserPosition memory position = userPositions[msg.sender][poolId];
        uint256 liquidityShare = (liquidity * 10000) / position.liquidity;
        
        tokenAAmount = (position.tokenAAmount * liquidityShare) / 10000;
        tokenBAmount = (position.tokenBAmount * liquidityShare) / 10000;
        
        return (tokenAAmount, tokenBAmount);
    }

    /**
     * @dev Collect fees from pool position
     */
    function _collectFeesFromPool(uint256 poolId, address user) 
        internal 
        returns (uint256 fees) 
    {
        // In production, call actual pool fee collection
        // This is a simplified implementation
        
        UserPosition memory position = userPositions[user][poolId];
        LiquidityPool memory pool = pools[poolId];
        
        // Calculate fees based on liquidity and time
        uint256 timeElapsed = block.timestamp - position.lastUpdate;
        uint256 feeRate = pool.feeRate; // basis points
        
        fees = (position.liquidity * feeRate * timeElapsed) / (365 days * 10000);
        
        return fees;
    }

    /**
     * @dev Check if position should be rebalanced
     */
    function _shouldRebalance(uint256 poolId) internal view returns (bool) {
        // Get current price from Chainlink
        LiquidityPool memory pool = pools[poolId];
        
        if (address(priceFeeds[pool.tokenA]) == address(0) || 
            address(priceFeeds[pool.tokenB]) == address(0)) {
            return false;
        }
        
        try priceFeeds[pool.tokenA].latestRoundData() returns (
            uint80, int256 priceA, uint256, uint256, uint80
        ) {
            try priceFeeds[pool.tokenB].latestRoundData() returns (
                uint80, int256 priceB, uint256, uint256, uint80
            ) {
                // Calculate price ratio and compare with position ratio
                // Simplified logic - in production, implement proper range checking
                return true; // Always allow rebalancing for now
            } catch {
                return false;
            }
        } catch {
            return false;
        }
    }

    /**
     * @dev Calculate optimal token amounts for rebalancing
     */
    function _calculateOptimalAmounts(
        uint256 poolId,
        uint256 currentAmountA,
        uint256 currentAmountB
    ) internal view returns (uint256 optimalAmountA, uint256 optimalAmountB) {
        // Simplified calculation - maintain 50/50 value ratio
        uint256 totalValue = _calculatePositionValue(poolId, currentAmountA, currentAmountB);
        
        optimalAmountA = currentAmountA;
        optimalAmountB = currentAmountB;
        
        return (optimalAmountA, optimalAmountB);
    }

    /**
     * @dev Calculate position value in USD
     */
    function _calculatePositionValue(
        uint256 poolId,
        uint256 tokenAAmount,
        uint256 tokenBAmount
    ) internal view returns (uint256 totalValue) {
        LiquidityPool memory pool = pools[poolId];
        
        uint256 valueA = _getTokenValueUSD(pool.tokenA, tokenAAmount);
        uint256 valueB = _getTokenValueUSD(pool.tokenB, tokenBAmount);
        
        return valueA + valueB;
    }

    /**
     * @dev Get token value in USD using Chainlink price feed
     */
    function _getTokenValueUSD(address token, uint256 amount) 
        internal 
        view 
        returns (uint256 value) 
    {
        if (address(priceFeeds[token]) == address(0)) {
            return 0;
        }
        
        try priceFeeds[token].latestRoundData() returns (
            uint80, int256 price, uint256, uint256, uint80
        ) {
            uint8 decimals = priceFeeds[token].decimals();
            value = (amount * uint256(price)) / (10 ** decimals);
        } catch {
            value = 0;
        }
        
        return value;
    }

    /**
     * @dev Calculate liquidity amount (simplified)
     */
    function _calculateLiquidity(uint256 amountA, uint256 amountB) 
        internal 
        pure 
        returns (uint256) 
    {
        // Simplified liquidity calculation
        return (amountA * amountB) / 1e18;
    }

    /**
     * @dev Calculate pool APR
     */
    function _calculatePoolAPR(uint256 poolId) internal view returns (uint256) {
        PoolMetrics memory metrics = poolMetrics[poolId];
        LiquidityPool memory pool = pools[poolId];
        
        if (metrics.tvl == 0) return 0;
        
        // APR = (24h volume * fee rate * 365) / TVL
        uint256 annualFees = (metrics.volume24h * pool.feeRate * 365) / 10000;
        uint256 apr = (annualFees * 10000) / metrics.tvl; // in basis points
        
        return apr;
    }

    /**
     * @dev Calculate impermanent loss
     */
    function _calculateImpermanentLoss(uint256 poolId, UserPosition memory position) 
        internal 
        view 
        returns (uint256) 
    {
        // Simplified IL calculation
        // In production, implement proper IL calculation based on price changes
        return 0;
    }

    /**
     * @dev Update pool metrics
     */
    function updatePoolMetrics(
        uint256 poolId,
        uint256 volume24h,
        uint256 tvl
    ) external onlyOwner {
        poolMetrics[poolId].volume24h = volume24h;
        poolMetrics[poolId].tvl = tvl;
        poolMetrics[poolId].feeAPR = _calculatePoolAPR(poolId);
    }
}
