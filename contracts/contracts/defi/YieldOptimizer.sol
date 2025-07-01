// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {AggregatorV3Interface} from "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title YieldOptimizer
 * @dev DeFi yield optimization with Chainlink price feeds integration
 * @notice Automatically allocates funds to highest-yielding protocols
 */
contract YieldOptimizer is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // Yield strategy struct
    struct YieldStrategy {
        address protocol;
        address token;
        uint256 apy; // Annual Percentage Yield in basis points
        uint256 tvl; // Total Value Locked
        uint256 riskScore; // Risk score 0-1000
        bool active;
        uint256 lastUpdated;
    }

    // User position struct
    struct UserPosition {
        uint256 totalDeposited;
        uint256 totalEarned;
        mapping(uint256 => uint256) strategyAllocations;
        uint256 lastRebalance;
    }

    // State variables
    mapping(uint256 => YieldStrategy) public strategies;
    mapping(address => UserPosition) public userPositions;
    mapping(address => AggregatorV3Interface) public priceFeeds;
    
    uint256 public strategyCount;
    uint256 public totalFundsManaged;
    uint256 public rebalanceThreshold = 500; // 5% in basis points
    uint256 public maxRiskScore = 600; // Maximum allowed risk score
    
    address public arbOSCore;
    address public treasury;
    uint256 public managementFee = 200; // 2% in basis points

    // Events
    event StrategyAdded(uint256 indexed strategyId, address protocol, uint256 apy);
    event StrategyUpdated(uint256 indexed strategyId, uint256 newApy, uint256 riskScore);
    event FundsDeposited(address indexed user, uint256 amount, uint256 strategyId);
    event FundsWithdrawn(address indexed user, uint256 amount, uint256 strategyId);
    event Rebalanced(address indexed user, uint256 timestamp);
    event YieldHarvested(address indexed user, uint256 amount);

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "YieldOptimizer: Only core");
        _;
    }

    constructor(address _treasury, address _arbOSCore) {
        treasury = _treasury;
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Add new yield strategy
     */
    function addStrategy(
        address protocol,
        address token,
        uint256 apy,
        uint256 riskScore,
        address priceFeed
    ) external onlyOwner {
        require(protocol != address(0), "YieldOptimizer: Invalid protocol");
        require(token != address(0), "YieldOptimizer: Invalid token");
        require(riskScore <= 1000, "YieldOptimizer: Invalid risk score");

        uint256 strategyId = strategyCount++;
        
        strategies[strategyId] = YieldStrategy({
            protocol: protocol,
            token: token,
            apy: apy,
            tvl: 0,
            riskScore: riskScore,
            active: true,
            lastUpdated: block.timestamp
        });

        if (priceFeed != address(0)) {
            priceFeeds[token] = AggregatorV3Interface(priceFeed);
        }

        emit StrategyAdded(strategyId, protocol, apy);
    }

    /**
     * @dev Deposit funds into optimal yield strategy
     */
    function deposit(address token, uint256 amount) external nonReentrant {
        require(amount > 0, "YieldOptimizer: Invalid amount");
        
        IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        
        uint256 optimalStrategy = getOptimalStrategy(token);
        require(strategies[optimalStrategy].active, "YieldOptimizer: Strategy inactive");

        UserPosition storage position = userPositions[msg.sender];
        position.totalDeposited += amount;
        position.strategyAllocations[optimalStrategy] += amount;

        strategies[optimalStrategy].tvl += amount;
        totalFundsManaged += amount;

        // Deploy funds to strategy protocol
        _deployToStrategy(optimalStrategy, amount);

        emit FundsDeposited(msg.sender, amount, optimalStrategy);
    }

    /**
     * @dev Withdraw funds from strategy
     */
    function withdraw(uint256 strategyId, uint256 amount) external nonReentrant {
        UserPosition storage position = userPositions[msg.sender];
        require(
            position.strategyAllocations[strategyId] >= amount,
            "YieldOptimizer: Insufficient balance"
        );

        // Withdraw from strategy protocol
        uint256 withdrawnAmount = _withdrawFromStrategy(strategyId, amount);
        
        position.strategyAllocations[strategyId] -= amount;
        position.totalDeposited -= amount;
        
        strategies[strategyId].tvl -= amount;
        totalFundsManaged -= amount;

        address token = strategies[strategyId].token;
        IERC20(token).safeTransfer(msg.sender, withdrawnAmount);

        emit FundsWithdrawn(msg.sender, withdrawnAmount, strategyId);
    }

    /**
     * @dev Rebalance user's portfolio to optimal strategies
     */
    function rebalance() external nonReentrant {
        UserPosition storage position = userPositions[msg.sender];
        require(position.totalDeposited > 0, "YieldOptimizer: No funds to rebalance");
        
        require(
            block.timestamp >= position.lastRebalance + 1 hours,
            "YieldOptimizer: Rebalance too frequent"
        );

        // Calculate optimal allocation
        uint256[] memory optimalAllocations = calculateOptimalAllocation(
            position.totalDeposited
        );

        // Rebalance funds across strategies
        _executeRebalance(msg.sender, optimalAllocations);
        
        position.lastRebalance = block.timestamp;
        emit Rebalanced(msg.sender, block.timestamp);
    }

    /**
     * @dev Harvest yield from all strategies
     */
    function harvestYield() external nonReentrant {
        UserPosition storage position = userPositions[msg.sender];
        uint256 totalYield = 0;

        for (uint256 i = 0; i < strategyCount; i++) {
            if (position.strategyAllocations[i] > 0) {
                uint256 strategyYield = _harvestFromStrategy(i, msg.sender);
                totalYield += strategyYield;
            }
        }

        if (totalYield > 0) {
            // Deduct management fee
            uint256 fee = (totalYield * managementFee) / 10000;
            uint256 userYield = totalYield - fee;

            position.totalEarned += userYield;
            
            // Transfer fee to treasury
            if (fee > 0) {
                // Transfer fee logic here
            }

            emit YieldHarvested(msg.sender, userYield);
        }
    }

    /**
     * @dev Get optimal strategy for token
     */
    function getOptimalStrategy(address token) public view returns (uint256) {
        uint256 bestStrategy = 0;
        uint256 bestScore = 0;

        for (uint256 i = 0; i < strategyCount; i++) {
            if (strategies[i].active && strategies[i].token == token) {
                uint256 score = _calculateStrategyScore(i);
                if (score > bestScore) {
                    bestScore = score;
                    bestStrategy = i;
                }
            }
        }

        return bestStrategy;
    }

    /**
     * @dev Calculate optimal allocation across strategies
     */
    function calculateOptimalAllocation(uint256 totalAmount) 
        public 
        view 
        returns (uint256[] memory allocations) 
    {
        allocations = new uint256[](strategyCount);
        
        // Simple allocation based on risk-adjusted yield
        uint256 totalScore = 0;
        uint256[] memory scores = new uint256[](strategyCount);
        
        for (uint256 i = 0; i < strategyCount; i++) {
            if (strategies[i].active && strategies[i].riskScore <= maxRiskScore) {
                scores[i] = _calculateStrategyScore(i);
                totalScore += scores[i];
            }
        }

        if (totalScore > 0) {
            for (uint256 i = 0; i < strategyCount; i++) {
                if (scores[i] > 0) {
                    allocations[i] = (totalAmount * scores[i]) / totalScore;
                }
            }
        }

        return allocations;
    }

    /**
     * @dev Calculate strategy score (risk-adjusted yield)
     */
    function _calculateStrategyScore(uint256 strategyId) internal view returns (uint256) {
        YieldStrategy memory strategy = strategies[strategyId];
        
        // Risk-adjusted yield = APY / (1 + riskScore/1000)
        uint256 riskAdjustment = 1000 + strategy.riskScore;
        uint256 adjustedYield = (strategy.apy * 1000) / riskAdjustment;
        
        return adjustedYield;
    }

    /**
     * @dev Deploy funds to strategy protocol
     */
    function _deployToStrategy(uint256 strategyId, uint256 amount) internal {
        YieldStrategy memory strategy = strategies[strategyId];
        
        // Approve and deposit to strategy protocol
        IERC20(strategy.token).safeApprove(strategy.protocol, amount);
        
        // In production, call actual protocol deposit function
        (bool success, ) = strategy.protocol.call(
            abi.encodeWithSignature("deposit(uint256)", amount)
        );
        
        require(success, "YieldOptimizer: Strategy deposit failed");
    }

    /**
     * @dev Withdraw funds from strategy protocol
     */
    function _withdrawFromStrategy(uint256 strategyId, uint256 amount) 
        internal 
        returns (uint256 withdrawnAmount) 
    {
        YieldStrategy memory strategy = strategies[strategyId];
        
        // Call protocol withdraw function
        (bool success, bytes memory result) = strategy.protocol.call(
            abi.encodeWithSignature("withdraw(uint256)", amount)
        );
        
        require(success, "YieldOptimizer: Strategy withdrawal failed");
        
        if (result.length > 0) {
            withdrawnAmount = abi.decode(result, (uint256));
        } else {
            withdrawnAmount = amount;
        }
        
        return withdrawnAmount;
    }

    /**
     * @dev Harvest yield from strategy
     */
    function _harvestFromStrategy(uint256 strategyId, address user) 
        internal 
        returns (uint256 yield) 
    {
        YieldStrategy memory strategy = strategies[strategyId];
        
        // Call protocol harvest function
        (bool success, bytes memory result) = strategy.protocol.call(
            abi.encodeWithSignature("harvest(address)", user)
        );
        
        if (success && result.length > 0) {
            yield = abi.decode(result, (uint256));
        }
        
        return yield;
    }

    /**
     * @dev Execute portfolio rebalancing
     */
    function _executeRebalance(address user, uint256[] memory targetAllocations) internal {
        UserPosition storage position = userPositions[user];
        
        for (uint256 i = 0; i < strategyCount; i++) {
            uint256 currentAllocation = position.strategyAllocations[i];
            uint256 targetAllocation = targetAllocations[i];
            
            if (currentAllocation > targetAllocation) {
                // Withdraw excess
                uint256 excess = currentAllocation - targetAllocation;
                _withdrawFromStrategy(i, excess);
                position.strategyAllocations[i] = targetAllocation;
                strategies[i].tvl -= excess;
            } else if (targetAllocation > currentAllocation) {
                // Deposit additional
                uint256 additional = targetAllocation - currentAllocation;
                _deployToStrategy(i, additional);
                position.strategyAllocations[i] = targetAllocation;
                strategies[i].tvl += additional;
            }
        }
    }

    /**
     * @dev Update strategy parameters
     */
    function updateStrategy(
        uint256 strategyId,
        uint256 newApy,
        uint256 newRiskScore
    ) external onlyOwner {
        require(strategyId < strategyCount, "YieldOptimizer: Invalid strategy");
        
        strategies[strategyId].apy = newApy;
        strategies[strategyId].riskScore = newRiskScore;
        strategies[strategyId].lastUpdated = block.timestamp;
        
        emit StrategyUpdated(strategyId, newApy, newRiskScore);
    }

    /**
     * @dev Get user position details
     */
    function getUserPosition(address user) 
        external 
        view 
        returns (
            uint256 totalDeposited,
            uint256 totalEarned,
            uint256[] memory allocations
        ) 
    {
        UserPosition storage position = userPositions[user];
        totalDeposited = position.totalDeposited;
        totalEarned = position.totalEarned;
        
        allocations = new uint256[](strategyCount);
        for (uint256 i = 0; i < strategyCount; i++) {
            allocations[i] = position.strategyAllocations[i];
        }
        
        return (totalDeposited, totalEarned, allocations);
    }

    /**
     * @dev Get strategy details
     */
    function getStrategy(uint256 strategyId) 
        external 
        view 
        returns (YieldStrategy memory) 
    {
        return strategies[strategyId];
    }

    /**
     * @dev Update management fee
     */
    function updateManagementFee(uint256 newFee) external onlyOwner {
        require(newFee <= 1000, "YieldOptimizer: Fee too high"); // Max 10%
        managementFee = newFee;
    }

    /**
     * @dev Emergency withdraw for user
     */
    function emergencyWithdraw() external nonReentrant {
        UserPosition storage position = userPositions[msg.sender];
        require(position.totalDeposited > 0, "YieldOptimizer: No funds");
        
        uint256 totalWithdrawn = 0;
        
        for (uint256 i = 0; i < strategyCount; i++) {
            if (position.strategyAllocations[i] > 0) {
                uint256 withdrawn = _withdrawFromStrategy(i, position.strategyAllocations[i]);
                totalWithdrawn += withdrawn;
                
                strategies[i].tvl -= position.strategyAllocations[i];
                position.strategyAllocations[i] = 0;
            }
        }
        
        position.totalDeposited = 0;
        totalFundsManaged -= totalWithdrawn;
        
        // Transfer all funds back to user
        // Note: This assumes all strategies use the same base token
        // In production, handle multiple tokens appropriately
    }
}
