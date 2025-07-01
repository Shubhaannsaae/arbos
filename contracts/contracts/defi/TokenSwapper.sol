// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {AggregatorV3Interface} from "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";

/**
 * @title TokenSwapper
 * @dev Advanced token swapping with MEV protection and optimal routing
 * @notice Integrates with multiple DEXs for best execution prices
 */
contract TokenSwapper is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // Swap route struct
    struct SwapRoute {
        address dex;
        address[] path;
        uint256 expectedOutput;
        uint256 gasEstimate;
        uint256 fee;
    }

    // Swap order struct
    struct SwapOrder {
        address user;
        address tokenIn;
        address tokenOut;
        uint256 amountIn;
        uint256 minAmountOut;
        uint256 deadline;
        uint256 maxSlippage; // in basis points
        bool executed;
        uint256 executedAmount;
    }

    // DEX configuration
    struct DEXConfig {
        address router;
        uint256 fee; // in basis points
        bool active;
        uint256 gasOverhead;
    }

    // State variables
    mapping(address => DEXConfig) public dexConfigs;
    mapping(address => AggregatorV3Interface) public priceFeeds;
    mapping(bytes32 => SwapOrder) public swapOrders;
    mapping(address => uint256) public userNonces;
    
    address[] public supportedDEXs;
    uint256 public maxSlippage = 500; // 5% default max slippage
    uint256 public mevProtectionDelay = 12; // 12 seconds
    
    address public arbOSCore;
    address public treasury;
    uint256 public protocolFee = 10; // 0.1% in basis points

    // Events
    event SwapExecuted(
        bytes32 indexed orderId,
        address indexed user,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        address dex
    );
    
    event OrderCreated(
        bytes32 indexed orderId,
        address indexed user,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    );
    
    event RouteOptimized(
        address indexed tokenIn,
        address indexed tokenOut,
        address bestDex,
        uint256 expectedOutput
    );

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "TokenSwapper: Only core");
        _;
    }

    constructor(address _arbOSCore, address _treasury) {
        arbOSCore = _arbOSCore;
        treasury = _treasury;
    }

    /**
     * @dev Add DEX configuration
     */
    function addDEX(
        address dex,
        address router,
        uint256 fee,
        uint256 gasOverhead
    ) external onlyOwner {
        require(dex != address(0), "TokenSwapper: Invalid DEX");
        require(router != address(0), "TokenSwapper: Invalid router");
        
        if (!dexConfigs[dex].active) {
            supportedDEXs.push(dex);
        }
        
        dexConfigs[dex] = DEXConfig({
            router: router,
            fee: fee,
            active: true,
            gasOverhead: gasOverhead
        });
    }

    /**
     * @dev Create swap order with MEV protection
     */
    function createSwapOrder(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint256 deadline,
        uint256 maxSlippageBps
    ) external nonReentrant returns (bytes32 orderId) {
        require(amountIn > 0, "TokenSwapper: Invalid amount");
        require(deadline > block.timestamp, "TokenSwapper: Invalid deadline");
        require(maxSlippageBps <= maxSlippage, "TokenSwapper: Slippage too high");
        
        // Transfer tokens to contract
        IERC20(tokenIn).safeTransferFrom(msg.sender, address(this), amountIn);
        
        // Generate order ID
        orderId = keccak256(abi.encodePacked(
            msg.sender,
            tokenIn,
            tokenOut,
            amountIn,
            userNonces[msg.sender]++,
            block.timestamp
        ));
        
        // Create swap order
        swapOrders[orderId] = SwapOrder({
            user: msg.sender,
            tokenIn: tokenIn,
            tokenOut: tokenOut,
            amountIn: amountIn,
            minAmountOut: minAmountOut,
            deadline: deadline,
            maxSlippage: maxSlippageBps,
            executed: false,
            executedAmount: 0
        });
        
        emit OrderCreated(orderId, msg.sender, tokenIn, tokenOut, amountIn);
        
        return orderId;
    }

    /**
     * @dev Execute swap order with optimal routing
     */
    function executeSwap(bytes32 orderId) external nonReentrant {
        SwapOrder storage order = swapOrders[orderId];
        require(!order.executed, "TokenSwapper: Order already executed");
        require(block.timestamp <= order.deadline, "TokenSwapper: Order expired");
        require(
            block.timestamp >= order.deadline - mevProtectionDelay,
            "TokenSwapper: MEV protection active"
        );
        
        // Find optimal route
        SwapRoute memory optimalRoute = _findOptimalRoute(
            order.tokenIn,
            order.tokenOut,
            order.amountIn
        );
        
        require(
            optimalRoute.expectedOutput >= order.minAmountOut,
            "TokenSwapper: Insufficient output amount"
        );
        
        // Execute swap
        uint256 actualOutput = _executeSwapOnDEX(
            optimalRoute,
            order.amountIn,
            order.minAmountOut
        );
        
        require(actualOutput >= order.minAmountOut, "TokenSwapper: Slippage exceeded");
        
        // Calculate and deduct protocol fee
        uint256 protocolFeeAmount = (actualOutput * protocolFee) / 10000;
        uint256 userOutput = actualOutput - protocolFeeAmount;
        
        // Transfer tokens
        IERC20(order.tokenOut).safeTransfer(order.user, userOutput);
        
        if (protocolFeeAmount > 0) {
            IERC20(order.tokenOut).safeTransfer(treasury, protocolFeeAmount);
        }
        
        // Mark order as executed
        order.executed = true;
        order.executedAmount = actualOutput;
        
        emit SwapExecuted(
            orderId,
            order.user,
            order.tokenIn,
            order.tokenOut,
            order.amountIn,
            actualOutput,
            optimalRoute.dex
        );
    }

    /**
     * @dev Instant swap with immediate execution
     */
    function instantSwap(
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut,
        uint256 maxSlippageBps
    ) external nonReentrant returns (uint256 amountOut) {
        require(amountIn > 0, "TokenSwapper: Invalid amount");
        require(maxSlippageBps <= maxSlippage, "TokenSwapper: Slippage too high");
        
        // Transfer tokens
        IERC20(tokenIn).safeTransferFrom(msg.sender, address(this), amountIn);
        
        // Find optimal route
        SwapRoute memory optimalRoute = _findOptimalRoute(tokenIn, tokenOut, amountIn);
        
        require(
            optimalRoute.expectedOutput >= minAmountOut,
            "TokenSwapper: Insufficient output amount"
        );
        
        // Execute swap
        amountOut = _executeSwapOnDEX(optimalRoute, amountIn, minAmountOut);
        
        require(amountOut >= minAmountOut, "TokenSwapper: Slippage exceeded");
        
        // Calculate protocol fee
        uint256 protocolFeeAmount = (amountOut * protocolFee) / 10000;
        uint256 userOutput = amountOut - protocolFeeAmount;
        
        // Transfer tokens
        IERC20(tokenOut).safeTransfer(msg.sender, userOutput);
        
        if (protocolFeeAmount > 0) {
            IERC20(tokenOut).safeTransfer(treasury, protocolFeeAmount);
        }
        
        emit SwapExecuted(
            bytes32(0),
            msg.sender,
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            optimalRoute.dex
        );
        
        return userOutput;
    }

    /**
     * @dev Get swap quote from all DEXs
     */
    function getSwapQuote(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) external view returns (SwapRoute[] memory routes) {
        routes = new SwapRoute[](supportedDEXs.length);
        uint256 validRoutes = 0;
        
        for (uint256 i = 0; i < supportedDEXs.length; i++) {
            address dex = supportedDEXs[i];
            
            if (dexConfigs[dex].active) {
                SwapRoute memory route = _getRouteQuote(dex, tokenIn, tokenOut, amountIn);
                
                if (route.expectedOutput > 0) {
                    routes[validRoutes] = route;
                    validRoutes++;
                }
            }
        }
        
        // Resize array to remove empty slots
        assembly {
            mstore(routes, validRoutes)
        }
        
        return routes;
    }

    /**
     * @dev Find optimal route across all DEXs
     */
    function _findOptimalRoute(
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal view returns (SwapRoute memory optimalRoute) {
        uint256 bestOutput = 0;
        
        for (uint256 i = 0; i < supportedDEXs.length; i++) {
            address dex = supportedDEXs[i];
            
            if (dexConfigs[dex].active) {
                SwapRoute memory route = _getRouteQuote(dex, tokenIn, tokenOut, amountIn);
                
                // Consider gas costs in optimization
                uint256 netOutput = route.expectedOutput;
                if (route.gasEstimate > 0) {
                    uint256 gasCost = route.gasEstimate * tx.gasprice;
                    uint256 gasCostInToken = _convertToToken(tokenOut, gasCost);
                    
                    if (netOutput > gasCostInToken) {
                        netOutput -= gasCostInToken;
                    } else {
                        continue;
                    }
                }
                
                if (netOutput > bestOutput) {
                    bestOutput = netOutput;
                    optimalRoute = route;
                }
            }
        }
        
        require(bestOutput > 0, "TokenSwapper: No valid route found");
        
        emit RouteOptimized(tokenIn, tokenOut, optimalRoute.dex, optimalRoute.expectedOutput);
        
        return optimalRoute;
    }

    /**
     * @dev Get quote from specific DEX
     */
    function _getRouteQuote(
        address dex,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal view returns (SwapRoute memory route) {
        DEXConfig memory config = dexConfigs[dex];
        
        // Build path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;
        
        try this.getAmountsOut(config.router, amountIn, path) returns (
            uint256[] memory amounts
        ) {
            uint256 expectedOutput = amounts[amounts.length - 1];
            
            // Apply DEX fee
            uint256 feeAmount = (expectedOutput * config.fee) / 10000;
            expectedOutput -= feeAmount;
            
            route = SwapRoute({
                dex: dex,
                path: path,
                expectedOutput: expectedOutput,
                gasEstimate: config.gasOverhead,
                fee: config.fee
            });
        } catch {
            // Route not available
            route = SwapRoute({
                dex: dex,
                path: path,
                expectedOutput: 0,
                gasEstimate: 0,
                fee: 0
            });
        }
        
        return route;
    }

    /**
     * @dev Execute swap on specific DEX
     */
    function _executeSwapOnDEX(
        SwapRoute memory route,
        uint256 amountIn,
        uint256 minAmountOut
    ) internal returns (uint256 amountOut) {
        DEXConfig memory config = dexConfigs[route.dex];
        
        // Approve router
        IERC20(route.path[0]).safeApprove(config.router, amountIn);
        
        // Execute swap (simplified - in production, call actual DEX router)
        (bool success, bytes memory result) = config.router.call(
            abi.encodeWithSignature(
                "swapExactTokensForTokens(uint256,uint256,address[],address,uint256)",
                amountIn,
                minAmountOut,
                route.path,
                address(this),
                block.timestamp + 300
            )
        );
        
        require(success, "TokenSwapper: Swap failed");
        
        if (result.length > 0) {
            uint256[] memory amounts = abi.decode(result, (uint256[]));
            amountOut = amounts[amounts.length - 1];
        } else {
            // Fallback: use expected output
            amountOut = route.expectedOutput;
        }
        
        return amountOut;
    }

    /**
     * @dev External function for getAmountsOut (for try-catch)
     */
    function getAmountsOut(
        address router,
        uint256 amountIn,
        address[] memory path
    ) external view returns (uint256[] memory amounts) {
        (bool success, bytes memory result) = router.staticcall(
            abi.encodeWithSignature(
                "getAmountsOut(uint256,address[])",
                amountIn,
                path
            )
        );
        
        require(success, "TokenSwapper: Quote failed");
        amounts = abi.decode(result, (uint256[]));
        
        return amounts;
    }

    /**
     * @dev Convert ETH amount to token amount using price feed
     */
    function _convertToToken(address token, uint256 ethAmount) 
        internal 
        view 
        returns (uint256) 
    {
        if (address(priceFeeds[token]) == address(0)) {
            return 0;
        }
        
        try priceFeeds[token].latestRoundData() returns (
            uint80, int256 price, uint256, uint256, uint80
        ) {
            if (price > 0) {
                uint8 decimals = priceFeeds[token].decimals();
                return (ethAmount * (10 ** decimals)) / uint256(price);
            }
        } catch {}
        
        return 0;
    }

    /**
     * @dev Cancel swap order
     */
    function cancelOrder(bytes32 orderId) external nonReentrant {
        SwapOrder storage order = swapOrders[orderId];
        require(order.user == msg.sender, "TokenSwapper: Not order owner");
        require(!order.executed, "TokenSwapper: Order already executed");
        
        // Return tokens to user
        IERC20(order.tokenIn).safeTransfer(order.user, order.amountIn);
        
        // Mark as executed to prevent reuse
        order.executed = true;
    }

    /**
     * @dev Add price feed for token
     */
    function addPriceFeed(address token, address priceFeed) external onlyOwner {
        priceFeeds[token] = AggregatorV3Interface(priceFeed);
    }

    /**
     * @dev Update protocol fee
     */
    function updateProtocolFee(uint256 newFee) external onlyOwner {
        require(newFee <= 100, "TokenSwapper: Fee too high"); // Max 1%
        protocolFee = newFee;
    }

    /**
     * @dev Update max slippage
     */
    function updateMaxSlippage(uint256 newMaxSlippage) external onlyOwner {
        require(newMaxSlippage <= 1000, "TokenSwapper: Slippage too high"); // Max 10%
        maxSlippage = newMaxSlippage;
    }

    /**
     * @dev Emergency withdraw function
     */
    function emergencyWithdraw(address token, uint256 amount) external onlyOwner {
        IERC20(token).safeTransfer(owner(), amount);
    }
}
