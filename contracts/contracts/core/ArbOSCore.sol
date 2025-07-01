// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {Pausable} from "@openzeppelin/contracts/security/Pausable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

import {IArbitrage} from "../interfaces/IArbitrage.sol";
import {IPortfolio} from "../interfaces/IPortfolio.sol";
import {ISecurity} from "../interfaces/ISecurity.sol";

/**
 * @title ArbOSCore
 * @dev Core contract for the Arbitrage Operating System
 * @notice Manages core system operations, integrations, and governance
 */
contract ArbOSCore is Ownable, Pausable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // Version
    string public constant VERSION = "1.0.0";
    
    // Core modules
    address public arbitrageEngine;
    address public portfolioManager;
    address public securityModule;
    address public chainlinkConsumer;
    
    // System parameters
    uint256 public maxSlippage = 300; // 3% in basis points
    uint256 public minProfitThreshold = 10; // 0.1% in basis points
    uint256 public treasuryFee = 100; // 1% in basis points
    address public treasury;
    
    // User data
    mapping(address => bool) public authorizedOperators;
    mapping(address => uint256) public userNonces;
    mapping(address => bool) public whitelistedTokens;
    
    // Events
    event ModuleUpdated(string indexed module, address indexed newAddress);
    event OperatorStatusChanged(address indexed operator, bool authorized);
    event ParameterUpdated(string indexed parameter, uint256 newValue);
    event TokenWhitelisted(address indexed token, bool status);
    event EmergencyAction(string action, address indexed executor);

    // Modifiers
    modifier onlyAuthorized() {
        require(
            authorizedOperators[msg.sender] || msg.sender == owner(),
            "ArbOSCore: Not authorized"
        );
        _;
    }

    modifier onlyWhitelistedToken(address token) {
        require(whitelistedTokens[token], "ArbOSCore: Token not whitelisted");
        _;
    }

    constructor(
        address _treasury,
        address _arbitrageEngine,
        address _portfolioManager,
        address _securityModule
    ) {
        require(_treasury != address(0), "ArbOSCore: Invalid treasury");
        
        treasury = _treasury;
        arbitrageEngine = _arbitrageEngine;
        portfolioManager = _portfolioManager;
        securityModule = _securityModule;
        
        // Initialize authorized operators
        authorizedOperators[msg.sender] = true;
    }

    /**
     * @dev Execute arbitrage operation through the system
     * @param opportunityId The ID of the arbitrage opportunity
     */
    function executeArbitrage(uint256 opportunityId) 
        external 
        onlyAuthorized 
        whenNotPaused 
        nonReentrant 
    {
        require(arbitrageEngine != address(0), "ArbOSCore: Arbitrage engine not set");
        
        // Security check
        if (securityModule != address(0)) {
            require(
                ISecurity(securityModule).isContractSecure(arbitrageEngine),
                "ArbOSCore: Arbitrage engine not secure"
            );
        }
        
        // Execute arbitrage
        bool success = IArbitrage(arbitrageEngine).executeArbitrage(opportunityId);
        require(success, "ArbOSCore: Arbitrage execution failed");
        
        // Increment nonce
        userNonces[msg.sender]++;
    }

    /**
     * @dev Trigger portfolio rebalancing
     */
    function rebalancePortfolio() 
        external 
        onlyAuthorized 
        whenNotPaused 
        nonReentrant 
    {
        require(portfolioManager != address(0), "ArbOSCore: Portfolio manager not set");
        
        // Check if rebalancing is needed
        require(
            IPortfolio(portfolioManager).shouldRebalance(),
            "ArbOSCore: Rebalancing not needed"
        );
        
        // Execute rebalancing
        IPortfolio(portfolioManager).rebalancePortfolio();
    }

    /**
     * @dev Update system module address
     * @param module Name of the module
     * @param newAddress New module address
     */
    function updateModule(string calldata module, address newAddress) 
        external 
        onlyOwner 
    {
        require(newAddress != address(0), "ArbOSCore: Invalid address");
        
        bytes32 moduleHash = keccak256(abi.encodePacked(module));
        
        if (moduleHash == keccak256(abi.encodePacked("arbitrage"))) {
            arbitrageEngine = newAddress;
        } else if (moduleHash == keccak256(abi.encodePacked("portfolio"))) {
            portfolioManager = newAddress;
        } else if (moduleHash == keccak256(abi.encodePacked("security"))) {
            securityModule = newAddress;
        } else if (moduleHash == keccak256(abi.encodePacked("chainlink"))) {
            chainlinkConsumer = newAddress;
        } else {
            revert("ArbOSCore: Unknown module");
        }
        
        emit ModuleUpdated(module, newAddress);
    }

    /**
     * @dev Set operator authorization status
     * @param operator Address of the operator
     * @param authorized Authorization status
     */
    function setOperatorStatus(address operator, bool authorized) 
        external 
        onlyOwner 
    {
        require(operator != address(0), "ArbOSCore: Invalid operator");
        authorizedOperators[operator] = authorized;
        emit OperatorStatusChanged(operator, authorized);
    }

    /**
     * @dev Update system parameters
     * @param parameter Parameter name
     * @param value New parameter value
     */
    function updateParameter(string calldata parameter, uint256 value) 
        external 
        onlyOwner 
    {
        bytes32 paramHash = keccak256(abi.encodePacked(parameter));
        
        if (paramHash == keccak256(abi.encodePacked("maxSlippage"))) {
            require(value <= 1000, "ArbOSCore: Max slippage too high"); // Max 10%
            maxSlippage = value;
        } else if (paramHash == keccak256(abi.encodePacked("minProfitThreshold"))) {
            require(value >= 1, "ArbOSCore: Min profit threshold too low");
            minProfitThreshold = value;
        } else if (paramHash == keccak256(abi.encodePacked("treasuryFee"))) {
            require(value <= 500, "ArbOSCore: Treasury fee too high"); // Max 5%
            treasuryFee = value;
        } else {
            revert("ArbOSCore: Unknown parameter");
        }
        
        emit ParameterUpdated(parameter, value);
    }

    /**
     * @dev Whitelist or blacklist a token
     * @param token Token address
     * @param status Whitelist status
     */
    function setTokenWhitelist(address token, bool status) 
        external 
        onlyOwner 
    {
        require(token != address(0), "ArbOSCore: Invalid token");
        whitelistedTokens[token] = status;
        emit TokenWhitelisted(token, status);
    }

    /**
     * @dev Emergency pause function
     */
    function emergencyPause() external onlyOwner {
        _pause();
        emit EmergencyAction("PAUSE", msg.sender);
    }

    /**
     * @dev Emergency unpause function
     */
    function emergencyUnpause() external onlyOwner {
        _unpause();
        emit EmergencyAction("UNPAUSE", msg.sender);
    }

    /**
     * @dev Withdraw stuck tokens (emergency function)
     * @param token Token address
     * @param amount Amount to withdraw
     */
    function emergencyWithdraw(address token, uint256 amount) 
        external 
        onlyOwner 
    {
        if (token == address(0)) {
            payable(treasury).transfer(amount);
        } else {
            IERC20(token).safeTransfer(treasury, amount);
        }
        emit EmergencyAction("WITHDRAW", msg.sender);
    }

    /**
     * @dev Get system status
     */
    function getSystemStatus() 
        external 
        view 
        returns (
            bool isActive,
            uint256 totalOperators,
            uint256 whitelistedTokenCount,
            address[] memory modules
        ) 
    {
        isActive = !paused();
        
        // Count operators (simplified - in production use EnumerableSet)
        totalOperators = authorizedOperators[owner()] ? 1 : 0;
        
        // Count whitelisted tokens (simplified)
        whitelistedTokenCount = 0;
        
        modules = new address[](4);
        modules[0] = arbitrageEngine;
        modules[1] = portfolioManager;
        modules[2] = securityModule;
        modules[3] = chainlinkConsumer;
    }

    /**
     * @dev Check if user is authorized
     * @param user User address
     */
    function isAuthorized(address user) external view returns (bool) {
        return authorizedOperators[user] || user == owner();
    }

    /**
     * @dev Get user nonce
     * @param user User address
     */
    function getUserNonce(address user) external view returns (uint256) {
        return userNonces[user];
    }

    // Receive ETH
    receive() external payable {}
}
