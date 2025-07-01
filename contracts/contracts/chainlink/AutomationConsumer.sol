// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {AutomationCompatibleInterface} from "@chainlink/contracts/src/v0.8/automation/AutomationCompatible.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title AutomationConsumer
 * @dev Chainlink Automation consumer for ArbOS periodic tasks
 * @notice Handles automated portfolio rebalancing and arbitrage monitoring
 */
contract AutomationConsumer is AutomationCompatibleInterface, Ownable {
    
    // Automation intervals
    uint256 public constant REBALANCE_INTERVAL = 24 hours;
    uint256 public constant ARBITRAGE_CHECK_INTERVAL = 5 minutes;
    uint256 public constant SECURITY_SCAN_INTERVAL = 1 hours;
    
    // State variables
    address public arbOSCore;
    address public portfolioManager;
    address public arbitrageEngine;
    address public securityModule;
    
    uint256 public lastRebalanceTime;
    uint256 public lastArbitrageCheck;
    uint256 public lastSecurityScan;
    
    bool public rebalanceEnabled = true;
    bool public arbitrageMonitoringEnabled = true;
    bool public securityScanEnabled = true;
    
    // Counters
    uint256 public totalRebalances;
    uint256 public totalArbitrageChecks;
    uint256 public totalSecurityScans;
    
    // Events
    event UpkeepPerformed(string indexed taskType, uint256 timestamp);
    event AutomationConfigUpdated(string parameter, bool value);
    event ContractAddressUpdated(string indexed contractType, address newAddress);

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "AutomationConsumer: Only core");
        _;
    }

    constructor(
        address _arbOSCore,
        address _portfolioManager,
        address _arbitrageEngine,
        address _securityModule
    ) {
        arbOSCore = _arbOSCore;
        portfolioManager = _portfolioManager;
        arbitrageEngine = _arbitrageEngine;
        securityModule = _securityModule;
        
        lastRebalanceTime = block.timestamp;
        lastArbitrageCheck = block.timestamp;
        lastSecurityScan = block.timestamp;
    }

    /**
     * @dev Check if upkeep is needed
     * @param /* checkData */ Unused parameter
     * @return upkeepNeeded Whether upkeep is needed
     * @return performData Data to pass to performUpkeep
     */
    function checkUpkeep(bytes calldata /* checkData */)
        external
        view
        override
        returns (bool upkeepNeeded, bytes memory performData)
    {
        // Check if portfolio rebalancing is needed
        bool needsRebalance = rebalanceEnabled && 
            (block.timestamp - lastRebalanceTime) >= REBALANCE_INTERVAL &&
            _shouldRebalancePortfolio();
        
        // Check if arbitrage monitoring is needed
        bool needsArbitrageCheck = arbitrageMonitoringEnabled &&
            (block.timestamp - lastArbitrageCheck) >= ARBITRAGE_CHECK_INTERVAL;
        
        // Check if security scan is needed
        bool needsSecurityScan = securityScanEnabled &&
            (block.timestamp - lastSecurityScan) >= SECURITY_SCAN_INTERVAL;
        
        upkeepNeeded = needsRebalance || needsArbitrageCheck || needsSecurityScan;
        
        if (upkeepNeeded) {
            // Encode which tasks need to be performed
            performData = abi.encode(needsRebalance, needsArbitrageCheck, needsSecurityScan);
        }
        
        return (upkeepNeeded, performData);
    }

    /**
     * @dev Perform the upkeep
     * @param performData Data from checkUpkeep
     */
    function performUpkeep(bytes calldata performData) external override {
        // Decode which tasks to perform
        (bool needsRebalance, bool needsArbitrageCheck, bool needsSecurityScan) = 
            abi.decode(performData, (bool, bool, bool));
        
        // Perform portfolio rebalancing
        if (needsRebalance && _shouldRebalancePortfolio()) {
            _performRebalancing();
            lastRebalanceTime = block.timestamp;
            totalRebalances++;
            emit UpkeepPerformed("REBALANCE", block.timestamp);
        }
        
        // Perform arbitrage monitoring
        if (needsArbitrageCheck) {
            _performArbitrageMonitoring();
            lastArbitrageCheck = block.timestamp;
            totalArbitrageChecks++;
            emit UpkeepPerformed("ARBITRAGE_CHECK", block.timestamp);
        }
        
        // Perform security scan
        if (needsSecurityScan) {
            _performSecurityScan();
            lastSecurityScan = block.timestamp;
            totalSecurityScans++;
            emit UpkeepPerformed("SECURITY_SCAN", block.timestamp);
        }
    }

    /**
     * @dev Check if portfolio should be rebalanced
     */
    function _shouldRebalancePortfolio() internal view returns (bool) {
        if (portfolioManager == address(0)) return false;
        
        try this.shouldRebalanceExternal() returns (bool shouldRebalance) {
            return shouldRebalance;
        } catch {
            return false;
        }
    }

    /**
     * @dev External function to check rebalancing (for try-catch)
     */
    function shouldRebalanceExternal() external view returns (bool) {
        (bool success, bytes memory data) = portfolioManager.staticcall(
            abi.encodeWithSignature("shouldRebalance()")
        );
        
        if (success && data.length > 0) {
            return abi.decode(data, (bool));
        }
        
        return false;
    }

    /**
     * @dev Perform portfolio rebalancing
     */
    function _performRebalancing() internal {
        if (portfolioManager == address(0)) return;
        
        try this.rebalanceExternal() {
            // Rebalancing succeeded
        } catch {
            // Rebalancing failed, continue with other tasks
        }
    }

    /**
     * @dev External function to perform rebalancing (for try-catch)
     */
    function rebalanceExternal() external {
        require(msg.sender == address(this), "AutomationConsumer: Internal only");
        
        (bool success, ) = portfolioManager.call(
            abi.encodeWithSignature("rebalancePortfolio()")
        );
        
        require(success, "AutomationConsumer: Rebalancing failed");
    }

    /**
     * @dev Perform arbitrage opportunity monitoring
     */
    function _performArbitrageMonitoring() internal {
        if (arbitrageEngine == address(0)) return;
        
        try this.updateArbitragePricesExternal() {
            // Price update succeeded
        } catch {
            // Price update failed, continue
        }
    }

    /**
     * @dev External function to update arbitrage prices (for try-catch)
     */
    function updateArbitragePricesExternal() external {
        require(msg.sender == address(this), "AutomationConsumer: Internal only");
        
        // Update price feeds in arbitrage engine
        (bool success, ) = arbitrageEngine.call(
            abi.encodeWithSignature("updateAllPrices()")
        );
        
        require(success, "AutomationConsumer: Price update failed");
    }

    /**
     * @dev Perform security system scan
     */
    function _performSecurityScan() internal {
        if (securityModule == address(0)) return;
        
        try this.performSecurityScanExternal() {
            // Security scan succeeded
        } catch {
            // Security scan failed, continue
        }
    }

    /**
     * @dev External function to perform security scan (for try-catch)
     */
    function performSecurityScanExternal() external {
        require(msg.sender == address(this), "AutomationConsumer: Internal only");
        
        // Trigger security module scan
        (bool success, ) = securityModule.call(
            abi.encodeWithSignature("performRoutineScan()")
        );
        
        require(success, "AutomationConsumer: Security scan failed");
    }

    /**
     * @dev Enable/disable rebalancing automation
     */
    function setRebalanceEnabled(bool enabled) external onlyOwner {
        rebalanceEnabled = enabled;
        emit AutomationConfigUpdated("rebalanceEnabled", enabled);
    }

    /**
     * @dev Enable/disable arbitrage monitoring
     */
    function setArbitrageMonitoringEnabled(bool enabled) external onlyOwner {
        arbitrageMonitoringEnabled = enabled;
        emit AutomationConfigUpdated("arbitrageMonitoringEnabled", enabled);
    }

    /**
     * @dev Enable/disable security scanning
     */
    function setSecurityScanEnabled(bool enabled) external onlyOwner {
        securityScanEnabled = enabled;
        emit AutomationConfigUpdated("securityScanEnabled", enabled);
    }

    /**
     * @dev Update contract addresses
     */
    function updateContractAddresses(
        address _arbOSCore,
        address _portfolioManager,
        address _arbitrageEngine,
        address _securityModule
    ) external onlyOwner {
        if (_arbOSCore != address(0)) {
            arbOSCore = _arbOSCore;
            emit ContractAddressUpdated("arbOSCore", _arbOSCore);
        }
        
        if (_portfolioManager != address(0)) {
            portfolioManager = _portfolioManager;
            emit ContractAddressUpdated("portfolioManager", _portfolioManager);
        }
        
        if (_arbitrageEngine != address(0)) {
            arbitrageEngine = _arbitrageEngine;
            emit ContractAddressUpdated("arbitrageEngine", _arbitrageEngine);
        }
        
        if (_securityModule != address(0)) {
            securityModule = _securityModule;
            emit ContractAddressUpdated("securityModule", _securityModule);
        }
    }

    /**
     * @dev Get automation status
     */
    function getAutomationStatus() external view returns (
        bool rebalanceActive,
        bool arbitrageActive,
        bool securityActive,
        uint256 nextRebalance,
        uint256 nextArbitrageCheck,
        uint256 nextSecurityScan
    ) {
        rebalanceActive = rebalanceEnabled;
        arbitrageActive = arbitrageMonitoringEnabled;
        securityActive = securityScanEnabled;
        
        nextRebalance = lastRebalanceTime + REBALANCE_INTERVAL;
        nextArbitrageCheck = lastArbitrageCheck + ARBITRAGE_CHECK_INTERVAL;
        nextSecurityScan = lastSecurityScan + SECURITY_SCAN_INTERVAL;
    }

    /**
     * @dev Get automation statistics
     */
    function getAutomationStats() external view returns (
        uint256 rebalances,
        uint256 arbitrageChecks,
        uint256 securityScans
    ) {
        return (totalRebalances, totalArbitrageChecks, totalSecurityScans);
    }

    /**
     * @dev Manual trigger for emergency rebalancing
     */
    function manualRebalance() external onlyOwner {
        _performRebalancing();
        lastRebalanceTime = block.timestamp;
        totalRebalances++;
        emit UpkeepPerformed("MANUAL_REBALANCE", block.timestamp);
    }
}
