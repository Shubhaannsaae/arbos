// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {Pausable} from "@openzeppelin/contracts/security/Pausable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

import {ISecurity} from "../interfaces/ISecurity.sol";

/**
 * @title SecurityModule
 * @dev Advanced security and risk management module for ArbOS
 */
contract SecurityModule is ISecurity, Ownable, Pausable, ReentrancyGuard {
    using EnumerableSet for EnumerableSet.UintSet;
    using EnumerableSet for EnumerableSet.AddressSet;

    // Constants
    uint256 public constant MAX_RISK_SCORE = 1000;
    uint256 public constant CRITICAL_RISK_THRESHOLD = 800;
    uint256 public constant HIGH_RISK_THRESHOLD = 600;
    uint256 public constant MEDIUM_RISK_THRESHOLD = 400;

    // State variables
    mapping(uint256 => SecurityAlert) public securityAlerts;
    mapping(address => uint256) public contractSecurityScores;
    mapping(address => uint256) public userSecurityScores;
    mapping(address => bool) public blacklistedAddresses;
    mapping(address => uint256) public lastTransactionTime;
    mapping(address => uint256) public transactionCount;
    mapping(bytes32 => TransactionRisk) public transactionRisks;

    EnumerableSet.UintSet private activeAlerts;
    EnumerableSet.AddressSet private monitoredContracts;

    uint256 public nextAlertId = 1;
    uint256 public emergencyPauseThreshold = CRITICAL_RISK_THRESHOLD;
    address public arbOSCore;

    // Risk parameters
    uint256 public maxTransactionsPerHour = 100;
    uint256 public maxTransactionValue = 1000 ether;
    uint256 public suspiciousPatternThreshold = 5;

    // Events from interface already declared

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "SecurityModule: Only core");
        _;
    }

    modifier notBlacklisted(address addr) {
        require(!blacklistedAddresses[addr], "SecurityModule: Address blacklisted");
        _;
    }

    constructor(address _arbOSCore) {
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Assess transaction risk before execution
     */
    function assessTransactionRisk(
        address from,
        address to,
        uint256 value,
        bytes calldata data
    ) 
        external 
        view 
        override 
        returns (TransactionRisk memory) 
    {
        uint256 riskScore = 0;
        string memory riskReason = "";
        
        // Check blacklisted addresses
        if (blacklistedAddresses[from] || blacklistedAddresses[to]) {
            riskScore = MAX_RISK_SCORE;
            riskReason = "Blacklisted address";
        } else {
            // Calculate risk based on multiple factors
            (riskScore, riskReason) = _calculateTransactionRisk(from, to, value, data);
        }

        bool flagged = riskScore >= MEDIUM_RISK_THRESHOLD;

        return TransactionRisk({
            from: from,
            to: to,
            value: value,
            riskScore: riskScore,
            flagged: flagged,
            riskReason: riskReason
        });
    }

    /**
     * @dev Raise security alert
     */
    function raiseSecurityAlert(
        RiskLevel riskLevel,
        string calldata description,
        address affectedContract
    ) external override onlyCore {
        uint256 alertId = nextAlertId++;
        
        securityAlerts[alertId] = SecurityAlert({
            alertId: alertId,
            riskLevel: riskLevel,
            description: description,
            affectedContract: affectedContract,
            timestamp: block.timestamp,
            resolved: false
        });

        activeAlerts.add(alertId);

        // Auto-pause if critical risk
        if (riskLevel == RiskLevel.CRITICAL) {
            _pause();
        }

        emit SecurityAlertRaised(alertId, riskLevel, affectedContract);
    }

    /**
     * @dev Resolve security alert
     */
    function resolveAlert(uint256 alertId) external override onlyOwner {
        require(activeAlerts.contains(alertId), "SecurityModule: Alert not found");
        
        securityAlerts[alertId].resolved = true;
        activeAlerts.remove(alertId);
    }

    /**
     * @dev Check if contract is secure
     */
    function isContractSecure(address contract_) 
        external 
        view 
        override 
        returns (bool) 
    {
        if (blacklistedAddresses[contract_]) {
            return false;
        }

        uint256 securityScore = contractSecurityScores[contract_];
        return securityScore < HIGH_RISK_THRESHOLD;
    }

    /**
     * @dev Get security score for user
     */
    function getSecurityScore(address user) 
        external 
        view 
        override 
        returns (uint256) 
    {
        return userSecurityScores[user];
    }

    /**
     * @dev Emergency pause function
     */
    function emergencyPause() external override onlyOwner {
        _pause();
        emit AnomalyDetected(address(this), "EMERGENCY_PAUSE", CRITICAL_RISK_THRESHOLD);
    }

    /**
     * @dev Emergency unpause function
     */
    function emergencyUnpause() external override onlyOwner {
        _unpause();
    }

    /**
     * @dev Monitor transaction for suspicious patterns
     */
    function monitorTransaction(
        address from,
        address to,
        uint256 value,
        bytes calldata data
    ) external onlyCore {
        bytes32 txHash = keccak256(abi.encodePacked(from, to, value, block.timestamp));
        
        TransactionRisk memory risk = assessTransactionRisk(from, to, value, data);
        transactionRisks[txHash] = risk;

        if (risk.flagged) {
            emit TransactionFlagged(txHash, from, risk.riskScore);
            
            // Update user security score
            _updateUserSecurityScore(from, risk.riskScore);
        }

        // Update transaction tracking
        _updateTransactionTracking(from);
        
        // Check for anomalies
        _checkForAnomalies(from, to, value);
    }

    /**
     * @dev Add address to blacklist
     */
    function addToBlacklist(address addr) external onlyOwner {
        blacklistedAddresses[addr] = true;
        userSecurityScores[addr] = MAX_RISK_SCORE;
    }

    /**
     * @dev Remove address from blacklist
     */
    function removeFromBlacklist(address addr) external onlyOwner {
        blacklistedAddresses[addr] = false;
        userSecurityScores[addr] = 0;
    }

    /**
     * @dev Set contract security score
     */
    function setContractSecurityScore(address contract_, uint256 score) 
        external 
        onlyOwner 
    {
        require(score <= MAX_RISK_SCORE, "SecurityModule: Invalid score");
        contractSecurityScores[contract_] = score;
        
        if (!monitoredContracts.contains(contract_)) {
            monitoredContracts.add(contract_);
        }
    }

    /**
     * @dev Update security parameters
     */
    function updateSecurityParameters(
        uint256 _maxTransactionsPerHour,
        uint256 _maxTransactionValue,
        uint256 _suspiciousPatternThreshold
    ) external onlyOwner {
        maxTransactionsPerHour = _maxTransactionsPerHour;
        maxTransactionValue = _maxTransactionValue;
        suspiciousPatternThreshold = _suspiciousPatternThreshold;
    }

    /**
     * @dev Calculate transaction risk score
     */
    function _calculateTransactionRisk(
        address from,
        address to,
        uint256 value,
        bytes calldata data
    ) internal view returns (uint256 riskScore, string memory riskReason) {
        riskScore = 0;
        
        // High value transaction risk
        if (value > maxTransactionValue) {
            riskScore += 200;
            riskReason = "High value transaction";
        }

        // Frequency risk
        if (transactionCount[from] > maxTransactionsPerHour) {
            riskScore += 150;
            riskReason = string(abi.encodePacked(riskReason, " High frequency"));
        }

        // New contract interaction risk
        if (to.code.length > 0 && contractSecurityScores[to] == 0) {
            riskScore += 100;
            riskReason = string(abi.encodePacked(riskReason, " Unknown contract"));
        }

        // Known risky contract
        if (contractSecurityScores[to] > HIGH_RISK_THRESHOLD) {
            riskScore += contractSecurityScores[to] / 2;
            riskReason = string(abi.encodePacked(riskReason, " Risky contract"));
        }

        // Complex transaction data
        if (data.length > 1000) {
            riskScore += 50;
            riskReason = string(abi.encodePacked(riskReason, " Complex transaction"));
        }

        // User history risk
        uint256 userRisk = userSecurityScores[from];
        if (userRisk > 0) {
            riskScore += userRisk / 4;
            riskReason = string(abi.encodePacked(riskReason, " User history"));
        }

        // Cap at maximum
        if (riskScore > MAX_RISK_SCORE) {
            riskScore = MAX_RISK_SCORE;
        }

        return (riskScore, riskReason);
    }

    /**
     * @dev Update user security score based on behavior
     */
    function _updateUserSecurityScore(address user, uint256 transactionRisk) internal {
        uint256 currentScore = userSecurityScores[user];
        
        // Increase score for risky behavior
        if (transactionRisk >= HIGH_RISK_THRESHOLD) {
            currentScore += 50;
        } else if (transactionRisk >= MEDIUM_RISK_THRESHOLD) {
            currentScore += 20;
        }

        // Decay score over time for good behavior
        uint256 timeSinceLastTx = block.timestamp - lastTransactionTime[user];
        if (timeSinceLastTx > 1 days && currentScore > 0) {
            currentScore = currentScore > 10 ? currentScore - 10 : 0;
        }

        userSecurityScores[user] = currentScore > MAX_RISK_SCORE ? MAX_RISK_SCORE : currentScore;
    }

    /**
     * @dev Update transaction tracking for user
     */
    function _updateTransactionTracking(address user) internal {
        uint256 currentHour = block.timestamp / 3600;
        uint256 lastTxHour = lastTransactionTime[user] / 3600;
        
        if (currentHour != lastTxHour) {
            transactionCount[user] = 1;
        } else {
            transactionCount[user]++;
        }
        
        lastTransactionTime[user] = block.timestamp;
    }

    /**
     * @dev Check for anomalous patterns
     */
    function _checkForAnomalies(address from, address to, uint256 value) internal {
        // Check for rapid fire transactions
        if (transactionCount[from] > suspiciousPatternThreshold) {
            emit AnomalyDetected(from, "RAPID_TRANSACTIONS", transactionCount[from]);
        }

        // Check for unusual value patterns
        if (value > 0 && value % 1 ether == 0 && value > 10 ether) {
            emit AnomalyDetected(from, "ROUND_NUMBER_TRANSACTION", value);
        }

        // Check for contract interaction patterns
        if (to.code.length > 0 && contractSecurityScores[to] > MEDIUM_RISK_THRESHOLD) {
            emit AnomalyDetected(to, "RISKY_CONTRACT_INTERACTION", contractSecurityScores[to]);
        }
    }

    /**
     * @dev Get active alerts
     */
    function getActiveAlerts() external view returns (uint256[] memory) {
        uint256 length = activeAlerts.length();
        uint256[] memory alerts = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            alerts[i] = activeAlerts.at(i);
        }
        
        return alerts;
    }

    /**
     * @dev Get monitored contracts
     */
    function getMonitoredContracts() external view returns (address[] memory) {
        uint256 length = monitoredContracts.length();
        address[] memory contracts = new address[](length);
        
        for (uint256 i = 0; i < length; i++) {
            contracts[i] = monitoredContracts.at(i);
        }
        
        return contracts;
    }

    /**
     * @dev Update ArbOS core address
     */
    function updateArbOSCore(address _arbOSCore) external onlyOwner {
        arbOSCore = _arbOSCore;
    }
}
