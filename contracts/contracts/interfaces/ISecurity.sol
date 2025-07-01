// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ISecurity
 * @dev Interface for security and risk management
 */
interface ISecurity {
    enum RiskLevel {
        LOW,
        MEDIUM,
        HIGH,
        CRITICAL
    }

    struct SecurityAlert {
        uint256 alertId;
        RiskLevel riskLevel;
        string description;
        address affectedContract;
        uint256 timestamp;
        bool resolved;
    }

    struct TransactionRisk {
        address from;
        address to;
        uint256 value;
        uint256 riskScore;
        bool flagged;
        string riskReason;
    }

    event SecurityAlertRaised(
        uint256 indexed alertId,
        RiskLevel indexed riskLevel,
        address indexed affectedContract
    );

    event TransactionFlagged(
        bytes32 indexed txHash,
        address indexed from,
        uint256 riskScore
    );

    event AnomalyDetected(
        address indexed contract_,
        string anomalyType,
        uint256 severity
    );

    function assessTransactionRisk(
        address from,
        address to,
        uint256 value,
        bytes calldata data
    ) external view returns (TransactionRisk memory);

    function raiseSecurityAlert(
        RiskLevel riskLevel,
        string calldata description,
        address affectedContract
    ) external;

    function resolveAlert(uint256 alertId) external;
    
    function isContractSecure(address contract_) external view returns (bool);
    
    function getSecurityScore(address user) external view returns (uint256);
    
    function emergencyPause() external;
    
    function emergencyUnpause() external;
}
