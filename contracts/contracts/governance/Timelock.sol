// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title Timelock
 * @dev Timelock controller for delayed execution of governance proposals
 * @notice Provides time-delayed execution with cancellation capabilities
 */
contract Timelock is Ownable, ReentrancyGuard {
    
    // Transaction struct
    struct Transaction {
        address target;
        uint256 value;
        string signature;
        bytes data;
        uint256 eta;
        bool executed;
        bool canceled;
    }

    // State variables
    mapping(bytes32 => Transaction) public queuedTransactions;
    mapping(address => bool) public admins;
    
    uint256 public constant GRACE_PERIOD = 14 days;
    uint256 public constant MINIMUM_DELAY = 2 days;
    uint256 public constant MAXIMUM_DELAY = 30 days;
    
    uint256 public delay;
    address public pendingAdmin;
    
    // Events
    event NewAdmin(address indexed newAdmin);
    event NewPendingAdmin(address indexed newPendingAdmin);
    event NewDelay(uint256 indexed newDelay);
    event CancelTransaction(
        bytes32 indexed txHash,
        address indexed target,
        uint256 value,
        string signature,
        bytes data,
        uint256 eta
    );
    event ExecuteTransaction(
        bytes32 indexed txHash,
        address indexed target,
        uint256 value,
        string signature,
        bytes data,
        uint256 eta
    );
    event QueueTransaction(
        bytes32 indexed txHash,
        address indexed target,
        uint256 value,
        string signature,
        bytes data,
        uint256 eta
    );

    modifier onlyAdmin() {
        require(admins[msg.sender], "Timelock: Only admin");
        _;
    }

    modifier onlyTimelock() {
        require(msg.sender == address(this), "Timelock: Only timelock");
        _;
    }

    constructor(address admin_, uint256 delay_) {
        require(delay_ >= MINIMUM_DELAY, "Timelock: Delay must exceed minimum delay");
        require(delay_ <= MAXIMUM_DELAY, "Timelock: Delay must not exceed maximum delay");

        admins[admin_] = true;
        delay = delay_;
    }

    /**
     * @dev Set pending admin
     */
    function setPendingAdmin(address pendingAdmin_) external onlyTimelock {
        pendingAdmin = pendingAdmin_;
        emit NewPendingAdmin(pendingAdmin);
    }

    /**
     * @dev Accept admin role
     */
    function acceptAdmin() external {
        require(msg.sender == pendingAdmin, "Timelock: Only pending admin");
        
        admins[pendingAdmin] = true;
        pendingAdmin = address(0);
        
        emit NewAdmin(pendingAdmin);
    }

    /**
     * @dev Set new delay
     */
    function setDelay(uint256 delay_) external onlyTimelock {
        require(delay_ >= MINIMUM_DELAY, "Timelock: Delay must exceed minimum delay");
        require(delay_ <= MAXIMUM_DELAY, "Timelock: Delay must not exceed maximum delay");
        
        delay = delay_;
        emit NewDelay(delay);
    }

    /**
     * @dev Queue a transaction
     */
    function queueTransaction(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data,
        uint256 eta
    ) external onlyAdmin returns (bytes32) {
        require(eta >= getBlockTimestamp() + delay, "Timelock: ETA must satisfy delay");

        bytes32 txHash = keccak256(abi.encode(target, value, signature, data, eta));
        
        require(!queuedTransactions[txHash].eta != 0, "Timelock: Transaction already queued");
        
        queuedTransactions[txHash] = Transaction({
            target: target,
            value: value,
            signature: signature,
            data: data,
            eta: eta,
            executed: false,
            canceled: false
        });

        emit QueueTransaction(txHash, target, value, signature, data, eta);
        return txHash;
    }

    /**
     * @dev Cancel a queued transaction
     */
    function cancelTransaction(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data,
        uint256 eta
    ) external onlyAdmin {
        bytes32 txHash = keccak256(abi.encode(target, value, signature, data, eta));
        
        Transaction storage transaction = queuedTransactions[txHash];
        require(transaction.eta != 0, "Timelock: Transaction not queued");
        require(!transaction.executed, "Timelock: Transaction already executed");
        require(!transaction.canceled, "Timelock: Transaction already canceled");
        
        transaction.canceled = true;

        emit CancelTransaction(txHash, target, value, signature, data, eta);
    }

    /**
     * @dev Execute a queued transaction
     */
    function executeTransaction(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data,
        uint256 eta
    ) external payable onlyAdmin nonReentrant returns (bytes memory) {
        bytes32 txHash = keccak256(abi.encode(target, value, signature, data, eta));
        
        Transaction storage transaction = queuedTransactions[txHash];
        require(transaction.eta != 0, "Timelock: Transaction not queued");
        require(!transaction.executed, "Timelock: Transaction already executed");
        require(!transaction.canceled, "Timelock: Transaction canceled");
        require(getBlockTimestamp() >= eta, "Timelock: Transaction not ready");
        require(getBlockTimestamp() <= eta + GRACE_PERIOD, "Timelock: Transaction stale");

        transaction.executed = true;

        bytes memory callData;
        if (bytes(signature).length == 0) {
            callData = data;
        } else {
            callData = abi.encodePacked(bytes4(keccak256(bytes(signature))), data);
        }

        (bool success, bytes memory returnData) = target.call{value: value}(callData);
        require(success, "Timelock: Transaction execution reverted");

        emit ExecuteTransaction(txHash, target, value, signature, data, eta);

        return returnData;
    }

    /**
     * @dev Get transaction hash
     */
    function getTransactionHash(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data,
        uint256 eta
    ) external pure returns (bytes32) {
        return keccak256(abi.encode(target, value, signature, data, eta));
    }

    /**
     * @dev Check if transaction is queued
     */
    function isTransactionQueued(bytes32 txHash) external view returns (bool) {
        Transaction memory transaction = queuedTransactions[txHash];
        return transaction.eta != 0 && !transaction.executed && !transaction.canceled;
    }

    /**
     * @dev Check if transaction is ready for execution
     */
    function isTransactionReady(bytes32 txHash) external view returns (bool) {
        Transaction memory transaction = queuedTransactions[txHash];
        return transaction.eta != 0 && 
               !transaction.executed && 
               !transaction.canceled &&
               getBlockTimestamp() >= transaction.eta &&
               getBlockTimestamp() <= transaction.eta + GRACE_PERIOD;
    }

    /**
     * @dev Get transaction details
     */
    function getTransaction(bytes32 txHash) external view returns (Transaction memory) {
        return queuedTransactions[txHash];
    }

    /**
     * @dev Add admin
     */
    function addAdmin(address admin) external onlyTimelock {
        require(admin != address(0), "Timelock: Invalid admin address");
        admins[admin] = true;
    }

    /**
     * @dev Remove admin
     */
    function removeAdmin(address admin) external onlyTimelock {
        require(admin != address(0), "Timelock: Invalid admin address");
        admins[admin] = false;
    }

    /**
     * @dev Emergency pause function
     */
    function emergencyPause() external onlyAdmin {
        // In production, implement emergency pause logic
        // This could disable transaction execution temporarily
    }

    /**
     * @dev Batch queue multiple transactions
     */
    function batchQueueTransactions(
        address[] memory targets,
        uint256[] memory values,
        string[] memory signatures,
        bytes[] memory datas,
        uint256[] memory etas
    ) external onlyAdmin returns (bytes32[] memory) {
        require(
            targets.length == values.length &&
            targets.length == signatures.length &&
            targets.length == datas.length &&
            targets.length == etas.length,
            "Timelock: Array length mismatch"
        );

        bytes32[] memory txHashes = new bytes32[](targets.length);
        
        for (uint256 i = 0; i < targets.length; i++) {
            txHashes[i] = queueTransaction(
                targets[i],
                values[i],
                signatures[i],
                datas[i],
                etas[i]
            );
        }

        return txHashes;
    }

    /**
     * @dev Batch execute multiple transactions
     */
    function batchExecuteTransactions(
        address[] memory targets,
        uint256[] memory values,
        string[] memory signatures,
        bytes[] memory datas,
        uint256[] memory etas
    ) external payable onlyAdmin {
        require(
            targets.length == values.length &&
            targets.length == signatures.length &&
            targets.length == datas.length &&
            targets.length == etas.length,
            "Timelock: Array length mismatch"
        );

        for (uint256 i = 0; i < targets.length; i++) {
            executeTransaction(
                targets[i],
                values[i],
                signatures[i],
                datas[i],
                etas[i]
            );
        }
    }

    /**
     * @dev Get current block timestamp
     */
    function getBlockTimestamp() public view returns (uint256) {
        return block.timestamp;
    }

    /**
     * @dev Check if address is admin
     */
    function isAdmin(address account) external view returns (bool) {
        return admins[account];
    }

    /**
     * @dev Get delay amount
     */
    function getDelay() external view returns (uint256) {
        return delay;
    }

    /**
     * @dev Get grace period
     */
    function getGracePeriod() external pure returns (uint256) {
        return GRACE_PERIOD;
    }

    /**
     * @dev Fallback function to receive ETH
     */
    receive() external payable {}

    /**
     * @dev Fallback function for calls
     */
    fallback() external payable {}
}
