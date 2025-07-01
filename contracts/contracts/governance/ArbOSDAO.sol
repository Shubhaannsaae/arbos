// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {EnumerableSet} from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

/**
 * @title ArbOSDAO
 * @dev Decentralized governance for the ArbOS ecosystem
 * @notice Manages proposals, voting, and execution of governance decisions
 */
contract ArbOSDAO is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using EnumerableSet for EnumerableSet.UintSet;

    // Proposal states
    enum ProposalState {
        Pending,
        Active,
        Canceled,
        Defeated,
        Succeeded,
        Queued,
        Expired,
        Executed
    }

    // Proposal struct
    struct Proposal {
        uint256 id;
        address proposer;
        address[] targets;
        uint256[] values;
        string[] signatures;
        bytes[] calldatas;
        uint256 startBlock;
        uint256 endBlock;
        string description;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        bool canceled;
        bool executed;
        mapping(address => Receipt) receipts;
    }

    // Vote receipt
    struct Receipt {
        bool hasVoted;
        uint8 support; // 0=against, 1=for, 2=abstain
        uint256 votes;
    }

    // Governance parameters
    struct GovernanceParams {
        uint256 votingDelay;     // blocks
        uint256 votingPeriod;    // blocks  
        uint256 proposalThreshold; // minimum tokens to propose
        uint256 quorumNumerator;   // quorum percentage numerator
        uint256 quorumDenominator; // quorum percentage denominator
        uint256 timelockDelay;     // timelock delay in seconds
    }

    // State variables
    mapping(uint256 => Proposal) private _proposals;
    mapping(address => uint256) public latestProposalIds;
    
    EnumerableSet.UintSet private activeProposals;
    
    uint256 private _proposalCount;
    IERC20 public governanceToken;
    address public timelock;
    address public arbOSCore;
    
    GovernanceParams public params;

    // Events
    event ProposalCreated(
        uint256 id,
        address proposer,
        address[] targets,
        uint256[] values,
        string[] signatures,
        bytes[] calldatas,
        uint256 startBlock,
        uint256 endBlock,
        string description
    );
    
    event VoteCast(
        address indexed voter,
        uint256 proposalId,
        uint8 support,
        uint256 weight,
        string reason
    );
    
    event ProposalCanceled(uint256 id);
    event ProposalQueued(uint256 id, uint256 eta);
    event ProposalExecuted(uint256 id);
    event QuorumNumeratorUpdated(uint256 oldQuorum, uint256 newQuorum);

    modifier onlyTimelock() {
        require(msg.sender == timelock, "ArbOSDAO: Only timelock");
        _;
    }

    constructor(
        address _governanceToken,
        address _timelock,
        address _arbOSCore,
        GovernanceParams memory _params
    ) {
        governanceToken = IERC20(_governanceToken);
        timelock = _timelock;
        arbOSCore = _arbOSCore;
        params = _params;
    }

    /**
     * @dev Create a new proposal
     */
    function propose(
        address[] memory targets,
        uint256[] memory values,
        string[] memory signatures,
        bytes[] memory calldatas,
        string memory description
    ) external returns (uint256) {
        require(
            getVotes(msg.sender, block.number - 1) >= proposalThreshold(),
            "ArbOSDAO: Proposer votes below threshold"
        );
        
        require(
            targets.length == values.length &&
            targets.length == signatures.length &&
            targets.length == calldatas.length,
            "ArbOSDAO: Proposal function information mismatch"
        );
        
        require(targets.length != 0, "ArbOSDAO: Empty proposal");
        require(targets.length <= 10, "ArbOSDAO: Too many actions");

        uint256 latestProposalId = latestProposalIds[msg.sender];
        if (latestProposalId != 0) {
            ProposalState proposerLatestProposalState = state(latestProposalId);
            require(
                proposerLatestProposalState != ProposalState.Active,
                "ArbOSDAO: One live proposal per proposer"
            );
            require(
                proposerLatestProposalState != ProposalState.Pending,
                "ArbOSDAO: One live proposal per proposer"
            );
        }

        uint256 proposalId = ++_proposalCount;
        
        Proposal storage newProposal = _proposals[proposalId];
        newProposal.id = proposalId;
        newProposal.proposer = msg.sender;
        newProposal.targets = targets;
        newProposal.values = values;
        newProposal.signatures = signatures;
        newProposal.calldatas = calldatas;
        newProposal.startBlock = block.number + votingDelay();
        newProposal.endBlock = newProposal.startBlock + votingPeriod();
        newProposal.description = description;

        latestProposalIds[msg.sender] = proposalId;
        activeProposals.add(proposalId);

        emit ProposalCreated(
            proposalId,
            msg.sender,
            targets,
            values,
            signatures,
            calldatas,
            newProposal.startBlock,
            newProposal.endBlock,
            description
        );

        return proposalId;
    }

    /**
     * @dev Cast a vote for a proposal
     */
    function castVote(uint256 proposalId, uint8 support) external returns (uint256) {
        return _castVote(msg.sender, proposalId, support, "");
    }

    /**
     * @dev Cast a vote with reason
     */
    function castVoteWithReason(
        uint256 proposalId,
        uint8 support,
        string calldata reason
    ) external returns (uint256) {
        return _castVote(msg.sender, proposalId, support, reason);
    }

    /**
     * @dev Queue a succeeded proposal
     */
    function queue(uint256 proposalId) external {
        require(
            state(proposalId) == ProposalState.Succeeded,
            "ArbOSDAO: Proposal can only be queued if it is succeeded"
        );
        
        Proposal storage proposal = _proposals[proposalId];
        uint256 eta = block.timestamp + params.timelockDelay;
        
        for (uint256 i = 0; i < proposal.targets.length; i++) {
            _queueOrRevertInternal(
                proposal.targets[i],
                proposal.values[i],
                proposal.signatures[i],
                proposal.calldatas[i],
                eta
            );
        }
        
        activeProposals.remove(proposalId);
        emit ProposalQueued(proposalId, eta);
    }

    /**
     * @dev Execute a queued proposal
     */
    function execute(uint256 proposalId) external payable nonReentrant {
        require(
            state(proposalId) == ProposalState.Queued,
            "ArbOSDAO: Proposal can only be executed if it is queued"
        );
        
        Proposal storage proposal = _proposals[proposalId];
        proposal.executed = true;
        
        for (uint256 i = 0; i < proposal.targets.length; i++) {
            _executeTransaction(
                proposal.targets[i],
                proposal.values[i],
                proposal.signatures[i],
                proposal.calldatas[i]
            );
        }
        
        emit ProposalExecuted(proposalId);
    }

    /**
     * @dev Cancel a proposal
     */
    function cancel(uint256 proposalId) external {
        require(state(proposalId) != ProposalState.Executed, "ArbOSDAO: Cannot cancel executed proposal");
        
        Proposal storage proposal = _proposals[proposalId];
        require(
            msg.sender == proposal.proposer ||
            getVotes(proposal.proposer, block.number - 1) < proposalThreshold(),
            "ArbOSDAO: Proposer above threshold"
        );

        proposal.canceled = true;
        activeProposals.remove(proposalId);

        for (uint256 i = 0; i < proposal.targets.length; i++) {
            _cancelTransaction(
                proposal.targets[i],
                proposal.values[i],
                proposal.signatures[i],
                proposal.calldatas[i]
            );
        }

        emit ProposalCanceled(proposalId);
    }

    /**
     * @dev Get proposal state
     */
    function state(uint256 proposalId) public view returns (ProposalState) {
        require(_proposalExists(proposalId), "ArbOSDAO: Unknown proposal id");
        
        Proposal storage proposal = _proposals[proposalId];
        
        if (proposal.canceled) {
            return ProposalState.Canceled;
        } else if (block.number <= proposal.startBlock) {
            return ProposalState.Pending;
        } else if (block.number <= proposal.endBlock) {
            return ProposalState.Active;
        } else if (proposal.forVotes <= proposal.againstVotes || proposal.forVotes < quorum(proposal.endBlock)) {
            return ProposalState.Defeated;
        } else if (proposal.executed) {
            return ProposalState.Executed;
        } else if (_isQueued(proposalId)) {
            return ProposalState.Queued;
        } else {
            return ProposalState.Succeeded;
        }
    }

    /**
     * @dev Get votes for an account at a specific block
     */
    function getVotes(address account, uint256 blockNumber) public view returns (uint256) {
        // In production, implement checkpoint-based voting power
        // For now, return current balance
        return governanceToken.balanceOf(account);
    }

    /**
     * @dev Get current quorum requirement
     */
    function quorum(uint256 blockNumber) public view returns (uint256) {
        uint256 totalSupply = governanceToken.totalSupply();
        return (totalSupply * params.quorumNumerator) / params.quorumDenominator;
    }

    /**
     * @dev Get voting delay
     */
    function votingDelay() public view returns (uint256) {
        return params.votingDelay;
    }

    /**
     * @dev Get voting period
     */
    function votingPeriod() public view returns (uint256) {
        return params.votingPeriod;
    }

    /**
     * @dev Get proposal threshold
     */
    function proposalThreshold() public view returns (uint256) {
        return params.proposalThreshold;
    }

    /**
     * @dev Get proposal details
     */
    function getProposal(uint256 proposalId) external view returns (
        uint256 id,
        address proposer,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 abstainVotes,
        uint256 startBlock,
        uint256 endBlock,
        bool canceled,
        bool executed
    ) {
        require(_proposalExists(proposalId), "ArbOSDAO: Unknown proposal id");
        
        Proposal storage proposal = _proposals[proposalId];
        return (
            proposal.id,
            proposal.proposer,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.abstainVotes,
            proposal.startBlock,
            proposal.endBlock,
            proposal.canceled,
            proposal.executed
        );
    }

    /**
     * @dev Get proposal actions
     */
    function getActions(uint256 proposalId) external view returns (
        address[] memory targets,
        uint256[] memory values,
        string[] memory signatures,
        bytes[] memory calldatas
    ) {
        require(_proposalExists(proposalId), "ArbOSDAO: Unknown proposal id");
        
        Proposal storage proposal = _proposals[proposalId];
        return (
            proposal.targets,
            proposal.values,
            proposal.signatures,
            proposal.calldatas
        );
    }

    /**
     * @dev Get receipt for a voter on a proposal
     */
    function getReceipt(uint256 proposalId, address voter) external view returns (Receipt memory) {
        return _proposals[proposalId].receipts[voter];
    }

    /**
     * @dev Internal function to cast vote
     */
    function _castVote(
        address voter,
        uint256 proposalId,
        uint8 support,
        string memory reason
    ) internal returns (uint256) {
        require(state(proposalId) == ProposalState.Active, "ArbOSDAO: Voting is closed");
        require(support <= 2, "ArbOSDAO: Invalid vote type");
        
        Proposal storage proposal = _proposals[proposalId];
        Receipt storage receipt = proposal.receipts[voter];
        require(!receipt.hasVoted, "ArbOSDAO: Voter already voted");

        uint256 weight = getVotes(voter, proposal.startBlock);
        
        if (support == 0) {
            proposal.againstVotes += weight;
        } else if (support == 1) {
            proposal.forVotes += weight;
        } else {
            proposal.abstainVotes += weight;
        }

        receipt.hasVoted = true;
        receipt.support = support;
        receipt.votes = weight;

        emit VoteCast(voter, proposalId, support, weight, reason);

        return weight;
    }

    /**
     * @dev Check if proposal exists
     */
    function _proposalExists(uint256 proposalId) internal view returns (bool) {
        return proposalId > 0 && proposalId <= _proposalCount;
    }

    /**
     * @dev Check if proposal is queued
     */
    function _isQueued(uint256 proposalId) internal view returns (bool) {
        // In production, check timelock queue status
        return false; // Simplified
    }

    /**
     * @dev Queue transaction in timelock
     */
    function _queueOrRevertInternal(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data,
        uint256 eta
    ) internal {
        if (timelock != address(0)) {
            (bool success, ) = timelock.call(
                abi.encodeWithSignature(
                    "queueTransaction(address,uint256,string,bytes,uint256)",
                    target,
                    value,
                    signature,
                    data,
                    eta
                )
            );
            require(success, "ArbOSDAO: Timelock queue failed");
        }
    }

    /**
     * @dev Execute transaction
     */
    function _executeTransaction(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data
    ) internal {
        bytes memory callData;
        
        if (bytes(signature).length == 0) {
            callData = data;
        } else {
            callData = abi.encodePacked(bytes4(keccak256(bytes(signature))), data);
        }

        (bool success, ) = target.call{value: value}(callData);
        require(success, "ArbOSDAO: Transaction execution reverted");
    }

    /**
     * @dev Cancel transaction in timelock
     */
    function _cancelTransaction(
        address target,
        uint256 value,
        string memory signature,
        bytes memory data
    ) internal {
        if (timelock != address(0)) {
            (bool success, ) = timelock.call(
                abi.encodeWithSignature(
                    "cancelTransaction(address,uint256,string,bytes,uint256)",
                    target,
                    value,
                    signature,
                    data,
                    block.timestamp + params.timelockDelay
                )
            );
            // Don't revert if cancel fails
        }
    }

    /**
     * @dev Update governance parameters (only via governance)
     */
    function updateGovernanceParams(GovernanceParams memory newParams) external onlyTimelock {
        require(newParams.quorumDenominator > 0, "ArbOSDAO: Invalid quorum denominator");
        require(newParams.quorumNumerator <= newParams.quorumDenominator, "ArbOSDAO: Invalid quorum");
        
        uint256 oldQuorum = params.quorumNumerator;
        params = newParams;
        
        emit QuorumNumeratorUpdated(oldQuorum, newParams.quorumNumerator);
    }

    /**
     * @dev Update timelock address (only via governance)
     */
    function updateTimelock(address newTimelock) external onlyTimelock {
        timelock = newTimelock;
    }

    /**
     * @dev Get active proposals
     */
    function getActiveProposals() external view returns (uint256[] memory) {
        uint256 length = activeProposals.length();
        uint256[] memory proposals = new uint256[](length);
        
        for (uint256 i = 0; i < length; i++) {
            proposals[i] = activeProposals.at(i);
        }
        
        return proposals;
    }

    /**
     * @dev Get total number of proposals
     */
    function proposalCount() external view returns (uint256) {
        return _proposalCount;
    }

    // Receive ETH
    receive() external payable {}
}
