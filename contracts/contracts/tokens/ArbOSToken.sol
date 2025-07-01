// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {ERC20Votes} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {Pausable} from "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title ArbOSToken
 * @dev ERC20 token for the ArbOS ecosystem with governance capabilities
 * @notice Used for governance, staking, and rewards with voting power
 */
contract ArbOSToken is ERC20, ERC20Permit, ERC20Votes, Ownable, Pausable {
    uint256 public constant INITIAL_SUPPLY = 1_000_000_000 * 10 ** 18; // 1 billion tokens
    uint256 public constant MAX_SUPPLY = 10_000_000_000 * 10 ** 18; // 10 billion max supply
    
    // Minting configuration
    mapping(address => bool) public minters;
    mapping(address => uint256) public mintingAllowance;
    
    // Staking configuration
    mapping(address => uint256) public stakedBalances;
    mapping(address => uint256) public stakingTimestamps;
    uint256 public totalStaked;
    uint256 public stakingRewardRate = 500; // 5% APY in basis points
    
    // Governance integration
    address public governanceContract;
    
    // Events
    event MinterAdded(address indexed minter, uint256 allowance);
    event MinterRemoved(address indexed minter);
    event TokensStaked(address indexed user, uint256 amount);
    event TokensUnstaked(address indexed user, uint256 amount, uint256 rewards);
    event GovernanceContractUpdated(address indexed newGovernance);

    modifier onlyMinter() {
        require(minters[msg.sender], "ArbOSToken: Not authorized minter");
        _;
    }

    constructor() 
        ERC20("ArbOS Token", "ARBOS") 
        ERC20Permit("ArbOS Token")
    {
        _mint(msg.sender, INITIAL_SUPPLY);
    }

    /**
     * @dev Mint new tokens with supply cap enforcement
     * @param to Recipient address
     * @param amount Amount to mint
     */
    function mint(address to, uint256 amount) external onlyMinter whenNotPaused {
        require(to != address(0), "ArbOSToken: Cannot mint to zero address");
        require(totalSupply() + amount <= MAX_SUPPLY, "ArbOSToken: Exceeds max supply");
        require(mintingAllowance[msg.sender] >= amount, "ArbOSToken: Exceeds minting allowance");
        
        mintingAllowance[msg.sender] -= amount;
        _mint(to, amount);
    }

    /**
     * @dev Burn tokens from caller's balance
     * @param amount Amount to burn
     */
    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }

    /**
     * @dev Burn tokens from specified account (with allowance)
     * @param account Account to burn from
     * @param amount Amount to burn
     */
    function burnFrom(address account, uint256 amount) external {
        _spendAllowance(account, msg.sender, amount);
        _burn(account, amount);
    }

    /**
     * @dev Stake tokens to earn rewards and voting power
     * @param amount Amount to stake
     */
    function stake(uint256 amount) external whenNotPaused {
        require(amount > 0, "ArbOSToken: Cannot stake zero");
        require(balanceOf(msg.sender) >= amount, "ArbOSToken: Insufficient balance");
        
        // Claim any pending rewards before staking more
        if (stakedBalances[msg.sender] > 0) {
            _claimStakingRewards(msg.sender);
        }
        
        _transfer(msg.sender, address(this), amount);
        stakedBalances[msg.sender] += amount;
        stakingTimestamps[msg.sender] = block.timestamp;
        totalStaked += amount;
        
        emit TokensStaked(msg.sender, amount);
    }

    /**
     * @dev Unstake tokens and claim rewards
     * @param amount Amount to unstake
     */
    function unstake(uint256 amount) external {
        require(amount > 0, "ArbOSToken: Cannot unstake zero");
        require(stakedBalances[msg.sender] >= amount, "ArbOSToken: Insufficient staked balance");
        
        uint256 rewards = _claimStakingRewards(msg.sender);
        
        stakedBalances[msg.sender] -= amount;
        totalStaked -= amount;
        stakingTimestamps[msg.sender] = block.timestamp;
        
        _transfer(address(this), msg.sender, amount);
        
        emit TokensUnstaked(msg.sender, amount, rewards);
    }

    /**
     * @dev Claim staking rewards without unstaking
     */
    function claimRewards() external {
        uint256 rewards = _claimStakingRewards(msg.sender);
        require(rewards > 0, "ArbOSToken: No rewards to claim");
    }

    /**
     * @dev Calculate pending staking rewards
     * @param user User address
     */
    function pendingRewards(address user) external view returns (uint256) {
        if (stakedBalances[user] == 0) {
            return 0;
        }
        
        uint256 stakingDuration = block.timestamp - stakingTimestamps[user];
        uint256 annualReward = (stakedBalances[user] * stakingRewardRate) / 10000;
        
        return (annualReward * stakingDuration) / 365 days;
    }

    /**
     * @dev Internal function to claim staking rewards
     */
    function _claimStakingRewards(address user) internal returns (uint256 rewards) {
        if (stakedBalances[user] == 0) {
            return 0;
        }
        
        uint256 stakingDuration = block.timestamp - stakingTimestamps[user];
        uint256 annualReward = (stakedBalances[user] * stakingRewardRate) / 10000;
        rewards = (annualReward * stakingDuration) / 365 days;
        
        if (rewards > 0 && totalSupply() + rewards <= MAX_SUPPLY) {
            stakingTimestamps[user] = block.timestamp;
            _mint(user, rewards);
        }
        
        return rewards;
    }

    /**
     * @dev Add authorized minter
     * @param minter Minter address
     * @param allowance Minting allowance
     */
    function addMinter(address minter, uint256 allowance) external onlyOwner {
        require(minter != address(0), "ArbOSToken: Invalid minter address");
        minters[minter] = true;
        mintingAllowance[minter] = allowance;
        emit MinterAdded(minter, allowance);
    }

    /**
     * @dev Remove authorized minter
     * @param minter Minter address
     */
    function removeMinter(address minter) external onlyOwner {
        minters[minter] = false;
        mintingAllowance[minter] = 0;
        emit MinterRemoved(minter);
    }

    /**
     * @dev Update staking reward rate
     * @param newRate New reward rate in basis points
     */
    function updateStakingRewardRate(uint256 newRate) external onlyOwner {
        require(newRate <= 2000, "ArbOSToken: Rate too high"); // Max 20% APY
        stakingRewardRate = newRate;
    }

    /**
     * @dev Set governance contract address
     * @param governance Governance contract address
     */
    function setGovernanceContract(address governance) external onlyOwner {
        governanceContract = governance;
        emit GovernanceContractUpdated(governance);
    }

    /**
     * @dev Emergency pause function
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause function
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Get staking information for user
     */
    function getStakingInfo(address user) external view returns (
        uint256 stakedAmount,
        uint256 stakingTimestamp,
        uint256 pendingRewardsAmount
    ) {
        stakedAmount = stakedBalances[user];
        stakingTimestamp = stakingTimestamps[user];
        
        if (stakedAmount > 0) {
            uint256 stakingDuration = block.timestamp - stakingTimestamp;
            uint256 annualReward = (stakedAmount * stakingRewardRate) / 10000;
            pendingRewardsAmount = (annualReward * stakingDuration) / 365 days;
        }
        
        return (stakedAmount, stakingTimestamp, pendingRewardsAmount);
    }

    // Override required functions
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }

    function _afterTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override(ERC20, ERC20Votes) {
        super._afterTokenTransfer(from, to, amount);
    }

    function _mint(address to, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._mint(to, amount);
    }

    function _burn(address account, uint256 amount) internal override(ERC20, ERC20Votes) {
        super._burn(account, amount);
    }
}
