// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {Pausable} from "@openzeppelin/contracts/security/Pausable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title RewardToken
 * @dev ERC20 token for rewards distribution in the ArbOS ecosystem
 * @notice Used for incentivizing users with vesting and distribution controls
 */
contract RewardToken is ERC20, Ownable, Pausable, ReentrancyGuard {
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10 ** 18; // 100 million tokens
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10 ** 18; // 1 billion max supply
    
    // Distribution configuration
    mapping(address => bool) public distributors;
    mapping(address => uint256) public distributionAllowance;
    
    // Vesting configuration
    struct VestingSchedule {
        uint256 totalAmount;
        uint256 claimedAmount;
        uint256 startTime;
        uint256 duration;
        uint256 cliffPeriod;
        bool revoked;
    }
    
    mapping(address => VestingSchedule[]) public vestingSchedules;
    mapping(address => uint256) public totalVested;
    
    // Reward pools
    struct RewardPool {
        uint256 totalRewards;
        uint256 rewardRate; // tokens per second
        uint256 lastUpdateTime;
        uint256 rewardPerTokenStored;
        mapping(address => uint256) userRewardPerTokenPaid;
        mapping(address => uint256) rewards;
        mapping(address => uint256) balances;
        uint256 totalSupply;
        bool active;
    }
    
    mapping(bytes32 => RewardPool) public rewardPools;
    bytes32[] public poolIds;
    
    // Events
    event DistributorAdded(address indexed distributor, uint256 allowance);
    event DistributorRemoved(address indexed distributor);
    event VestingScheduleCreated(
        address indexed beneficiary,
        uint256 amount,
        uint256 duration,
        uint256 cliffPeriod
    );
    event TokensVested(address indexed beneficiary, uint256 amount);
    event VestingRevoked(address indexed beneficiary, uint256 vestingIndex);
    event RewardPoolCreated(bytes32 indexed poolId, uint256 rewardRate);
    event RewardClaimed(address indexed user, bytes32 indexed poolId, uint256 amount);

    modifier onlyDistributor() {
        require(distributors[msg.sender], "RewardToken: Not authorized distributor");
        _;
    }

    modifier updateReward(bytes32 poolId, address account) {
        RewardPool storage pool = rewardPools[poolId];
        pool.rewardPerTokenStored = rewardPerToken(poolId);
        pool.lastUpdateTime = lastTimeRewardApplicable(poolId);
        
        if (account != address(0)) {
            pool.rewards[account] = earned(poolId, account);
            pool.userRewardPerTokenPaid[account] = pool.rewardPerTokenStored;
        }
        _;
    }

    constructor() ERC20("ArbOS Reward Token", "ARBOR") {
        _mint(msg.sender, INITIAL_SUPPLY);
    }

    /**
     * @dev Mint new reward tokens with supply cap
     * @param to Recipient address
     * @param amount Amount to mint
     */
    function mint(address to, uint256 amount) external onlyDistributor whenNotPaused {
        require(to != address(0), "RewardToken: Cannot mint to zero address");
        require(totalSupply() + amount <= MAX_SUPPLY, "RewardToken: Exceeds max supply");
        require(distributionAllowance[msg.sender] >= amount, "RewardToken: Exceeds distribution allowance");
        
        distributionAllowance[msg.sender] -= amount;
        _mint(to, amount);
    }

    /**
     * @dev Burn reward tokens
     * @param amount Amount to burn
     */
    function burn(uint256 amount) external {
        _burn(msg.sender, amount);
    }

    /**
     * @dev Create vesting schedule for beneficiary
     * @param beneficiary Address of the beneficiary
     * @param amount Total amount to vest
     * @param duration Vesting duration in seconds
     * @param cliffPeriod Cliff period in seconds
     */
    function createVestingSchedule(
        address beneficiary,
        uint256 amount,
        uint256 duration,
        uint256 cliffPeriod
    ) external onlyDistributor whenNotPaused {
        require(beneficiary != address(0), "RewardToken: Invalid beneficiary");
        require(amount > 0, "RewardToken: Amount must be greater than 0");
        require(duration > 0, "RewardToken: Duration must be greater than 0");
        require(cliffPeriod <= duration, "RewardToken: Cliff period cannot exceed duration");
        require(balanceOf(msg.sender) >= amount, "RewardToken: Insufficient balance");
        
        // Transfer tokens to this contract for vesting
        _transfer(msg.sender, address(this), amount);
        
        vestingSchedules[beneficiary].push(VestingSchedule({
            totalAmount: amount,
            claimedAmount: 0,
            startTime: block.timestamp,
            duration: duration,
            cliffPeriod: cliffPeriod,
            revoked: false
        }));
        
        totalVested[beneficiary] += amount;
        
        emit VestingScheduleCreated(beneficiary, amount, duration, cliffPeriod);
    }

    /**
     * @dev Claim vested tokens
     * @param vestingIndex Index of the vesting schedule
     */
    function claimVestedTokens(uint256 vestingIndex) external nonReentrant {
        require(vestingIndex < vestingSchedules[msg.sender].length, "RewardToken: Invalid vesting index");
        
        VestingSchedule storage schedule = vestingSchedules[msg.sender][vestingIndex];
        require(!schedule.revoked, "RewardToken: Vesting schedule revoked");
        require(block.timestamp >= schedule.startTime + schedule.cliffPeriod, "RewardToken: Cliff period not met");
        
        uint256 vestedAmount = _calculateVestedAmount(schedule);
        uint256 claimableAmount = vestedAmount - schedule.claimedAmount;
        
        require(claimableAmount > 0, "RewardToken: No tokens to claim");
        
        schedule.claimedAmount += claimableAmount;
        totalVested[msg.sender] -= claimableAmount;
        
        _transfer(address(this), msg.sender, claimableAmount);
        
        emit TokensVested(msg.sender, claimableAmount);
    }

    /**
     * @dev Calculate vested amount for a schedule
     */
    function _calculateVestedAmount(VestingSchedule memory schedule) internal view returns (uint256) {
        if (block.timestamp < schedule.startTime + schedule.cliffPeriod) {
            return 0;
        }
        
        if (block.timestamp >= schedule.startTime + schedule.duration) {
            return schedule.totalAmount;
        }
        
        uint256 elapsedTime = block.timestamp - schedule.startTime;
        return (schedule.totalAmount * elapsedTime) / schedule.duration;
    }

    /**
     * @dev Get claimable vested amount
     * @param beneficiary Beneficiary address
     * @param vestingIndex Vesting schedule index
     */
    function getClaimableAmount(address beneficiary, uint256 vestingIndex) 
        external 
        view 
        returns (uint256) 
    {
        if (vestingIndex >= vestingSchedules[beneficiary].length) {
            return 0;
        }
        
        VestingSchedule memory schedule = vestingSchedules[beneficiary][vestingIndex];
        if (schedule.revoked) {
            return 0;
        }
        
        uint256 vestedAmount = _calculateVestedAmount(schedule);
        return vestedAmount - schedule.claimedAmount;
    }

    /**
     * @dev Create reward pool
     * @param poolId Unique pool identifier
     * @param rewardRate Reward rate in tokens per second
     * @param totalRewards Total rewards for the pool
     */
    function createRewardPool(
        bytes32 poolId,
        uint256 rewardRate,
        uint256 totalRewards
    ) external onlyOwner {
        require(rewardPools[poolId].rewardRate == 0, "RewardToken: Pool already exists");
        require(rewardRate > 0, "RewardToken: Invalid reward rate");
        require(totalRewards > 0, "RewardToken: Invalid total rewards");
        require(balanceOf(msg.sender) >= totalRewards, "RewardToken: Insufficient balance");
        
        // Transfer rewards to contract
        _transfer(msg.sender, address(this), totalRewards);
        
        RewardPool storage pool = rewardPools[poolId];
        pool.totalRewards = totalRewards;
        pool.rewardRate = rewardRate;
        pool.lastUpdateTime = block.timestamp;
        pool.active = true;
        
        poolIds.push(poolId);
        
        emit RewardPoolCreated(poolId, rewardRate);
    }

    /**
     * @dev Stake tokens in reward pool
     * @param poolId Pool identifier
     * @param amount Amount to stake
     */
    function stakeInPool(bytes32 poolId, uint256 amount) 
        external 
        updateReward(poolId, msg.sender) 
        whenNotPaused 
    {
        require(amount > 0, "RewardToken: Cannot stake 0");
        require(rewardPools[poolId].active, "RewardToken: Pool not active");
        require(balanceOf(msg.sender) >= amount, "RewardToken: Insufficient balance");
        
        RewardPool storage pool = rewardPools[poolId];
        pool.totalSupply += amount;
        pool.balances[msg.sender] += amount;
        
        _transfer(msg.sender, address(this), amount);
    }

    /**
     * @dev Withdraw staked tokens from pool
     * @param poolId Pool identifier
     * @param amount Amount to withdraw
     */
    function withdrawFromPool(bytes32 poolId, uint256 amount) 
        external 
        updateReward(poolId, msg.sender) 
    {
        require(amount > 0, "RewardToken: Cannot withdraw 0");
        
        RewardPool storage pool = rewardPools[poolId];
        require(pool.balances[msg.sender] >= amount, "RewardToken: Insufficient staked balance");
        
        pool.totalSupply -= amount;
        pool.balances[msg.sender] -= amount;
        
        _transfer(address(this), msg.sender, amount);
    }

    /**
     * @dev Claim rewards from pool
     * @param poolId Pool identifier
     */
    function claimReward(bytes32 poolId) 
        external 
        updateReward(poolId, msg.sender) 
        nonReentrant 
    {
        RewardPool storage pool = rewardPools[poolId];
        uint256 reward = pool.rewards[msg.sender];
        
        if (reward > 0) {
            pool.rewards[msg.sender] = 0;
            _transfer(address(this), msg.sender, reward);
            emit RewardClaimed(msg.sender, poolId, reward);
        }
    }

    /**
     * @dev Calculate reward per token
     */
    function rewardPerToken(bytes32 poolId) public view returns (uint256) {
        RewardPool storage pool = rewardPools[poolId];
        
        if (pool.totalSupply == 0) {
            return pool.rewardPerTokenStored;
        }
        
        return pool.rewardPerTokenStored + 
               ((lastTimeRewardApplicable(poolId) - pool.lastUpdateTime) * 
                pool.rewardRate * 1e18) / pool.totalSupply;
    }

    /**
     * @dev Calculate earned rewards for user
     */
    function earned(bytes32 poolId, address account) public view returns (uint256) {
        RewardPool storage pool = rewardPools[poolId];
        
        return (pool.balances[account] * 
               (rewardPerToken(poolId) - pool.userRewardPerTokenPaid[account])) / 1e18 + 
               pool.rewards[account];
    }

    /**
     * @dev Get last time reward is applicable
     */
    function lastTimeRewardApplicable(bytes32 poolId) public view returns (uint256) {
        return block.timestamp;
    }

    /**
     * @dev Add distributor
     * @param distributor Distributor address
     * @param allowance Distribution allowance
     */
    function addDistributor(address distributor, uint256 allowance) external onlyOwner {
        require(distributor != address(0), "RewardToken: Invalid distributor");
        distributors[distributor] = true;
        distributionAllowance[distributor] = allowance;
        emit DistributorAdded(distributor, allowance);
    }

    /**
     * @dev Remove distributor
     * @param distributor Distributor address
     */
    function removeDistributor(address distributor) external onlyOwner {
        distributors[distributor] = false;
        distributionAllowance[distributor] = 0;
        emit DistributorRemoved(distributor);
    }

    /**
     * @dev Revoke vesting schedule
     * @param beneficiary Beneficiary address
     * @param vestingIndex Vesting schedule index
     */
    function revokeVesting(address beneficiary, uint256 vestingIndex) external onlyOwner {
        require(vestingIndex < vestingSchedules[beneficiary].length, "RewardToken: Invalid vesting index");
        
        VestingSchedule storage schedule = vestingSchedules[beneficiary][vestingIndex];
        require(!schedule.revoked, "RewardToken: Already revoked");
        
        uint256 vestedAmount = _calculateVestedAmount(schedule);
        uint256 claimableAmount = vestedAmount - schedule.claimedAmount;
        uint256 revokedAmount = schedule.totalAmount - vestedAmount;
        
        schedule.revoked = true;
        totalVested[beneficiary] -= (claimableAmount + revokedAmount);
        
        // Transfer claimable amount to beneficiary
        if (claimableAmount > 0) {
            _transfer(address(this), beneficiary, claimableAmount);
        }
        
        // Return revoked amount to owner
        if (revokedAmount > 0) {
            _transfer(address(this), owner(), revokedAmount);
        }
        
        emit VestingRevoked(beneficiary, vestingIndex);
    }

    /**
     * @dev Emergency pause
     */
    function pause() external onlyOwner {
        _pause();
    }

    /**
     * @dev Unpause
     */
    function unpause() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Get pool information
     */
    function getPoolInfo(bytes32 poolId) external view returns (
        uint256 totalRewards,
        uint256 rewardRate,
        uint256 totalSupply,
        bool active
    ) {
        RewardPool storage pool = rewardPools[poolId];
        return (pool.totalRewards, pool.rewardRate, pool.totalSupply, pool.active);
    }

    /**
     * @dev Get user balance in pool
     */
    function balanceInPool(bytes32 poolId, address account) external view returns (uint256) {
        return rewardPools[poolId].balances[account];
    }

    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, amount);
    }
}
