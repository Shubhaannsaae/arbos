// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {CCIPReceiver} from "@chainlink/contracts-ccip/src/v0.8/ccip/applications/CCIPReceiver.sol";
import {Client} from "@chainlink/contracts-ccip/src/v0.8/ccip/libraries/Client.sol";
import {IRouterClient} from "@chainlink/contracts-ccip/src/v0.8/ccip/interfaces/IRouterClient.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";

/**
 * @title ArbOSCCIPReceiver
 * @dev CCIP receiver for cross-chain arbitrage operations
 * @notice Handles cross-chain messages for the ArbOS ecosystem
 */
contract ArbOSCCIPReceiver is CCIPReceiver, Ownable {
    
    // Custom errors
    error NotAllowlistedSourceChain(uint64 sourceChainSelector);
    error NotAllowlistedSender(address sender);
    error InvalidReceiverAddress();

    // Events
    event MessageReceived(
        bytes32 indexed messageId,
        uint64 indexed sourceChainSelector,
        address sender,
        bytes data
    );
    
    event MessageSent(
        bytes32 indexed messageId,
        uint64 indexed destinationChainSelector,
        address receiver,
        bytes data,
        uint256 fees
    );

    // State variables
    mapping(uint64 => bool) public allowlistedSourceChains;
    mapping(address => bool) public allowlistedSenders;
    mapping(bytes32 => bool) public processedMessages;
    
    address public arbOSCore;
    
    // Gas limit for CCIP messages
    uint256 public constant GAS_LIMIT = 200_000;

    modifier onlyAllowlistedSourceChain(uint64 _sourceChainSelector) {
        if (!allowlistedSourceChains[_sourceChainSelector])
            revert NotAllowlistedSourceChain(_sourceChainSelector);
        _;
    }

    modifier onlyAllowlistedSender(address _sender) {
        if (!allowlistedSenders[_sender])
            revert NotAllowlistedSender(_sender);
        _;
    }

    modifier onlyCore() {
        require(msg.sender == arbOSCore, "CCIPReceiver: Only core");
        _;
    }

    constructor(address _router, address _arbOSCore) CCIPReceiver(_router) {
        arbOSCore = _arbOSCore;
    }

    /**
     * @notice Handles received CCIP messages
     * @param any2EvmMessage The CCIP message
     */
    function _ccipReceive(
        Client.Any2EVMMessage memory any2EvmMessage
    )
        internal
        override
        onlyAllowlistedSourceChain(any2EvmMessage.sourceChainSelector)
        onlyAllowlistedSender(abi.decode(any2EvmMessage.sender, (address)))
    {
        bytes32 messageId = any2EvmMessage.messageId;
        
        // Prevent replay attacks
        require(!processedMessages[messageId], "CCIPReceiver: Message already processed");
        processedMessages[messageId] = true;

        address sender = abi.decode(any2EvmMessage.sender, (address));
        bytes memory data = any2EvmMessage.data;

        // Process the message based on message type
        _processMessage(
            messageId,
            any2EvmMessage.sourceChainSelector,
            sender,
            data
        );

        emit MessageReceived(
            messageId,
            any2EvmMessage.sourceChainSelector,
            sender,
            data
        );
    }

    /**
     * @notice Send a message to another chain
     * @param destinationChainSelector The destination chain
     * @param receiver The receiver contract address
     * @param data The message data
     * @return messageId The CCIP message ID
     */
    function sendMessage(
        uint64 destinationChainSelector,
        address receiver,
        bytes calldata data
    ) external onlyCore returns (bytes32 messageId) {
        if (receiver == address(0)) revert InvalidReceiverAddress();

        // Create CCIP message
        Client.EVM2AnyMessage memory evm2AnyMessage = Client.EVM2AnyMessage({
            receiver: abi.encode(receiver),
            data: data,
            tokenAmounts: new Client.EVMTokenAmount[](0),
            extraArgs: Client._argsToBytes(
                Client.EVMExtraArgsV1({gasLimit: GAS_LIMIT})
            ),
            feeToken: address(0) // Pay fees in native token
        });

        // Get the fee required to send the message
        uint256 fees = IRouterClient(getRouter()).getFee(
            destinationChainSelector,
            evm2AnyMessage
        );

        require(address(this).balance >= fees, "CCIPReceiver: Insufficient balance for fees");

        // Send the message
        messageId = IRouterClient(getRouter()).ccipSend{value: fees}(
            destinationChainSelector,
            evm2AnyMessage
        );

        emit MessageSent(messageId, destinationChainSelector, receiver, data, fees);
        
        return messageId;
    }

    /**
     * @notice Send a message with token transfer
     * @param destinationChainSelector The destination chain
     * @param receiver The receiver contract address
     * @param data The message data
     * @param token The token to transfer
     * @param amount The amount to transfer
     * @return messageId The CCIP message ID
     */
    function sendMessageWithTokens(
        uint64 destinationChainSelector,
        address receiver,
        bytes calldata data,
        address token,
        uint256 amount
    ) external onlyCore returns (bytes32 messageId) {
        if (receiver == address(0)) revert InvalidReceiverAddress();

        // Transfer tokens to this contract
        IERC20(token).transferFrom(msg.sender, address(this), amount);
        IERC20(token).approve(getRouter(), amount);

        // Create token amount array
        Client.EVMTokenAmount[] memory tokenAmounts = new Client.EVMTokenAmount[](1);
        tokenAmounts[0] = Client.EVMTokenAmount({
            token: token,
            amount: amount
        });

        // Create CCIP message
        Client.EVM2AnyMessage memory evm2AnyMessage = Client.EVM2AnyMessage({
            receiver: abi.encode(receiver),
            data: data,
            tokenAmounts: tokenAmounts,
            extraArgs: Client._argsToBytes(
                Client.EVMExtraArgsV1({gasLimit: GAS_LIMIT})
            ),
            feeToken: address(0)
        });

        // Calculate and pay fees
        uint256 fees = IRouterClient(getRouter()).getFee(
            destinationChainSelector,
            evm2AnyMessage
        );

        require(address(this).balance >= fees, "CCIPReceiver: Insufficient balance for fees");

        messageId = IRouterClient(getRouter()).ccipSend{value: fees}(
            destinationChainSelector,
            evm2AnyMessage
        );

        emit MessageSent(messageId, destinationChainSelector, receiver, data, fees);
        
        return messageId;
    }

    /**
     * @dev Process received message based on type
     */
    function _processMessage(
        bytes32 messageId,
        uint64 sourceChainSelector,
        address sender,
        bytes memory data
    ) internal {
        // Decode message type from first byte
        uint8 messageType = uint8(data[0]);
        
        if (messageType == 1) {
            // Arbitrage opportunity notification
            _processArbitrageMessage(data[1:]);
        } else if (messageType == 2) {
            // Portfolio rebalancing instruction
            _processPortfolioMessage(data[1:]);
        } else if (messageType == 3) {
            // Security alert
            _processSecurityMessage(data[1:]);
        }
        // Add more message types as needed
    }

    /**
     * @dev Process arbitrage-related messages
     */
    function _processArbitrageMessage(bytes memory data) internal {
        // Decode arbitrage data and trigger local arbitrage engine
        (address tokenA, address tokenB, uint256 amount, uint256 expectedProfit) = 
            abi.decode(data, (address, address, uint256, uint256));
        
        // Forward to arbitrage engine through core contract
        if (arbOSCore != address(0)) {
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processArbitrageOpportunity(address,address,uint256,uint256)",
                    tokenA,
                    tokenB,
                    amount,
                    expectedProfit
                )
            );
            require(success, "CCIPReceiver: Arbitrage processing failed");
        }
    }

    /**
     * @dev Process portfolio-related messages
     */
    function _processPortfolioMessage(bytes memory data) internal {
        // Decode portfolio instruction
        (address user, uint256 action) = abi.decode(data, (address, uint256));
        
        // Forward to portfolio manager through core contract
        if (arbOSCore != address(0)) {
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processPortfolioInstruction(address,uint256)",
                    user,
                    action
                )
            );
            require(success, "CCIPReceiver: Portfolio processing failed");
        }
    }

    /**
     * @dev Process security-related messages
     */
    function _processSecurityMessage(bytes memory data) internal {
        // Decode security alert
        (uint256 alertLevel, string memory description) = 
            abi.decode(data, (uint256, string));
        
        // Forward to security module through core contract
        if (arbOSCore != address(0)) {
            (bool success, ) = arbOSCore.call(
                abi.encodeWithSignature(
                    "processSecurityAlert(uint256,string)",
                    alertLevel,
                    description
                )
            );
            require(success, "CCIPReceiver: Security processing failed");
        }
    }

    /**
     * @dev Allowlist a source chain
     */
    function allowlistSourceChain(uint64 _sourceChainSelector) external onlyOwner {
        allowlistedSourceChains[_sourceChainSelector] = true;
    }

    /**
     * @dev Deny a source chain
     */
    function denySourceChain(uint64 _sourceChainSelector) external onlyOwner {
        allowlistedSourceChains[_sourceChainSelector] = false;
    }

    /**
     * @dev Allowlist a sender
     */
    function allowlistSender(address _sender) external onlyOwner {
        allowlistedSenders[_sender] = true;
    }

    /**
     * @dev Deny a sender
     */
    function denySender(address _sender) external onlyOwner {
        allowlistedSenders[_sender] = false;
    }

    /**
     * @dev Update ArbOS core address
     */
    function updateArbOSCore(address _arbOSCore) external onlyOwner {
        arbOSCore = _arbOSCore;
    }

    /**
     * @dev Emergency withdraw function
     */
    function withdraw(address beneficiary) public onlyOwner {
        uint256 amount = address(this).balance;
        require(amount > 0, "CCIPReceiver: Nothing to withdraw");
        
        (bool sent, ) = beneficiary.call{value: amount}("");
        require(sent, "CCIPReceiver: Failed to withdraw");
    }

    /**
     * @dev Withdraw tokens
     */
    function withdrawToken(address beneficiary, address token) public onlyOwner {
        uint256 amount = IERC20(token).balanceOf(address(this));
        require(amount > 0, "CCIPReceiver: Nothing to withdraw");
        
        IERC20(token).transfer(beneficiary, amount);
    }

    // Receive ETH
    receive() external payable {}
}
