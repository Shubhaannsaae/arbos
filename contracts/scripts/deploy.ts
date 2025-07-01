import { ethers, network } from "hardhat";
import { Contract, ContractFactory } from "ethers";
import { verify } from "./verify";

interface DeploymentAddresses {
  arbOSCore: string;
  portfolioManager: string;
  arbitrageEngine: string;
  securityModule: string;
  chainlinkConsumer: string;
  ccipReceiver: string;
  automationConsumer: string;
  vrfConsumer: string;
  functionsConsumer: string;
  yieldOptimizer: string;
  liquidityManager: string;
  tokenSwapper: string;
  arbOSDAO: string;
  timelock: string;
  arbOSToken: string;
  rewardToken: string;
}

interface NetworkConfig {
  name: string;
  chainId: number;
  rpcUrl: string;
  chainlinkRouter?: string;
  vrfCoordinator?: string;
  functionsRouter?: string;
  subscriptionId?: string;
  keyHash?: string;
  donId?: string;
}

const networkConfigs: { [key: string]: NetworkConfig } = {
  ethereum: {
    name: "Ethereum Mainnet",
    chainId: 1,
    rpcUrl: "https://eth.llamarpc.com",
    chainlinkRouter: "0x80226fc0Ee2b096224EeAc085Bb9a8cba1146f7D",
    vrfCoordinator: "0x271682DEB8C4E0901D1a1550aD2e64D568E69909",
    functionsRouter: "0x65C939B26d3d949A6E2bE41B1F5659dB13b5f4a",
    keyHash: "0x8af398995b04c28e9951adb9721ef74c74f93e6a478f39e7e0777be13527e7ef",
    donId: "0x66756e2d657468657265756d2d6d61696e6e65742d31000000000000000000"
  },
  avalanche: {
    name: "Avalanche Mainnet",
    chainId: 43114,
    rpcUrl: "https://api.avax.network/ext/bc/C/rpc",
    chainlinkRouter: "0xF4c7E640EdA248ef95972845a62bdC74237805dB",
    vrfCoordinator: "0xd5D517aBE5cF79B7e95eC98dB0f0277788aFF634",
    functionsRouter: "0x3c82F31d0b6e267eF9b5b5e5d2B9A2C8a2F4c5e7",
    keyHash: "0x354d2f95da55398f44b7cff77da56283d9c6c829a4bdf1bbcaf2ad6a4d081f61",
    donId: "0x66756e2d6176616c616e6368652d6d61696e6e65742d31000000000000000000"
  },
  polygon: {
    name: "Polygon Mainnet", 
    chainId: 137,
    rpcUrl: "https://polygon-rpc.com",
    chainlinkRouter: "0x3C3D92629A02a8D95D5CB9650fe49C3544f69B43",
    vrfCoordinator: "0xAE975071Be8F8eE67addBC1A82488F1C24858067",
    functionsRouter: "0x4f8a84C442F9675610c680990EdDb2CCDDB2E906",
    keyHash: "0xcc294a196eeeb44da2888d17c0625cc88d70d9760a69d58d853ba6581a9ab0cd",
    donId: "0x66756e2d706f6c79676f6e2d6d61696e6e65742d31000000000000000000000"
  }
};

async function main() {
  console.log("🚀 Starting ArbOS deployment...");
  
  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  const networkName = network.name;
  const config = networkConfigs[networkName];
  
  if (!config) {
    throw new Error(`❌ Network ${networkName} not supported`);
  }
  
  console.log(`📍 Deploying to ${config.name} (${config.chainId})`);
  console.log(`💰 Deployer: ${deployer.address}`);
  console.log(`💸 Balance: ${ethers.formatEther(await ethers.provider.getBalance(deployer.address))} ETH`);
  
  const deploymentAddresses: Partial<DeploymentAddresses> = {};
  
  // Deploy Tokens First
  console.log("\n🪙 Deploying Tokens...");
  
  const ArbOSToken = await ethers.getContractFactory("ArbOSToken");
  const arbOSToken = await ArbOSToken.deploy();
  await arbOSToken.waitForDeployment();
  deploymentAddresses.arbOSToken = await arbOSToken.getAddress();
  console.log(`✅ ArbOSToken deployed: ${deploymentAddresses.arbOSToken}`);
  
  const RewardToken = await ethers.getContractFactory("RewardToken");
  const rewardToken = await RewardToken.deploy();
  await rewardToken.waitForDeployment();
  deploymentAddresses.rewardToken = await rewardToken.getAddress();
  console.log(`✅ RewardToken deployed: ${deploymentAddresses.rewardToken}`);
  
  // Deploy Governance
  console.log("\n🏛️ Deploying Governance...");
  
  const Timelock = await ethers.getContractFactory("Timelock");
  const timelock = await Timelock.deploy(
    deployer.address, // admin
    172800 // 2 days delay
  );
  await timelock.waitForDeployment();
  deploymentAddresses.timelock = await timelock.getAddress();
  console.log(`✅ Timelock deployed: ${deploymentAddresses.timelock}`);
  
  const ArbOSDAO = await ethers.getContractFactory("ArbOSDAO");
  const arbOSDAO = await ArbOSDAO.deploy(
    deploymentAddresses.arbOSToken,
    deploymentAddresses.timelock,
    ethers.ZeroAddress, // Will be updated later
    {
      votingDelay: 1, // 1 block
      votingPeriod: 45818, // ~1 week
      proposalThreshold: ethers.parseEther("100000"), // 100k tokens
      quorumNumerator: 4, // 4%
      quorumDenominator: 100,
      timelockDelay: 172800 // 2 days
    }
  );
  await arbOSDAO.waitForDeployment();
  deploymentAddresses.arbOSDAO = await arbOSDAO.getAddress();
  console.log(`✅ ArbOSDAO deployed: ${deploymentAddresses.arbOSDAO}`);
  
  // Deploy Chainlink Services
  console.log("\n🔗 Deploying Chainlink Services...");
  
  const ChainlinkConsumer = await ethers.getContractFactory("ChainlinkConsumer");
  const chainlinkConsumer = await ChainlinkConsumer.deploy();
  await chainlinkConsumer.waitForDeployment();
  deploymentAddresses.chainlinkConsumer = await chainlinkConsumer.getAddress();
  console.log(`✅ ChainlinkConsumer deployed: ${deploymentAddresses.chainlinkConsumer}`);
  
  if (config.chainlinkRouter) {
    const CCIPReceiver = await ethers.getContractFactory("ArbOSCCIPReceiver");
    const ccipReceiver = await CCIPReceiver.deploy(
      config.chainlinkRouter,
      ethers.ZeroAddress // Will be updated later
    );
    await ccipReceiver.waitForDeployment();
    deploymentAddresses.ccipReceiver = await ccipReceiver.getAddress();
    console.log(`✅ CCIPReceiver deployed: ${deploymentAddresses.ccipReceiver}`);
  }
  
  if (config.vrfCoordinator && config.keyHash) {
    const VRFConsumer = await ethers.getContractFactory("VRFConsumer");
    const vrfConsumer = await VRFConsumer.deploy(
      config.vrfCoordinator,
      1, // subscription ID (update with actual)
      config.keyHash
    );
    await vrfConsumer.waitForDeployment();
    deploymentAddresses.vrfConsumer = await vrfConsumer.getAddress();
    console.log(`✅ VRFConsumer deployed: ${deploymentAddresses.vrfConsumer}`);
  }
  
  if (config.functionsRouter && config.donId) {
    const FunctionsConsumer = await ethers.getContractFactory("FunctionsConsumer");
    const functionsConsumer = await FunctionsConsumer.deploy(
      config.functionsRouter,
      1, // subscription ID (update with actual)
      config.donId
    );
    await functionsConsumer.waitForDeployment();
    deploymentAddresses.functionsConsumer = await functionsConsumer.getAddress();
    console.log(`✅ FunctionsConsumer deployed: ${deploymentAddresses.functionsConsumer}`);
  }
  
  // Deploy Core Contracts
  console.log("\n🏗️ Deploying Core Contracts...");
  
  const treasury = deployer.address; // Use deployer as treasury for now
  
  const ArbOSCore = await ethers.getContractFactory("ArbOSCore");
  const arbOSCore = await ArbOSCore.deploy(
    treasury,
    ethers.ZeroAddress, // arbitrageEngine - will be set later
    ethers.ZeroAddress, // portfolioManager - will be set later
    ethers.ZeroAddress  // securityModule - will be set later
  );
  await arbOSCore.waitForDeployment();
  deploymentAddresses.arbOSCore = await arbOSCore.getAddress();
  console.log(`✅ ArbOSCore deployed: ${deploymentAddresses.arbOSCore}`);
  
  const PortfolioManager = await ethers.getContractFactory("PortfolioManager");
  const portfolioManager = await PortfolioManager.deploy(
    deploymentAddresses.chainlinkConsumer,
    deploymentAddresses.arbOSCore
  );
  await portfolioManager.waitForDeployment();
  deploymentAddresses.portfolioManager = await portfolioManager.getAddress();
  console.log(`✅ PortfolioManager deployed: ${deploymentAddresses.portfolioManager}`);
  
  const ArbitrageEngine = await ethers.getContractFactory("ArbitrageEngine");
  const arbitrageEngine = await ArbitrageEngine.deploy(
    deploymentAddresses.chainlinkConsumer,
    deploymentAddresses.arbOSCore
  );
  await arbitrageEngine.waitForDeployment();
  deploymentAddresses.arbitrageEngine = await arbitrageEngine.getAddress();
  console.log(`✅ ArbitrageEngine deployed: ${deploymentAddresses.arbitrageEngine}`);
  
  const SecurityModule = await ethers.getContractFactory("SecurityModule");
  const securityModule = await SecurityModule.deploy(
    deploymentAddresses.arbOSCore
  );
  await securityModule.waitForDeployment();
  deploymentAddresses.securityModule = await securityModule.getAddress();
  console.log(`✅ SecurityModule deployed: ${deploymentAddresses.securityModule}`);
  
  const AutomationConsumer = await ethers.getContractFactory("AutomationConsumer");
  const automationConsumer = await AutomationConsumer.deploy(
    deploymentAddresses.arbOSCore,
    deploymentAddresses.portfolioManager,
    deploymentAddresses.arbitrageEngine,
    deploymentAddresses.securityModule
  );
  await automationConsumer.waitForDeployment();
  deploymentAddresses.automationConsumer = await automationConsumer.getAddress();
  console.log(`✅ AutomationConsumer deployed: ${deploymentAddresses.automationConsumer}`);
  
  // Deploy DeFi Contracts
  console.log("\n💰 Deploying DeFi Contracts...");
  
  const YieldOptimizer = await ethers.getContractFactory("YieldOptimizer");
  const yieldOptimizer = await YieldOptimizer.deploy(
    treasury,
    deploymentAddresses.arbOSCore
  );
  await yieldOptimizer.waitForDeployment();
  deploymentAddresses.yieldOptimizer = await yieldOptimizer.getAddress();
  console.log(`✅ YieldOptimizer deployed: ${deploymentAddresses.yieldOptimizer}`);
  
  const LiquidityManager = await ethers.getContractFactory("LiquidityManager");
  const liquidityManager = await LiquidityManager.deploy(
    deploymentAddresses.arbOSCore,
    "0x1F98431c8aD98523631AE4a59f267346ea31F984", // Uniswap V3 Factory
    "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"  // Uniswap V3 Position Manager
  );
  await liquidityManager.waitForDeployment();
  deploymentAddresses.liquidityManager = await liquidityManager.getAddress();
  console.log(`✅ LiquidityManager deployed: ${deploymentAddresses.liquidityManager}`);
  
  const TokenSwapper = await ethers.getContractFactory("TokenSwapper");
  const tokenSwapper = await TokenSwapper.deploy(
    deploymentAddresses.arbOSCore,
    treasury
  );
  await tokenSwapper.waitForDeployment();
  deploymentAddresses.tokenSwapper = await tokenSwapper.getAddress();
  console.log(`✅ TokenSwapper deployed: ${deploymentAddresses.tokenSwapper}`);
  
  // Configuration
  console.log("\n⚙️ Configuring contracts...");
  
  // Update ArbOSCore with module addresses
  await arbOSCore.updateModule("arbitrage", deploymentAddresses.arbitrageEngine);
  await arbOSCore.updateModule("portfolio", deploymentAddresses.portfolioManager);
  await arbOSCore.updateModule("security", deploymentAddresses.securityModule);
  await arbOSCore.updateModule("chainlink", deploymentAddresses.chainlinkConsumer);
  console.log("✅ ArbOSCore modules configured");
  
  // Update DAO with ArbOSCore address
  await arbOSDAO.updateTimelock(deploymentAddresses.timelock);
  console.log("✅ DAO configured");
  
  // Update Chainlink services
  if (deploymentAddresses.ccipReceiver && deploymentAddresses.vrfConsumer && deploymentAddresses.functionsConsumer) {
    await chainlinkConsumer.updateServiceContracts(
      deploymentAddresses.ccipReceiver,
      deploymentAddresses.automationConsumer,
      deploymentAddresses.vrfConsumer,
      deploymentAddresses.functionsConsumer
    );
    console.log("✅ Chainlink services configured");
  }
  
  // Save deployment addresses
  const fs = await import("fs/promises");
  const deploymentData = {
    network: networkName,
    chainId: config.chainId,
    timestamp: new Date().toISOString(),
    deployer: deployer.address,
    addresses: deploymentAddresses
  };
  
  await fs.writeFile(
    `deployments/${networkName}.json`,
    JSON.stringify(deploymentData, null, 2)
  );
  
  console.log("\n📄 Deployment Summary:");
  console.log("========================");
  Object.entries(deploymentAddresses).forEach(([name, address]) => {
    console.log(`${name}: ${address}`);
  });
  
  console.log(`\n💾 Deployment saved to deployments/${networkName}.json`);
  
  // Verify contracts if not on hardhat network
  if (network.name !== "hardhat" && network.name !== "localhost") {
    console.log("\n🔍 Starting contract verification...");
    for (const [name, address] of Object.entries(deploymentAddresses)) {
      if (address) {
        try {
          await verify(address, []);
          console.log(`✅ ${name} verified`);
        } catch (error) {
          console.log(`❌ ${name} verification failed:`, error);
        }
      }
    }
  }
  
  console.log("\n🎉 Deployment completed successfully!");
}

// Error handling
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("❌ Deployment failed:", error);
    process.exit(1);
  });
