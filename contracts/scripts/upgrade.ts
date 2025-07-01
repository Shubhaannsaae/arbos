import { ethers, upgrades } from "hardhat";
import { Contract } from "ethers";
import { verify } from "./verify";

interface UpgradeConfig {
  contractName: string;
  proxyAddress: string;
  newImplementationArgs?: any[];
  skipVerification?: boolean;
}

async function upgradeContract(config: UpgradeConfig): Promise<Contract> {
  console.log(`🔄 Upgrading ${config.contractName}...`);
  
  const ContractFactory = await ethers.getContractFactory(config.contractName);
  
  console.log("📦 Deploying new implementation...");
  const upgraded = await upgrades.upgradeProxy(config.proxyAddress, ContractFactory);
  await upgraded.waitForDeployment();
  
  const newImplementationAddress = await upgrades.erc1967.getImplementationAddress(config.proxyAddress);
  
  console.log(`✅ ${config.contractName} upgraded successfully`);
  console.log(`📍 Proxy: ${config.proxyAddress}`);
  console.log(`📍 New Implementation: ${newImplementationAddress}`);
  
  // Verify new implementation
  if (!config.skipVerification) {
    try {
      await verify(newImplementationAddress, config.newImplementationArgs || []);
      console.log(`✅ New implementation verified`);
    } catch (error) {
      console.log(`❌ Verification failed:`, error);
    }
  }
  
  return upgraded;
}

async function upgradeArbOSCore(proxyAddress: string) {
  console.log("🔄 Upgrading ArbOSCore...");
  
  // Validate current implementation
  const currentImplementation = await upgrades.erc1967.getImplementationAddress(proxyAddress);
  console.log(`📍 Current implementation: ${currentImplementation}`);
  
  const upgraded = await upgradeContract({
    contractName: "ArbOSCore",
    proxyAddress: proxyAddress
  });
  
  // Validate upgrade
  await upgrades.validateUpgrade(proxyAddress, await ethers.getContractFactory("ArbOSCore"));
  console.log("✅ Upgrade validation passed");
  
  return upgraded;
}

async function upgradePortfolioManager(proxyAddress: string) {
  console.log("🔄 Upgrading PortfolioManager...");
  
  const upgraded = await upgradeContract({
    contractName: "PortfolioManager",
    proxyAddress: proxyAddress
  });
  
  return upgraded;
}

async function upgradeArbitrageEngine(proxyAddress: string) {
  console.log("🔄 Upgrading ArbitrageEngine...");
  
  const upgraded = await upgradeContract({
    contractName: "ArbitrageEngine", 
    proxyAddress: proxyAddress
  });
  
  return upgraded;
}

async function upgradeSecurityModule(proxyAddress: string) {
  console.log("🔄 Upgrading SecurityModule...");
  
  const upgraded = await upgradeContract({
    contractName: "SecurityModule",
    proxyAddress: proxyAddress
  });
  
  return upgraded;
}

async function upgradeBatch(upgrades: UpgradeConfig[]) {
  console.log(`🔄 Starting batch upgrade of ${upgrades.length} contracts...`);
  
  const results = [];
  
  for (const config of upgrades) {
    try {
      const upgraded = await upgradeContract(config);
      results.push({
        name: config.contractName,
        success: true,
        contract: upgraded
      });
    } catch (error) {
      console.error(`❌ Failed to upgrade ${config.contractName}:`, error);
      results.push({
        name: config.contractName,
        success: false,
        error: error
      });
    }
  }
  
  console.log("\n📊 Batch Upgrade Results:");
  console.log("==========================");
  results.forEach(result => {
    const status = result.success ? "✅" : "❌";
    console.log(`${status} ${result.name}`);
  });
  
  return results;
}

async function validateImplementation(contractName: string) {
  console.log(`🔍 Validating ${contractName} implementation...`);
  
  const ContractFactory = await ethers.getContractFactory(contractName);
  
  // Check for storage layout conflicts
  try {
    await upgrades.validateImplementation(ContractFactory);
    console.log(`✅ ${contractName} implementation is valid`);
    return true;
  } catch (error) {
    console.error(`❌ ${contractName} implementation validation failed:`, error);
    return false;
  }
}

async function forceImport(proxyAddress: string, contractName: string) {
  console.log(`🔧 Force importing ${contractName} proxy...`);
  
  const ContractFactory = await ethers.getContractFactory(contractName);
  
  try {
    await upgrades.forceImport(proxyAddress, ContractFactory);
    console.log(`✅ ${contractName} proxy imported successfully`);
  } catch (error) {
    console.error(`❌ Failed to import ${contractName} proxy:`, error);
    throw error;
  }
}

async function main() {
  const args = process.argv.slice(2);
  
  if (args.length < 1) {
    console.log(`
Usage: npx hardhat run scripts/upgrade.ts --network <network> -- <command> [args...]

Commands:
  single <contractName> <proxyAddress>  - Upgrade a single contract
  core <proxyAddress>                   - Upgrade ArbOSCore
  portfolio <proxyAddress>              - Upgrade PortfolioManager  
  arbitrage <proxyAddress>              - Upgrade ArbitrageEngine
  security <proxyAddress>               - Upgrade SecurityModule
  validate <contractName>               - Validate implementation
  import <proxyAddress> <contractName>  - Force import proxy
  batch <configFile>                    - Batch upgrade from JSON config

Examples:
  npx hardhat run scripts/upgrade.ts --network ethereum -- core 0x123...
  npx hardhat run scripts/upgrade.ts --network ethereum -- validate ArbOSCore
  npx hardhat run scripts/upgrade.ts --network ethereum -- batch upgrade-config.json
    `);
    process.exit(1);
  }
  
  const command = args[0];
  
  try {
    switch (command) {
      case "single":
        if (args.length < 3) {
          throw new Error("Usage: single <contractName> <proxyAddress>");
        }
        await upgradeContract({
          contractName: args[1],
          proxyAddress: args[2]
        });
        break;
        
      case "core":
        if (args.length < 2) {
          throw new Error("Usage: core <proxyAddress>");
        }
        await upgradeArbOSCore(args[1]);
        break;
        
      case "portfolio":
        if (args.length < 2) {
          throw new Error("Usage: portfolio <proxyAddress>");
        }
        await upgradePortfolioManager(args[1]);
        break;
        
      case "arbitrage":
        if (args.length < 2) {
          throw new Error("Usage: arbitrage <proxyAddress>");
        }
        await upgradeArbitrageEngine(args[1]);
        break;
        
      case "security":
        if (args.length < 2) {
          throw new Error("Usage: security <proxyAddress>");
        }
        await upgradeSecurityModule(args[1]);
        break;
        
      case "validate":
        if (args.length < 2) {
          throw new Error("Usage: validate <contractName>");
        }
        await validateImplementation(args[1]);
        break;
        
      case "import":
        if (args.length < 3) {
          throw new Error("Usage: import <proxyAddress> <contractName>");
        }
        await forceImport(args[1], args[2]);
        break;
        
      case "batch":
        if (args.length < 2) {
          throw new Error("Usage: batch <configFile>");
        }
        const fs = await import("fs/promises");
        const configData = await fs.readFile(args[1], "utf-8");
        const batchConfig = JSON.parse(configData);
        await upgradeBatch(batchConfig.upgrades);
        break;
        
      default:
        throw new Error(`Unknown command: ${command}`);
    }
    
    console.log("🎉 Upgrade completed successfully!");
    
  } catch (error) {
    console.error("❌ Upgrade failed:", error);
    process.exit(1);
  }
}

if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error("❌ Script failed:", error);
      process.exit(1);
    });
}

export {
  upgradeContract,
  upgradeArbOSCore,
  upgradePortfolioManager,
  upgradeArbitrageEngine,
  upgradeSecurityModule,
  upgradeBatch,
  validateImplementation,
  forceImport
};
