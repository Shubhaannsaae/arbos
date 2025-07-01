import { run } from "hardhat";

export async function verify(contractAddress: string, args: any[]) {
  console.log("🔍 Verifying contract...");
  try {
    await run("verify:verify", {
      address: contractAddress,
      constructorArguments: args,
    });
    console.log("✅ Contract verified successfully");
  } catch (e: any) {
    if (e.message.toLowerCase().includes("already verified")) {
      console.log("✅ Contract already verified");
    } else {
      console.log("❌ Verification failed:", e);
      throw e;
    }
  }
}

// Standalone verification script
async function main() {
  const args = process.argv.slice(2);
  
  if (args.length < 1) {
    console.log("Usage: npx hardhat run scripts/verify.ts --network <network> -- <address> [constructorArg1] [constructorArg2] ...");
    process.exit(1);
  }
  
  const contractAddress = args[0];
  const constructorArgs = args.slice(1);
  
  console.log(`🔍 Verifying contract at ${contractAddress}`);
  console.log(`📋 Constructor args: ${constructorArgs.join(", ")}`);
  
  await verify(contractAddress, constructorArgs);
}

if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error("❌ Verification failed:", error);
      process.exit(1);
    });
}
