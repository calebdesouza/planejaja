use nodestor_core::HardwareProfile;
use nodestor_vulkan::VulkanEngine;

fn main() {
    println!("Init HardwareProfile...");
    let profile = HardwareProfile {
        os: nodestor_core::OsType::Windows,
        os_version: "prod_test".into(),
        cpu_cores: 16,
        total_ram_bytes: 32 * 1024 * 1024 * 1024,
        gpus: vec![],
        storage: vec![],
        recommended_transport: nodestor_core::TransportBackend::Win32Fallback,
        missed_optimizations: vec![]
    };
    
    println!("Calling VulkanEngine::new...");
    let engine = VulkanEngine::new(&profile).expect("Failed Vulkan");
    println!("Success! Engine created.");
}


