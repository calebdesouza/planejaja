use crate::fabric_server::FabricServer;
use crate::fabric_client::FabricClient;
use dashmap::DashMap;
use std::sync::Arc;
use tracing::info;

/// Registro central de todos os servidores NVMe-oF disponíveis na rede.
///
/// Em produção: seria um serviço de descoberta (etcd ou DNS-SD).
pub struct FabricRegistry {
    servers: DashMap<String, Arc<FabricServer>>,
}

impl FabricRegistry {
    pub fn new() -> Self {
        Self { servers: DashMap::new() }
    }

    pub fn register_server(&self, name: &str, server: Arc<FabricServer>) {
        info!("📋 Servidor '{}' registrado no FabricRegistry ({})", name, server.bind_addr);
        self.servers.insert(name.to_string(), server);
    }

    pub fn get_server(&self, name: &str) -> Option<Arc<FabricServer>> {
        self.servers.get(name).map(|r| Arc::clone(r.value()))
    }

    pub fn connect_client(&self, server_name: &str, client_id: &str, infiniband: bool) -> Option<FabricClient> {
        self.get_server(server_name).map(|srv| {
            FabricClient::connect(client_id, srv, infiniband)
        })
    }

    pub fn server_count(&self) -> usize {
        self.servers.len()
    }
}

impl Default for FabricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_connect() {
        let registry = FabricRegistry::new();
        let server = Arc::new(FabricServer::new("192.168.1.10:4420"));
        server.register_tensor("llama3", "test.weight", vec![0xCCu8; 1024], 0);
        
        registry.register_server("cofre-principal", server);
        assert_eq!(registry.server_count(), 1);
        
        let mut client = registry.connect_client("cofre-principal", "gpu-01", false).unwrap();
        client.mount_model("llama3");
        
        let result = client.load_tensor("test.weight").unwrap();
        assert_eq!(result.data[0], 0xCC);
        assert!(result.total_cpu_usage_pct < 10.0);
        
        println!("✅ Registry: cliente conectado e tensor carregado com CPU {:.2}%", 
            result.total_cpu_usage_pct);
    }
}
