use nodestor_inference::pipeline::{InferenceConfig, InferencePipeline};
use std::env;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: scratch_run <path_to_gguf>");
        return;
    }
    
    let path = &args[1];
    println!("Loading model from {}", path);
    
    let config = InferenceConfig {
        model_path: path.to_string(),
        prefetch_depth: 1,
        buffer_size: 1024,
    };
    
    let mut pipeline = InferencePipeline::init(config).expect("Failed to init pipeline");
    
    println!("Model initialized! Starting generation...");
    match pipeline.generate("O universo é", 10).await {
        Ok((out, stats)) => {
            println!("Output: {}", out);
            println!("Tokens: {}", stats.generated_tokens);
            println!("TPS: {}", stats.tokens_per_second);
        }
        Err(e) => {
            eprintln!("Error during generation: {:?}", e);
        }
    }
}
