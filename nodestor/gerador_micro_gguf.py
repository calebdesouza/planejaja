import gguf
import numpy as np

# Configurações do Micro-Modelo (Minúsculo para não estourar nada e testar a matemática)
VOCAB_SIZE = 128
HIDDEN_SIZE = 64
FFN_SIZE = 172
NUM_HEADS = 4
NUM_KV_HEADS = 4
NUM_LAYERS = 2

print("Forjando o motor de ignição NodeStor...")

# 1. Inicializa o escritor GGUF
writer = gguf.GGUFWriter("micro_modelo_ignition.gguf", "llama")

# 2. Adiciona os Metadados exatos que o seu GraphInterpreter vai ler
writer.add_architecture()
writer.add_name("NodeStor-Micro-Ignition-FP32")
writer.add_uint32("llama.context_length", 512)
writer.add_uint32("llama.embedding_length", HIDDEN_SIZE)
writer.add_uint32("llama.block_count", NUM_LAYERS)
writer.add_uint32("llama.feed_forward_length", FFN_SIZE)
writer.add_uint32("llama.attention.head_count", NUM_HEADS)
writer.add_uint32("llama.attention.head_count_kv", NUM_KV_HEADS)
writer.add_uint32("llama.vocab_size", VOCAB_SIZE)

# 3. Forjando os Tensores FP32 (Com dados aleatórios para a matemática não zerar)
def add_tensor(name, shape):
    # Gera uma matriz de números pequenos para evitar explosão de gradiente/NaN
    data = np.random.normal(0, 0.02, shape).astype(np.float32)
    writer.add_tensor(name, data)

# Matriz de Embedding e Output
add_tensor("token_embd.weight", (VOCAB_SIZE, HIDDEN_SIZE))
add_tensor("output_norm.weight", (HIDDEN_SIZE,))
add_tensor("output.weight", (VOCAB_SIZE, HIDDEN_SIZE))

# Matrizes das 2 Camadas
for i in range(NUM_LAYERS):
    add_tensor(f"blk.{i}.attn_q.weight", (HIDDEN_SIZE, HIDDEN_SIZE))
    add_tensor(f"blk.{i}.attn_k.weight", (HIDDEN_SIZE, HIDDEN_SIZE))
    add_tensor(f"blk.{i}.attn_v.weight", (HIDDEN_SIZE, HIDDEN_SIZE))
    add_tensor(f"blk.{i}.attn_output.weight", (HIDDEN_SIZE, HIDDEN_SIZE))
    
    add_tensor(f"blk.{i}.ffn_gate.weight", (FFN_SIZE, HIDDEN_SIZE))
    add_tensor(f"blk.{i}.ffn_up.weight", (FFN_SIZE, HIDDEN_SIZE))
    add_tensor(f"blk.{i}.ffn_down.weight", (HIDDEN_SIZE, FFN_SIZE))
    
    add_tensor(f"blk.{i}.attn_norm.weight", (HIDDEN_SIZE,))
    add_tensor(f"blk.{i}.ffn_norm.weight", (HIDDEN_SIZE,))

# 4. Escreve no SSD
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print("Sucesso! 'micro_modelo_ignition.gguf' criado. Pronto para o APEX V2.")
