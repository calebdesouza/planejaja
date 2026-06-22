# NodeStor

**Soberania de IA local.** Rode qualquer modelo open-weight, em qualquer máquina, com streaming SSD→GPU, decodificação especulativa (lossless) e RAG embutido.

NodeStor é um motor de inferência em Rust + Vulkan, exposto como um pacote Python com uma CLI colorida e fácil — para desenvolvedores e para usuários leigos.

## Instalação

```bash
pip install nodestor
```

Isso instala o comando `nodestor` no seu terminal.

## Uso rápido

```bash
nodestor                       # abre o painel interativo (ótimo para começar)
nodestor help                  # lista os comandos e para que servem
nodestor scan                  # detecta GPU, VRAM, SSD e o melhor caminho SSD→GPU
nodestor run "Olá" --model meu_modelo.gguf
nodestor chat --model meu_modelo.gguf
nodestor serve --model meu_modelo.gguf   # API compatível OpenAI/Anthropic/Ollama
```

## Filosofia

- **Especulação, não quantização.** O núcleo roda o modelo com a qualidade máxima da arquitetura, sem perda. Quantização é opcional (economiza memória).
- **Qualquer máquina.** Com GPU usa Vulkan; sem GPU, cai graciosamente num caminho de CPU.
- **Aberto.** Pensado para pessoas comuns rodarem, ajustarem e melhorarem modelos open-source.

## Desenvolvimento

```bash
maturin develop --release      # compila a extensão nativa e instala em modo editável
```
