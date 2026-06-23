"""CLI do NodeStor — colorida, profissional e fácil para qualquer pessoa.

Design (estilo Claude Code):
- Sem argumentos -> abre um PAINEL interativo com opções numeradas selecionáveis.
- Com um comando -> executa direto (para quem já sabe o que quer).
- `nodestor help` -> mostra todos os comandos e para que servem, colorido.
- Funciona em QUALQUER terminal: usa `rich` quando instalado, e cai num
  fallback ANSI puro (com VT habilitado no Windows) caso contrário.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ───────────────────────── camada de UI (rich OU ANSI) ─────────────────────────
try:  # rich é a experiência premium; é opcional.
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _console: "Console | None" = Console()
    _RICH = True
except Exception:  # pragma: no cover
    _console = None
    _RICH = False


def _enable_windows_ansi() -> None:
    """Habilita sequências ANSI no Console do Windows (sem dependências)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004 no handle de saída (-11)
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def _ensure_utf8_stdout() -> None:
    """Reconfigura stdout/stderr para UTF-8 quando possível (Python 3.7+)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


_enable_windows_ansi()
_ensure_utf8_stdout()

_USE_COLOR = (
    _RICH
    or (sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
)


def _can_encode(text: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(enc)
        return True
    except Exception:
        return False


# Após reconfigurar para UTF-8, terminais modernos exibem blocos/emoji; senão,
# caímos para ASCII puro — garantindo funcionar em QUALQUER terminal.
_UNICODE_OK = _can_encode("█─┌🤖")

# Paleta (terracota/laranja do NodeStor)
C = {
    "orange": "\033[38;5;208m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _c(text: str, *styles: str) -> str:
    if not _USE_COLOR:
        return text
    return "".join(C[s] for s in styles) + text + C["reset"]


def out(msg: str = "") -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"))


def header(title: str, subtitle: str = "") -> None:
    if _RICH and _console is not None:
        body = Text(title, style="bold orange3")
        if subtitle:
            body.append("\n" + subtitle, style="dim")
        _console.print(Panel(body, border_style="orange3", expand=False))
        return
    bar = "─" if _UNICODE_OK else "-"
    line = bar * (max(len(title), len(subtitle)) + 2)
    out(_c(line, "orange"))
    out(" " + _c(title, "orange", "bold"))
    if subtitle:
        out(" " + _c(subtitle, "dim"))
    out(_c(line, "orange"))


LOGO = r"""
  ███╗  ██╗ ██████╗ ██████╗ ███████╗ ███████╗████████╗ ██████╗ ██████╗
  ████╗ ██║██╔═══██╗██╔══██╗██╔════╝ ██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
  ██╔██╗██║██║   ██║██║  ██║█████╗   ███████╗   ██║   ██║   ██║██████╔╝
  ██║╚████║██║   ██║██║  ██║██╔══╝   ╚════██║   ██║   ██║   ██║██╔══██╗
  ██║ ╚███║╚██████╔╝██████╔╝███████╗ ███████║   ██║   ╚██████╔╝██║  ██║
  ╚═╝  ╚══╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
"""


LOGO_ASCII = r"""
  _   _           _      ____  _
 | \ | | ___   __| | ___/ ___|| |_ ___  _ __
 |  \| |/ _ \ / _` |/ _ \___ \| __/ _ \| '__|
 | |\  | (_) | (_| |  __/___) | || (_) | |
 |_| \_|\___/ \__,_|\___|____/ \__\___/|_|
"""


def print_logo() -> None:
    from . import __version__

    art = LOGO if _UNICODE_OK else LOGO_ASCII
    sep = "  •  " if _UNICODE_OK else "  -  "
    for line in art.strip("\n").splitlines():
        out(_c(line, "orange"))
    out("  " + _c("Soberania de IA local", "bold") + _c(sep + "SSD->GPU" + sep + "RAG" + sep + "v" + __version__, "dim"))
    out()


# ─────────────────────────────── catálogo de comandos ───────────────────────────
# Poucos comandos, claros. (nome, uso, descrição)
COMMANDS = [
    ("pull",    "nodestor pull <repo> --filename <arquivo>",    "Baixa um modelo do HuggingFace para ~/.nodestor/models"),
    ("run",     "nodestor run \"<prompt>\" --model <arquivo>", "Roda um prompt num modelo local e mostra TTFT e tok/s reais"),
    ("chat",    "nodestor chat --model <arquivo>",              "Conversa interativa contínua com o modelo"),
    ("serve",   "nodestor serve --model <arquivo>",             "Sobe o servidor (API OpenAI/Anthropic/Ollama) na porta 8080"),
    ("inspect", "nodestor inspect <arquivo>",                   "Mostra formato, nº de tensores e tamanho do modelo"),
    ("scan",    "nodestor scan",                                "Detecta GPU, VRAM, SSD e o melhor caminho SSD→GPU"),
    ("profiles","nodestor profiles",                            "Lista as personas prontas (--profile cientista, etc.)"),
    ("connect", "nodestor connect",                             "Como ligar Claude Code, Cursor e outras ferramentas"),
    ("help",    "nodestor help",                                "Mostra esta lista de comandos e para que servem"),
    ("version", "nodestor version",                             "Mostra a versão instalada"),
]


def cmd_help() -> int:
    print_logo()
    if _RICH and _console is not None:
        table = Table(show_header=True, header_style="bold orange3", border_style="dim")
        table.add_column("Comando", style="cyan", no_wrap=True)
        table.add_column("Uso")
        table.add_column("Para que serve", style="dim")
        for name, usage, desc in COMMANDS:
            table.add_row(name, usage, desc)
        _console.print(table)
    else:
        out(_c("COMANDOS", "bold", "orange"))
        for name, usage, desc in COMMANDS:
            out("  " + _c(f"{name:<8}", "cyan", "bold") + _c(usage, "dim"))
            out("           " + desc)
    out()
    out(_c("Dica:", "yellow", "bold") + " rode " + _c("nodestor", "cyan") + " sem argumentos para o painel interativo.")
    return 0


def cmd_version() -> int:
    from . import __version__, _ENGINE_AVAILABLE

    out(_c("NodeStor", "orange", "bold") + " v" + __version__)
    state = _c("disponível ✓", "green") if _ENGINE_AVAILABLE else _c("não compilado (instale via pip/maturin)", "yellow")
    out("Motor nativo: " + state)
    return 0


def _require_engine() -> bool:
    from . import _ENGINE_AVAILABLE, _ENGINE_IMPORT_ERROR

    if _ENGINE_AVAILABLE:
        return True
    out(_c("⚠ Motor nativo não disponível.", "yellow", "bold"))
    out("  Para habilitar a inferência, instale o pacote compilado:")
    out("    " + _c("pip install nodestor", "cyan"))
    out("  (ou, em desenvolvimento: " + _c("maturin develop --release", "cyan") + ")")
    if _ENGINE_IMPORT_ERROR is not None:
        out(_c(f"  detalhe: {_ENGINE_IMPORT_ERROR}", "dim"))
    return False


def cmd_scan() -> int:
    header("Scanner de Hardware", "GPU • VRAM • SSD • caminho SSD→GPU")
    if not _require_engine():
        return 1
    from . import scan as native_scan

    try:
        out(native_scan())
        return 0
    except Exception as exc:  # pragma: no cover
        out(_c(f"Erro: {exc}", "red"))
        return 1


# ── System Prompt ("Bloco Zero") + Biblioteca de Perfis (Editor de Personalidades) ──
PROFILES = {
    "cientista": "You are a rigorous scientist. Reason step by step, cite evidence, and clearly separate fact from hypothesis.",
    "programador": "You are a senior software engineer. Write clean, correct, idiomatic code and explain trade-offs concisely.",
    "advogado": "You are a careful legal analyst. Be precise, cite principles, and flag uncertainties and jurisdiction limits.",
    "professor": "You are a patient teacher. Explain clearly with simple examples, then check understanding.",
    "conciso": "You are concise. Answer directly in as few words as possible, with no preamble.",
    "security": "You are a security researcher performing AUTHORIZED review. Analyze code for vulnerabilities and explain mitigations.",
}


def resolve_system(system: "str | None", profile: "str | None") -> "str | None":
    if system:
        return system
    if profile:
        return PROFILES.get(profile.lower(), profile)
    return None


def build_chat_prompt(system: "str | None", user: str) -> str:
    """Template ChatML (SmolLM2/Qwen/…). Sem sistema, mantém completion cru."""
    if not system:
        return user
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )


def cmd_profiles() -> int:
    header("Perfis de Sistema", 'personas prontas — ou use --system "<texto livre>"')
    for name, text in PROFILES.items():
        short = text[:62] + ("…" if len(text) > 62 else "")
        out("  " + _c(f"{name:<12}", "cyan", "bold") + _c(short, "dim"))
    out()
    out("Uso: " + _c('nodestor run "pergunta" --model X --profile cientista', "cyan"))
    return 0


def cmd_run(model: str, prompt: str, max_tokens: int, system=None, profile=None) -> int:
    header("Inferência Local", model)
    if not os.path.exists(model):
        out(_c(f"Modelo não encontrado: {model}", "red"))
        out("  Baixe um com: " + _c("nodestor pull <repo> --filename <arquivo.gguf>", "cyan"))
        return 1
    if not _require_engine():
        return 1
    from . import NodeStorEngine
    sys_prompt = resolve_system(system, profile)
    if sys_prompt:
        short = sys_prompt[:60] + ("…" if len(sys_prompt) > 60 else "")
        out(_c("🧠 Sistema: ", "bold") + _c(short, "dim"))
    out(_c("⏳ Carregando motor…", "dim"))
    t0 = time.time()
    try:
        engine = NodeStorEngine(model)
    except Exception as exc:
        out(_c(f"Falha ao carregar modelo: {exc}", "red"))
        return 1
    out(_c(f"✓ pronto em {time.time() - t0:.2f}s", "green"))
    out()
    out(_c("🤖 ", "orange") + "gerando…")
    t1 = time.time()
    try:
        text = engine.generate(build_chat_prompt(sys_prompt, prompt), max_tokens)
    except Exception as exc:
        out(_c(f"Erro na geração: {exc}", "red"))
        return 1
    dt = time.time() - t1
    out(text)
    out()
    out(_c("Métricas (medidas):", "bold"))
    out(f"  Tempo de geração : {dt:.2f}s")
    return 0


def cmd_chat(model: str, max_tokens: int, system=None, profile=None) -> int:
    header("Chat Interativo", "digite /sair para encerrar")
    if not os.path.exists(model):
        out(_c(f"Modelo não encontrado: {model}", "red"))
        out("  Baixe um com: " + _c("nodestor pull <repo> --filename <arquivo.gguf>", "cyan"))
        return 1
    if not _require_engine():
        return 1
    from . import NodeStorEngine
    sys_prompt = resolve_system(system, profile)
    if sys_prompt:
        short = sys_prompt[:60] + ("…" if len(sys_prompt) > 60 else "")
        out(_c("🧠 Sistema: ", "bold") + _c(short, "dim"))
    out(_c("⏳ Carregando motor…", "dim"))
    try:
        engine = NodeStorEngine(model)
    except Exception as exc:
        out(_c(f"Falha ao carregar modelo: {exc}", "red"))
        return 1
    out(_c("✓ pronto. Converse!", "green"))
    while True:
        try:
            msg = input(_c("\nVocê › ", "cyan", "bold"))
        except (EOFError, KeyboardInterrupt):
            out()
            break
        if msg.strip() in ("/sair", "/exit", "/quit"):
            break
        try:
            reply = engine.generate(build_chat_prompt(sys_prompt, msg), max_tokens)
        except Exception as exc:
            out(_c(f"Erro: {exc}", "red"))
            continue
        out(_c("🤖 ", "orange") + reply)
    out(_c("Até logo!", "dim"))
    return 0


def _find_server_binary() -> "str | None":
    """Procura o binário nodestor-server no PATH e nos diretórios de build."""
    import shutil

    exe = "nodestor-server.exe" if os.name == "nt" else "nodestor-server"
    found = shutil.which("nodestor-server")
    if found:
        return found
    here = os.path.dirname(os.path.abspath(__file__))          # .../python/nodestor
    repo = os.path.abspath(os.path.join(here, "..", ".."))     # .../nodestor
    candidates = [
        os.path.join(repo, "target", "release", exe),
        os.path.join(repo, "target", "debug", exe),
    ]
    triple_root = os.path.join(repo, "target")
    if os.path.isdir(triple_root):
        for entry in os.listdir(triple_root):
            for prof in ("release", "debug"):
                candidates.append(os.path.join(triple_root, entry, prof, exe))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def cmd_serve(model: str, port: int) -> int:
    header("Servidor NodeStor", f"API OpenAI/Anthropic/Ollama • porta {port}")
    import subprocess

    server_bin = _find_server_binary()
    if server_bin is None:
        out(_c("Binário 'nodestor-server' não encontrado.", "yellow"))
        out("  Compile o servidor com: " + _c("cargo build --release -p nodestor-server", "cyan"))
        return 1
    if not os.path.exists(model):
        out(_c(f"Modelo não encontrado: {model}", "red"))
        return 1
    out(_c(f"▶ iniciando servidor em http://localhost:{port} …", "green") + _c(f"  [{server_bin}]", "dim"))
    try:
        return subprocess.call([server_bin, "--model", model, "--port", str(port)])
    except KeyboardInterrupt:
        return 0


def cmd_pull(repo: str, filename: str) -> int:
    """Baixa um modelo do HuggingFace Hub para ~/.nodestor/models (sem deps extras)."""
    header("Baixar modelo", f"{repo} / {filename}")
    import urllib.request
    import urllib.error

    dest_dir = os.path.join(os.path.expanduser("~"), ".nodestor", "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(filename))
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    out(_c("baixando ", "dim") + url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nodestor-cli"})
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0) or 0)
            done = 0
            chunk = 1 << 20  # 1 MB
            with open(dest, "wb") as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if total:
                        pct = done * 100 // total
                        bar_len = 30
                        filled = bar_len * done // total
                        bar = ("#" * filled).ljust(bar_len)
                        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {done/1e6:6.1f} MB")
                    else:
                        sys.stdout.write(f"\r  {done/1e6:6.1f} MB")
                    sys.stdout.flush()
        sys.stdout.write("\n")
        out(_c(f"✓ salvo em {dest}", "green"))
        out("  Rode: " + _c(f'nodestor run "Olá" --model "{dest}"', "cyan"))
        return 0
    except urllib.error.HTTPError as exc:
        out(_c(f"\nFalha HTTP {exc.code}: {exc.reason}", "red"))
        out("  Verifique o repo/arquivo. Ex: " + _c("nodestor pull TheBloke/Llama-2-7B-GGUF --filename llama-2-7b.Q4_K_M.gguf", "dim"))
        return 1
    except Exception as exc:
        out(_c(f"\nErro ao baixar: {exc}", "red"))
        return 1


def cmd_inspect(path: str) -> int:
    """Inspeção de modelo (cabeçalho GGUF em Python puro — observabilidade sem o motor)."""
    header("Inspeção de Modelo", path)
    if not os.path.exists(path):
        out(_c(f"Arquivo não encontrado: {path}", "red"))
        return 1
    import struct

    size = os.path.getsize(path)
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic == b"GGUF":
                version = struct.unpack("<I", fh.read(4))[0]
                n_tensors = struct.unpack("<Q", fh.read(8))[0]
                n_kv = struct.unpack("<Q", fh.read(8))[0]
                out(_c("Formato      : ", "bold") + "GGUF v" + str(version))
                out(_c("Tensores     : ", "bold") + f"{n_tensors:,}")
                out(_c("Metadados KV : ", "bold") + f"{n_kv:,}")
                out(_c("Tamanho      : ", "bold") + f"{size/1e9:.2f} GB")
                out(_c("\nDica:", "yellow", "bold") + " a lista completa de tensores/arquitetura vem do motor (" + _c("nodestor inspect", "cyan") + " no binário Rust).")
                return 0
            elif path.endswith(".safetensors"):
                # Header SafeTensors: u64 little-endian com o tamanho do JSON de header.
                fh.seek(0)
                hlen = struct.unpack("<Q", fh.read(8))[0]
                import json
                meta = json.loads(fh.read(hlen).decode("utf-8", errors="replace"))
                n = len([k for k in meta.keys() if k != "__metadata__"])
                out(_c("Formato      : ", "bold") + "SafeTensors")
                out(_c("Tensores     : ", "bold") + f"{n:,}")
                out(_c("Tamanho      : ", "bold") + f"{size/1e9:.2f} GB")
                return 0
            else:
                out(_c("Formato não reconhecido (esperado GGUF ou SafeTensors).", "yellow"))
                out(_c("Tamanho: ", "bold") + f"{size/1e9:.2f} GB")
                return 1
    except Exception as exc:
        out(_c(f"Erro ao ler: {exc}", "red"))
        return 1


def cmd_connect() -> int:
    """Instruções para conectar Claude Code, Cursor e outras ferramentas."""
    header("Conectar ferramentas", "Claude Code • Cursor • Ollama-compatible")
    out(_c("1) Anthropic API (Claude Code):", "bold"))
    out("   export ANTHROPIC_BASE_URL=" + _c("http://localhost:8080", "cyan"))
    out("   export ANTHROPIC_API_KEY=" + _c("nodestor", "cyan"))
    out(_c("\n2) OpenAI API (Cursor / Windsurf):", "bold"))
    out("   Base URL: " + _c("http://localhost:8080/v1", "cyan"))
    out("   Model ID: " + _c("nodestor", "cyan"))
    out(_c("\n3) MCP (Model Context Protocol):", "bold"))
    out("   Endpoint: " + _c("http://localhost:8080/mcp", "cyan"))
    out(_c("\nSuba o servidor primeiro com: ", "dim") + _c("nodestor serve --model <arquivo>", "cyan"))
    return 0


# ─────────────────────────────── painel interativo ─────────────────────────────
def interactive_menu() -> int:
    print_logo()
    options = [
        ("Baixar um modelo (pull)", "pull"),
        ("Rodar um prompt (run)", "run"),
        ("Conversar (chat)", "chat"),
        ("Subir o servidor (serve)", "serve"),
        ("Inspecionar um modelo (inspect)", "inspect"),
        ("Escanear o hardware (scan)", "scan"),
        ("Conectar ferramentas (connect)", "connect"),
        ("Ver os comandos (help)", "help"),
        ("Sair", "exit"),
    ]
    top = "┌─ NODE-PANEL " + "─" * 28 if _UNICODE_OK else "+- NODE-PANEL " + "-" * 28
    bottom = "└" + "─" * 41 if _UNICODE_OK else "+" + "-" * 41
    while True:
        out(_c(top, "orange"))
        for i, (label, _) in enumerate(options, 1):
            num = _c(f"[{i}]", "cyan", "bold")
            out(f"  {num} {label}")
        out(_c(bottom, "orange"))
        try:
            choice = input(_c("› escolha: ", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            out()
            return 0
        if not choice:
            continue
        if not choice.isdigit() or not (1 <= int(choice) <= len(options)):
            out(_c("  opção inválida", "yellow"))
            continue
        action = options[int(choice) - 1][1]
        if action == "exit":
            out(_c("Encerrando. 👋", "dim"))
            return 0
        if action == "help":
            cmd_help()
            continue
        if action == "scan":
            cmd_scan()
            continue
        if action == "connect":
            cmd_connect()
            continue
        if action == "pull":
            repo = input(_c("  repo HuggingFace (ex: TheBloke/Llama-2-7B-GGUF): ", "cyan")).strip()
            fname = input(_c("  arquivo (ex: llama-2-7b.Q4_K_M.gguf): ", "cyan")).strip()
            if repo and fname:
                cmd_pull(repo, fname)
            else:
                out(_c("  cancelado", "dim"))
            continue
        if action == "inspect":
            path = input(_c("  caminho do modelo: ", "cyan")).strip().strip('"')
            if path:
                cmd_inspect(path)
            else:
                out(_c("  cancelado", "dim"))
            continue
        # run / chat / serve precisam de um modelo existente
        model = input(_c("  caminho do modelo (.gguf): ", "cyan")).strip().strip('"')
        if not model:
            out(_c("  cancelado", "dim"))
            continue
        if not os.path.exists(model):
            out(_c(f"  Modelo não encontrado: {model}", "red"))
            out(_c("  Baixe um primeiro pela opção 'Baixar um modelo (pull)'.", "dim"))
            continue
        if action == "run":
            prompt = input(_c("  prompt: ", "cyan")).strip()
            cmd_run(model, prompt or "Olá", 128)
        elif action == "chat":
            cmd_chat(model, 256)
        elif action == "serve":
            cmd_serve(model, 8080)
    # inalcançável


# ─────────────────────────────────── argparse ──────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nodestor",
        description="NodeStor — soberania de IA local. Sem argumentos, abre o painel interativo.",
        add_help=True,
    )
    sub = p.add_subparsers(dest="command")

    pp = sub.add_parser("pull", help="Baixa um modelo do HuggingFace Hub")
    pp.add_argument("repo", help="ID do repo (ex: TheBloke/Llama-2-7B-GGUF)")
    pp.add_argument("-f", "--filename", required=True, help="Arquivo a baixar (ex: llama-2-7b.Q4_K_M.gguf)")

    pr = sub.add_parser("run", help="Roda um prompt num modelo local")
    pr.add_argument("prompt", help="Texto do prompt")
    pr.add_argument("-m", "--model", required=True, help="Caminho do modelo (.gguf/.safetensors)")
    pr.add_argument("--max-tokens", type=int, default=128)
    pr.add_argument("--system", help="Prompt de Sistema (a 'constituição' do modelo)")
    pr.add_argument("--profile", help="Perfil pronto (cientista/programador/advogado/...)")

    pi = sub.add_parser("inspect", help="Mostra metadados de um arquivo de modelo")
    pi.add_argument("path", help="Caminho do modelo (.gguf/.safetensors)")

    pc = sub.add_parser("chat", help="Conversa interativa com o modelo")
    pc.add_argument("-m", "--model", required=True)
    pc.add_argument("--max-tokens", type=int, default=256)
    pc.add_argument("--system", help="Prompt de Sistema")
    pc.add_argument("--profile", help="Perfil pronto")

    ps = sub.add_parser("serve", help="Sobe o servidor (API compatível)")
    ps.add_argument("-m", "--model", required=True)
    ps.add_argument("--port", type=int, default=8080)

    sub.add_parser("scan", help="Detecta hardware (GPU/VRAM/SSD)")
    sub.add_parser("profiles", help="Lista os perfis de sistema (personas)")
    sub.add_parser("connect", help="Como ligar Claude Code/Cursor/etc.")
    sub.add_parser("help", help="Mostra os comandos e para que servem")
    sub.add_parser("version", help="Mostra a versão")
    return p


def main(argv: "list[str] | None" = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Sem comando -> painel interativo (ótimo para usuários leigos).
    if not argv:
        return interactive_menu()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "pull":
        return cmd_pull(args.repo, args.filename)
    if args.command == "run":
        return cmd_run(args.model, args.prompt, args.max_tokens, args.system, args.profile)
    if args.command == "chat":
        return cmd_chat(args.model, args.max_tokens, args.system, args.profile)
    if args.command == "serve":
        return cmd_serve(args.model, args.port)
    if args.command == "inspect":
        return cmd_inspect(args.path)
    if args.command == "scan":
        return cmd_scan()
    if args.command == "profiles":
        return cmd_profiles()
    if args.command == "connect":
        return cmd_connect()
    if args.command == "version":
        return cmd_version()
    # help (ou comando ausente)
    return cmd_help()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
