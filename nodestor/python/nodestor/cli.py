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
    ("run",     "nodestor run \"<prompt>\" --model <arquivo>", "Roda um prompt num modelo local e mostra TTFT e tok/s reais"),
    ("chat",    "nodestor chat --model <arquivo>",              "Conversa interativa contínua com o modelo"),
    ("serve",   "nodestor serve --model <arquivo>",             "Sobe o servidor (API OpenAI/Anthropic/Ollama) na porta 8080"),
    ("scan",    "nodestor scan",                                "Detecta GPU, VRAM, SSD e o melhor caminho SSD→GPU"),
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


def cmd_run(model: str, prompt: str, max_tokens: int) -> int:
    header("Inferência Local", model)
    if not _require_engine():
        return 1
    from . import NodeStorEngine

    if not os.path.exists(model):
        out(_c(f"Modelo não encontrado: {model}", "red"))
        return 1
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
        text = engine.generate(prompt, max_tokens)
    except Exception as exc:
        out(_c(f"Erro na geração: {exc}", "red"))
        return 1
    dt = time.time() - t1
    out(text)
    out()
    out(_c("Métricas (medidas):", "bold"))
    out(f"  Tempo de geração : {dt:.2f}s")
    return 0


def cmd_chat(model: str, max_tokens: int) -> int:
    header("Chat Interativo", "digite /sair para encerrar")
    if not _require_engine():
        return 1
    from . import NodeStorEngine

    if not os.path.exists(model):
        out(_c(f"Modelo não encontrado: {model}", "red"))
        return 1
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
            reply = engine.generate(msg, max_tokens)
        except Exception as exc:
            out(_c(f"Erro: {exc}", "red"))
            continue
        out(_c("🤖 ", "orange") + reply)
    out(_c("Até logo!", "dim"))
    return 0


def cmd_serve(model: str, port: int) -> int:
    header("Servidor NodeStor", f"API OpenAI/Anthropic/Ollama • porta {port}")
    import shutil
    import subprocess

    server_bin = shutil.which("nodestor-server")
    if server_bin is None:
        out(_c("Binário 'nodestor-server' não encontrado no PATH.", "yellow"))
        out("  Compile o servidor com: " + _c("cargo build --release -p nodestor-server", "cyan"))
        out("  e adicione ./target/release ao PATH.")
        return 1
    out(_c(f"▶ iniciando servidor em http://localhost:{port} …", "green"))
    try:
        return subprocess.call([server_bin, "--model", model, "--port", str(port)])
    except KeyboardInterrupt:
        return 0


# ─────────────────────────────── painel interativo ─────────────────────────────
def interactive_menu() -> int:
    print_logo()
    options = [
        ("Rodar um prompt (run)", "run"),
        ("Conversar (chat)", "chat"),
        ("Subir o servidor (serve)", "serve"),
        ("Escanear o hardware (scan)", "scan"),
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
        # run / chat / serve precisam de um modelo
        model = input(_c("  caminho do modelo (.gguf): ", "cyan")).strip().strip('"')
        if not model:
            out(_c("  cancelado", "dim"))
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

    pr = sub.add_parser("run", help="Roda um prompt num modelo local")
    pr.add_argument("prompt", help="Texto do prompt")
    pr.add_argument("-m", "--model", required=True, help="Caminho do modelo (.gguf/.safetensors)")
    pr.add_argument("--max-tokens", type=int, default=128)

    pc = sub.add_parser("chat", help="Conversa interativa com o modelo")
    pc.add_argument("-m", "--model", required=True)
    pc.add_argument("--max-tokens", type=int, default=256)

    ps = sub.add_parser("serve", help="Sobe o servidor (API compatível)")
    ps.add_argument("-m", "--model", required=True)
    ps.add_argument("--port", type=int, default=8080)

    sub.add_parser("scan", help="Detecta hardware (GPU/VRAM/SSD)")
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

    if args.command == "run":
        return cmd_run(args.model, args.prompt, args.max_tokens)
    if args.command == "chat":
        return cmd_chat(args.model, args.max_tokens)
    if args.command == "serve":
        return cmd_serve(args.model, args.port)
    if args.command == "scan":
        return cmd_scan()
    if args.command == "version":
        return cmd_version()
    # help (ou comando ausente)
    return cmd_help()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
