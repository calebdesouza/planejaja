"""NodeStor — soberania de IA local.

Pacote Python sobre o motor nativo (Rust + Vulkan): rode qualquer modelo
open-weight, em qualquer máquina, com streaming SSD→GPU e RAG embutido.

A extensão nativa é carregada de forma tolerante: se ela ainda não foi
compilada (ex.: ambiente de desenvolvimento sem `maturin develop`), o pacote
continua importável e a CLI exibe uma mensagem clara em vez de quebrar.
"""

__version__ = "0.1.0"

_ENGINE_IMPORT_ERROR = None
try:
    from ._nodestor import NodeStorEngine, scan  # type: ignore
    _ENGINE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depende do build nativo
    NodeStorEngine = None  # type: ignore
    scan = None  # type: ignore
    _ENGINE_AVAILABLE = False
    _ENGINE_IMPORT_ERROR = exc

__all__ = ["NodeStorEngine", "scan", "__version__", "_ENGINE_AVAILABLE"]
