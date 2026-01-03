import logging
from typing import List, Type, Optional

from .interface import TerminalBackend, TerminalConfig
from .tmux import TmuxBackend
from .pexpect_backend import PexpectBackend

logger = logging.getLogger(__name__)

# Priority order for backends
BACKENDS: List[Type[TerminalBackend]] = [
    TmuxBackend,      # Preferred - supports local attachment
    PexpectBackend,   # Fallback - always available
]


def get_available_backends() -> List[str]:
    """Get list of available backend names."""
    return [b.get_name() for b in BACKENDS if b.is_available()]


def create_terminal_backend(
    config: TerminalConfig,
    preferred_backend: Optional[str] = None
) -> TerminalBackend:
    """
    Create a terminal backend instance.

    Args:
        config: Terminal configuration
        preferred_backend: Name of preferred backend (e.g., 'tmux', 'pexpect')
                          If None, uses first available backend.

    Returns:
        TerminalBackend instance
    """
    # If preferred backend specified, try it first
    if preferred_backend:
        for backend_class in BACKENDS:
            if backend_class.get_name() == preferred_backend:
                if backend_class.is_available():
                    logger.info(f"Using preferred backend: {preferred_backend}")
                    return backend_class(config)
                else:
                    logger.warning(f"Preferred backend '{preferred_backend}' not available")
                    break

    # Find first available backend
    for backend_class in BACKENDS:
        if backend_class.is_available():
            logger.info(f"Using backend: {backend_class.get_name()}")
            return backend_class(config)

    # This shouldn't happen since PexpectBackend is always available
    raise RuntimeError("No terminal backend available")
